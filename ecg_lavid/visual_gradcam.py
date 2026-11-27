"""Score true-class Grad-CAM on the frozen Flowers and CUB intervention cohorts."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import functional as vf

from ecg_lavid.teacher import write_json
from ecg_lavid.train import sha256
from ecg_lavid.train_visual_loss_matrix import LossStudent
from ecg_lavid.visual_interventions import (
    DATASETS, MEAN, MIN_REGION_PIXELS, NEUTRAL_RGB, STD, PairedPixels, VisualDataset,
    different_views, lock_population, paths_for,
)
from ecg_lavid.visual_probe_extract import load_frozen


VIEWS = ("clean", "foreground-only", "background-replaced", "background-only")
ARCHITECTURES = ("resnet18", "mobilenetv2")
EXPECTED_SIZE = {"flowers": 5978, "cub200": 5794}


def save_npz(path, arrays):
    staging = path.with_suffix(".tmp.npz")
    np.savez_compressed(staging, **arrays)
    os.replace(staging, path)


def load_json(path):
    return json.loads(path.read_bytes())


class FinalSpatialLogit:
    def __init__(self, model, layer, classifier):
        self.model = model.eval().requires_grad_(False)
        self.classifier = classifier
        self.handle = layer.register_forward_hook(self.capture)
        self.activation = None
        self.grad_check_max_abs = None
        self.native_shape = None

    def capture(self, _module, _inputs, activation):
        self.activation = activation.detach().requires_grad_(True)
        return self.activation

    def __call__(self, inputs, labels):
        logits = self.model(inputs)
        activation, self.activation = self.activation, None
        true_logits = logits.gather(1, labels[:, None]).squeeze(1)
        gradient, = torch.autograd.grad(true_logits.sum(), activation)
        self.native_shape = list(activation.shape[1:])
        if self.grad_check_max_abs is None:
            # Both author heads use spatial mean pooling followed by a linear classifier.
            expected = self.classifier.weight[labels, :, None, None] / (activation.shape[-2] * activation.shape[-1])
            torch.testing.assert_close(gradient, expected.expand_as(gradient), atol=1e-7, rtol=1e-5)
            self.grad_check_max_abs = float((gradient - expected).abs().max())
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        nonnegative = (weights * activation.detach()).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(nonnegative, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        return cam, logits.detach(), true_logits.detach()


def cam_statistics(cam, mask):
    area = mask.sum(dim=(1, 2, 3))
    area_fraction = area.double() / mask[0].numel()
    empty = area == 0
    ceiling = cam.amax(dim=(1, 2, 3))
    floor = cam.amin(dim=(1, 2, 3))
    total = cam.sum(dim=(1, 2, 3), dtype=torch.float64)
    foreground = cam.masked_fill(~mask, 0).sum(dim=(1, 2, 3), dtype=torch.float64)
    nonfinite = ~(torch.isfinite(ceiling) & torch.isfinite(floor) & torch.isfinite(total) & torch.isfinite(foreground))
    zero = (total == 0) & ~nonfinite
    constant_positive = (floor == ceiling) & (ceiling > 0) & ~nonfinite
    valid = ~(empty | zero | nonfinite)
    fraction = (foreground / total).masked_fill(~(torch.isfinite(total) & (total > 0)), float("nan"))
    q_fg = (fraction / area_fraction).masked_fill(~valid, float("nan"))
    q_fg = torch.where(valid & constant_positive, 1.0, q_fg)
    return {
        "q_fg": q_fg, "foreground_mass_fraction": fraction, "foreground_mass": foreground, "cam_mass": total,
        "cam_peak": ceiling, "is_zero": zero, "is_constant_positive": constant_positive,
        "is_nonfinite": nonfinite, "is_empty_mask": empty, "is_valid": valid,
    }


@dataclass
class ModelRun:
    cam: FinalSpatialLogit
    name: str
    checkpoint: Path
    checkpoint_sha256: str
    reference: Path
    reference_sha256: str
    target_layer: str
    model_source_sha256: str
    reference_scores: np.ndarray
    observations: dict
    examples: dict
    comparison: dict


def load_model(args, arch, arm, checkpoint, prior, base, manifest_sha, device):
    checkpoint_sha, prior_sha = sha256(checkpoint), sha256(prior)
    stored = load_json(prior.with_suffix(".json"))
    if (stored["checkpoint_sha256"] != checkpoint_sha or stored["prediction_sha256"] != prior_sha
            or stored["manifest_sha256"] != manifest_sha or stored["checkpoint_epoch"] != 240
            or stored["dataset"] != args.dataset or stored["scored_records"] != len(base["image_id"])):
        raise ValueError(f"Reference prediction provenance differs for {arch}/{arm}")
    with np.load(prior, allow_pickle=False) as saved:
        for key, source_key in (("image_id", "image_id"), ("donor_image_id", "donor_image_id"), ("true_label", "y_true")):
            np.testing.assert_array_equal(base[key], saved[source_key], err_msg=f"Reference {arch}/{arm}/{key}")
        expected_views = saved["variants"].tolist()
        expected_scores = saved["log_probability"][[expected_views.index(view) for view in VIEWS]]
        support_name = saved["pixel_support_names"].tolist().index("receiver_foreground_pixels")
        np.testing.assert_array_equal(base["area_pixels"], saved["pixel_support_counts"][:, support_name])
    if arch == "resnet18":
        model, state, source = load_frozen(args.author_source, checkpoint, args.dataset, device)
        module, layer, classifier = model.layer4, "layer4", model.fc
    else:
        source = args.author_source / "lavid/models/mobilenetv2.py"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = LossStudent(arch, DATASETS[args.dataset][0], args.author_source)
        model.load_state_dict(state["student"], strict=True)
        model = model.to(device).eval()
        module = model.encoder.features[-1]
        layer = f"encoder.features.{len(model.encoder.features) - 1}"
        classifier = model.classifier
    if state["epoch"] != 240 or checkpoint.parent.parent.name != "seed-17":
        raise ValueError(f"Expected the completed seed-17 epoch-240 checkpoint: {checkpoint}")
    n = len(base["image_id"])
    arrays = {key: np.full((n, len(VIEWS)), np.nan, dtype=np.float64) for key in (
        "q_fg", "foreground_mass_fraction", "foreground_mass", "cam_mass", "cam_peak",
        "true_logit", "true_log_probability",
    )}
    arrays.update({key: np.zeros((n, len(VIEWS)), dtype=np.bool_) for key in (
        "is_zero", "is_constant_positive", "is_nonfinite", "is_empty_mask", "is_valid",
    )})
    arrays["pred_label"] = np.full((n, len(VIEWS)), -1, dtype=np.int64)
    num_examples = len(args.example_ids)
    examples = {
        "cam": np.empty((num_examples, len(VIEWS), 224, 224), np.float32),
        "rgb_float": np.empty((num_examples, len(VIEWS), 224, 224, 3), np.float32),
        "fg_mask": np.empty((num_examples, 224, 224), np.bool_),
    }
    comparison = {name: {"rows": 0, "argmax_disagreements": 0,
                         "max_abs_log_probability_difference": 0.0, "exact_vector_matches": 0}
                  for name in VIEWS}
    return ModelRun(FinalSpatialLogit(model, module, classifier), f"{arch}/{arm}", checkpoint, checkpoint_sha,
                    prior, prior_sha, layer, sha256(source), expected_scores, arrays, examples, comparison)


def collect_batch(run, positions, view_index, view, inputs, mask, labels, example_rows):
    cam, logits, true_logits = run.cam(inputs, labels)
    values = cam_statistics(cam, mask)
    log_probs = logits.log_softmax(dim=1)
    if not torch.isfinite(log_probs).all():
        raise ValueError(f"Nonfinite frozen classification scores for {run.name}")
    true_log_prob = log_probs.gather(1, labels[:, None]).squeeze(1)
    observed = log_probs.cpu().numpy()
    reference = run.reference_scores[view_index, positions]
    difference = np.abs(observed - reference)
    predicted = observed.argmax(axis=1)
    comparison = run.comparison[VIEWS[view_index]]
    comparison["rows"] += len(predicted)
    comparison["argmax_disagreements"] += int(np.count_nonzero(predicted != reference.argmax(axis=1)))
    comparison["exact_vector_matches"] += int(np.count_nonzero((difference == 0).all(axis=1)))
    comparison["max_abs_log_probability_difference"] = max(comparison["max_abs_log_probability_difference"],
                                                          float(difference.max()))
    for key, array in values.items():
        run.observations[key][positions, view_index] = array.cpu().numpy()
    run.observations["pred_label"][positions, view_index] = predicted
    run.observations["true_logit"][positions, view_index] = true_logits.cpu().numpy()
    run.observations["true_log_probability"][positions, view_index] = true_log_prob.cpu().numpy()
    if example_rows:
        batch_rows, target_rows = zip(*example_rows, strict=True)
        example_cam = cam[list(batch_rows), 0]
        peak = values["cam_peak"][list(batch_rows)].reshape(-1, 1, 1)
        plot_cam = torch.where(torch.isfinite(peak) & (peak > 0), example_cam / peak, example_cam)
        run.examples["cam"][list(target_rows), view_index] = plot_cam.cpu().numpy()
        run.examples["rgb_float"][list(target_rows), view_index] = view[list(batch_rows)].permute(0, 2, 3, 1).cpu().numpy()
        run.examples["fg_mask"][list(target_rows)] = mask[list(batch_rows), 0].cpu().numpy()


def publish(run, args, base, example_indices, scored, manifest_sha):
    directory = args.out / args.dataset / run.name
    directory.mkdir(parents=True, exist_ok=True)
    examples_path = directory / "examples.npz"
    if not example_indices and examples_path.exists():
        raise ValueError(f"This directory already contains display examples. Supply their IDs or use a fresh output directory: {directory}")
    constants = {"views": np.array(VIEWS), "layer_name": np.array(run.target_layer)}
    rows = {key: values[:scored] for key, values in (base | run.observations).items()}
    save_npz(directory / "per-image.npz", rows | constants)
    if example_indices:
        examples = {key: values[example_indices] for key, values in rows.items()}
        rgb = np.rint(run.examples["rgb_float"] * 255).astype(np.uint8)
        save_npz(examples_path, examples | run.examples | constants | {"rgb": rgb})
    stats = {}
    for index, name in enumerate(VIEWS):
        stats[name] = {key: int(rows[key][:, index].sum()) for key in (
            "is_valid", "is_zero", "is_constant_positive", "is_nonfinite", "is_empty_mask",
        )}
        stats[name]["correct"] = int((rows["pred_label"][:, index] == rows["true_label"]).sum())
        q = rows["q_fg"][rows["is_valid"][:, index], index]
        stats[name]["mean_q_fg"] = float(q.mean()) if len(q) else None
    source = directory / "per-image.npz"
    write_json(directory / "summary.json", {
        "dataset": args.dataset, "model": run.name, "seed": 17, "epoch": 240,
        "eligible_images": len(base["image_id"]), "scored_records": scored, "view_order": list(VIEWS),
        "target": "true-class pre-softmax classification logit",
        "target_layer": run.target_layer, "activation_channels_height_width": run.cam.native_shape,
        "spatial_grad_vs_analytic_max_abs": run.cam.grad_check_max_abs,
        "heatmap": "ReLU of the channel-weighted activation sum, with spatial mean gradients as weights, then bilinear to 224×224 (align_corners=False)",
        "q_fg": "foreground share of raw CAM mass divided by original foreground pixel area share; a positive constant map gives 1",
        "degenerate": "zero or nonfinite CAM mass, or an empty mask, yields NaN Q_fg",
        "background_only_mask": "original receiver foreground footprint, retained for metric calculation after pixels were removed",
        "precision": "FP32 without TF32; raw CAM mass sums use float64",
        "example_rgb": "rgb_float is the exact pre-normalization RGB input; rgb is the nearest display byte, including 0.5 fills",
        "example_cam": "each upsampled map is divided by its positive maximum only; cam_peak stores that raw maximum",
        "neutral_replacement_rgb": NEUTRAL_RGB, "checkpoint": str(run.checkpoint),
        "checkpoint_sha256": run.checkpoint_sha256, "author_model_sha256": run.model_source_sha256,
        "prediction_reference": str(run.reference), "prediction_reference_sha256": run.reference_sha256,
        "manifest_sha256": manifest_sha, "compute_source_sha256": sha256(Path(__file__)),
        "sample_selection_sha256": sha256(args.examples) if args.examples else None,
        "classification_reference_check": run.comparison, "view_results": stats,
        "output_sha256": {source.name: sha256(source), **({examples_path.name: sha256(examples_path)} if example_indices else {})},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, help="JSON mapping each dataset to four display image IDs")
    for name in ("images", "masks", "manifest", "author-source", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=tuple(EXPECTED_SIZE))
    parser.add_argument("--model", required=True, action="append", nargs=4, metavar=("ARCH", "ARM", "CHECKPOINT", "REFERENCE"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=5, type=int)
    args = parser.parse_args()
    model_keys = [(arch, arm) for arch, arm, _, _ in args.model]
    requested = {arch for arch, _ in model_keys}
    expected = {(arch, arm) for arch in requested for arm in ("A", "H")}
    if not requested.issubset(ARCHITECTURES) or set(model_keys) != expected or len(model_keys) != len(expected):
        parser.error("Supply one A and one H checkpoint per architecture (resnet18 or mobilenetv2)")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Use positive batch counts and a nonnegative worker count")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    begin = time.monotonic()
    manifest_sha = sha256(args.manifest)
    frozen = load_json(args.manifest)
    images = VisualDataset(args.images, args.dataset, False)
    mask_paths = paths_for(images, args.dataset, args.masks)
    selected, donors = lock_population(args.dataset, images, mask_paths, args.manifest)
    if len(selected) != EXPECTED_SIZE[args.dataset]:
        raise ValueError(f"Unexpected frozen mask count: {len(selected)}")
    image_ids = images.ids[selected]
    counts = np.array([frozen["foreground_pixels_by_image"][str(i)] for i in image_ids], dtype=np.int64)
    if ((counts < MIN_REGION_PIXELS) | (counts > 224**2 - MIN_REGION_PIXELS)).any():
        raise ValueError("A frozen recipient is outside the published mask eligibility bounds")
    base = {"image_id": image_ids, "donor_image_id": images.ids[donors], "true_label": images.labels[selected],
            "area_pixels": counts, "area_fraction": counts / 224**2}
    args.example_ids = load_json(args.examples)[args.dataset] if args.examples else []
    if args.examples and (len(args.example_ids) != 4 or len(set(args.example_ids)) != 4 or not set(args.example_ids) <= set(image_ids)):
        raise ValueError("Expected all four frozen example IDs in the eligible cohort")
    depths = {int(image_id): index for index, image_id in enumerate(image_ids)}
    example_indices = [depths[image_id] for image_id in args.example_ids]
    example_by_row = {position: index for index, position in enumerate(example_indices)}
    models = [load_model(args, arch, arm, Path(checkpoint), Path(reference), base, manifest_sha, device)
              for arch, arm, checkpoint, reference in args.model]
    print(f"{args.dataset}: frozen {len(image_ids)} originals, {len(models)} models, example IDs {args.example_ids}", flush=True)
    dataset = PairedPixels(args.dataset, images, mask_paths, selected, donors)
    batches = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    scored = 0
    for number, (rgb, mask, donor, donor_mask) in enumerate(batches, start=1):
        start, stop = scored, scored + len(rgb)
        row_slice = slice(start, stop)
        example_rows = [(position - start, example_by_row[position]) for position in example_by_row if start <= position < stop]
        np.testing.assert_array_equal(mask.sum(dim=(1, 2, 3)).numpy(), counts[row_slice])
        rgb = vf.to_dtype(rgb.to(device, non_blocking=True), torch.float32, scale=True)
        donor = vf.to_dtype(donor.to(device, non_blocking=True), torch.float32, scale=True)
        mask, donor_mask = mask.to(device, non_blocking=True), donor_mask.to(device, non_blocking=True)
        labels = torch.from_numpy(base["true_label"][row_slice]).to(device)
        for name, view in different_views(rgb, mask, donor, donor_mask):
            if name in VIEWS:
                normalized = vf.normalize(view, MEAN, STD).contiguous()
                for run in models:
                    collect_batch(run, row_slice, VIEWS.index(name), view, normalized, mask, labels, example_rows)
        scored = stop
        if number == 1 or number % 10 == 0 or scored == len(image_ids):
            mismatches = sum(view["argmax_disagreements"] for run in models for view in run.comparison.values())
            elapsed = time.monotonic() - begin
            print(f"{args.dataset}: {scored}/{len(image_ids)} images, {elapsed:.1f}s, {mismatches} prior argmax disagreements", flush=True)
    if scored != len(image_ids):
        raise ValueError("The frozen full cohort was not scored")
    for run in models:
        publish(run, args, base, example_indices, scored, manifest_sha)
        run.cam.handle.remove()
    print(f"{args.dataset}: complete ({scored} images per group in {time.monotonic() - begin:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
