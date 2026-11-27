"""Run identical paired pixel interventions on the author test images and released masks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as vf

from .teacher import write_json
from .train import sha256
from .train_visual_lavid import DATASETS, VisualDataset
from .visual_probe_extract import load_frozen


VARIANTS = ("clean", "grayscale", "foreground-gray", "silhouette", "foreground-only",
            "background-only", "background-replaced", "donor-hole-control", "sham-background")
NEUTRAL_RGB = .5
MIN_REGION_PIXELS = 64
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
SUPPORT_NAMES = ("receiver_foreground_pixels", "donor_hole_outside_receiver",
                 "donor_background_exposed", "pixel_differences_swap_vs_equal_hole")
MASK_METHOD = {"flowers": "Oxford segmented JPEG saturated-blue background threshold R,G≤45 and B≥200",
               "cub200": "Caltech official PNG, grayscale≥128 is foreground"}


def crop_rgb(path):
    with Image.open(path) as image:
        tensor = vf.to_image(image.convert("RGB"))
    return vf.center_crop(vf.resize(tensor, [256], antialias=True), [224, 224])


def mask_index(dataset, directory):
    result = {}
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if dataset == "flowers":
            numeral = p.stem.split("_")[-1]
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or not numeral.isdecimal():
                continue
            key = int(numeral)
        else:
            if p.suffix.lower() != ".png":
                continue
            key = (p.parent.name, p.stem)
        if key in result:
            raise ValueError(f"Duplicate released segmentation for {key}")
        result[key] = p
    if not result:
        raise ValueError(f"No released segmentation images found under {directory}")
    return result


def mask_for(dataset, p, source):
    with Image.open(p) as mask, Image.open(source) as image:
        if mask.size != image.size:
            raise ValueError(f"Mask and RGB image sizes differ: {p.name}, {source.name}")
        if dataset == "flowers":
            rgb = np.asarray(mask.convert("RGB"))
            # Oxford publishes the segmented image with a saturated blue background as JPEG.
            foreground = ~((rgb[..., 0] <= 45) & (rgb[..., 1] <= 45) & (rgb[..., 2] >= 200))
        else:
            foreground = np.asarray(mask.convert("L")) >= 128
    tensor = torch.from_numpy(foreground.copy()).unsqueeze(0).to(torch.uint8)
    return vf.center_crop(vf.resize(tensor, [256], interpolation=InterpolationMode.NEAREST_EXACT,
                                    antialias=False), [224, 224]).bool()


def paths_for(input_set, name, root):
    catalogue = mask_index(name, root)
    if name == "flowers":
        return [catalogue.get(int(index)) for index in input_set.ids]
    return [catalogue.get((path.parent.name, path.stem)) for path in input_set.paths]


def eligible_collection(images, name, mask_paths):
    selected = []
    counts = {}
    for index, (source, mask) in enumerate(zip(images.paths, mask_paths, strict=True)):
        if mask is None:
            counts[int(images.ids[index])] = 0
            continue
        fg = int(mask_for(name, mask, source).sum())
        counts[int(images.ids[index])] = fg
        if MIN_REGION_PIXELS <= fg <= 224 ** 2 - MIN_REGION_PIXELS:
            selected.append(index)
    return selected, counts


def donor_list(images, selected):
    ordered = sorted(selected, key=lambda i: (int(images.labels[i]), hashlib.sha256(
        f"flower-cub-background-20261126|{int(images.ids[i])}".encode()).digest()))
    classes = images.labels[ordered]
    shift = int(np.bincount(classes).max())
    if shift * 2 > len(ordered):
        raise ValueError("No different-species cyclic background matching exists")
    donors = np.roll(ordered, -shift)
    if np.any(classes == images.labels[donors]):
        raise ValueError("A donor shares the recipient species")
    mapping = dict(zip(ordered, donors.tolist(), strict=True))
    return [mapping[index] for index in selected]


def lock_population(name, images, mask_paths, manifest):
    if manifest.exists():
        data = json.loads(manifest.read_bytes())
        if (data["dataset"] != name or data["all_image_ids"] != images.ids.tolist() or
                data["minimum_region_pixels_after_crop"] != MIN_REGION_PIXELS or data["mask_decoder"] != MASK_METHOD[name]):
            raise ValueError("Intervention manifest belongs to a different dataset or split order")
        selection, donors = data["selected_indices"], data["donor_indices"]
        if (images.ids[selection].tolist() != data["image_ids"] or images.labels[selection].tolist() != data["labels"]
                or images.ids[donors].tolist() != data["donor_image_ids"] or donors != donor_list(images, selection)):
            raise ValueError("Already frozen image, label or background donor mapping changed")
        for index, signature in zip(selection, data["selected_mask_sha256"], strict=True):
            if sha256(mask_paths[index]) != signature:
                raise ValueError(f"Official segmentation bytes changed for image {images.ids[index]}")
        return selection, donors
    selection, counts = eligible_collection(images, name, mask_paths)
    if not selection:
        raise ValueError("No images have both sufficient foreground and background")
    donors = donor_list(images, selection)
    write_json(manifest, {"dataset": name, "all_image_ids": images.ids.tolist(), "selected_indices": selection,
                          "image_ids": images.ids[selection].tolist(), "labels": images.labels[selection].tolist(),
                          "selected_mask_sha256": [sha256(mask_paths[i]) for i in selection],
                          "foreground_pixels_by_image": counts,
                          "donor_indices": donors, "donor_image_ids": images.ids[donors].tolist(),
                          "minimum_region_pixels_after_crop": MIN_REGION_PIXELS,
                          "mask_decoder": MASK_METHOD[name]})
    return selection, donors


class PairedPixels(Dataset):
    def __init__(self, name, images, mask_paths, selected, donors):
        self.name, self.images, self.mask_paths = name, images, mask_paths
        self.selected, self.donors = selected, donors

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, k):
        source, donor = self.selected[k], self.donors[k]
        current_path, donor_path = self.images.paths[source], self.images.paths[donor]
        current_mask = mask_for(self.name, self.mask_paths[source], current_path)
        donor_mask = mask_for(self.name, self.mask_paths[donor], donor_path)
        return crop_rgb(current_path), current_mask, crop_rgb(donor_path), donor_mask


def different_views(clean, mask, donor, donor_mask):
    gray = vf.rgb_to_grayscale(clean, num_output_channels=3)
    donor_background = torch.where(donor_mask, NEUTRAL_RGB, donor)
    self_background = torch.where(mask, NEUTRAL_RGB, clean)
    yield "clean", clean
    yield "grayscale", gray
    yield "foreground-gray", torch.where(mask, gray, clean)
    yield "silhouette", torch.where(mask, .8, .2).expand_as(clean)
    yield "foreground-only", torch.where(mask, clean, NEUTRAL_RGB)
    yield "background-only", self_background
    yield "background-replaced", torch.where(mask, clean, donor_background)
    yield "donor-hole-control", torch.where(mask, clean, torch.where(donor_mask, NEUTRAL_RGB, clean))
    yield "sham-background", torch.where(mask, clean, self_background)


@torch.inference_mode()
def score(model, batches, device):
    chunks = {key: [] for key in VARIANTS}
    supports = []
    for image, mask, donor, donor_mask in batches:
        image = vf.to_dtype(image.to(device, non_blocking=True), torch.float32, scale=True)
        donor = vf.to_dtype(donor.to(device, non_blocking=True), torch.float32, scale=True)
        mask, donor_mask = mask.to(device, non_blocking=True), donor_mask.to(device, non_blocking=True)
        outside = ~mask & ~donor_mask
        supports.append(torch.stack([region.sum((1, 2, 3)) for region in (
            mask, ~mask & donor_mask, outside, outside & (image != donor).any(dim=1, keepdim=True)
        )], dim=1).cpu().numpy())
        for key, view in different_views(image, mask, donor, donor_mask):
            if key == "sham-background":
                if not torch.equal(image, view):
                    raise ValueError("Sham composition produced a changed RGB pixel")
                chunks[key].append(chunks["clean"][-1])
                continue
            logits = model(vf.normalize(view, MEAN, STD).contiguous())
            chunks[key].append(logits.log_softmax(dim=-1).cpu().numpy())
    return np.stack([np.concatenate(chunks[key]) for key in VARIANTS]), np.concatenate(supports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "masks", "manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("author-source", "checkpoint", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--architecture", choices=("resnet18", "mobilenetv2", "vit_small_patch16"), default="resnet18")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not args.manifest_only and (not (args.checkpoint and args.out and args.author_source) or args.out.suffix != ".npz"):
        parser.error("Model scoring requires a checkpoint, author source, and NPZ output")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Use an NPZ output and valid batch/worker counts")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    images = VisualDataset(args.images, args.dataset, False)
    mask_paths = paths_for(images, args.dataset, args.masks)
    ids, donors = lock_population(args.dataset, images, mask_paths, args.manifest)
    if args.manifest_only:
        print(f"Frozen identical recipients and donors for {len(ids)} of {len(images)} images", flush=True)
        return
    device = torch.device(args.device)
    if args.architecture == "resnet18":
        model, state, source = load_frozen(args.author_source, args.checkpoint, args.dataset, device)
        backbone = {"resnet_source_sha256": sha256(source)}
    else:
        from .train_visual_loss_matrix import LossStudent

        state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        model = LossStudent(args.architecture, DATASETS[args.dataset][0], args.author_source)
        model.load_state_dict(state["student"], strict=True)
        model = model.to(device).eval()
        backbone = {"architecture": args.architecture}
    data = PairedPixels(args.dataset, images, mask_paths, ids, donors)
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    logits, supports = score(model, loader, device)
    labels = images.labels[ids]
    accuracy = (logits.argmax(axis=-1) == labels).mean(axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp.npz")
    np.savez(temporary, variants=np.array(VARIANTS), image_id=images.ids[ids], y_true=labels,
             donor_image_id=images.ids[donors], log_probability=logits,
             pixel_support_names=np.array(SUPPORT_NAMES), pixel_support_counts=supports)
    temporary.replace(args.out)
    write_json(args.out.with_suffix(".json"), {
        "dataset": args.dataset, "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_epoch": state["epoch"],
        **backbone, "manifest_sha256": sha256(args.manifest),
        "prediction_sha256": sha256(args.out), "scored_records": len(labels), "all_records": len(images),
        "precision": "FP32, no TF32 convolution", "foreground_neutral_replacement_rgb": NEUTRAL_RGB,
        "top1": dict(zip(VARIANTS, accuracy.tolist())),
        "relative_drop_vs_clean": dict(zip(VARIANTS, (1 - accuracy / accuracy[0]).tolist())) if accuracy[0] else None,
        "per_image_pixel_support_median": dict(zip(SUPPORT_NAMES, np.median(supports, axis=0).tolist())),
        "per_image_pixel_support_minimum": dict(zip(SUPPORT_NAMES, np.min(supports, axis=0).tolist())),
        "mask": MASK_METHOD[args.dataset],
    })


if __name__ == "__main__":
    main()
