"""Score sealed PTB-XL loss controls with shared interior missing-waveform windows."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import CLASSES, LEADS
from .ptb_crossmodal_eval import PERTURBATION_SEED, changed_signal, parameters
from .train import cohort_splits, load_data, sha256
from .train_ptb_crossmodal import ARCHITECTURES, SEEDS, batch_options, build_network, freeze_json, hash_values, metrics, run_directory


ARMS = {"A": "A", "B": "raw-onehot", "C": "prob-onehot", "D": "ce-same-onehot", "E": "squentropy",
        "F": "ce-independent-onehot", "G": "ecoc-soft-ce", "Gstar": "ecoc-soft-ce-gradmatch"}
PAIRING = ("epochs", "batch_size", "lr", "weight_decay", "initial_state_sha256", "training_order_sha256",
           "normalization", "train_records", "train_patients")


def sealed_runs(args):
    for architecture in args.architectures:
        for seed in args.seeds:
            baseline = run_directory(args.baselines, architecture, 1.0, seed, "A")
            reference = json.loads((baseline / "config.json").read_bytes())
            reference_seal = json.loads((baseline / "seal.json").read_bytes())
            for letter in args.arms:
                arm = ARMS[letter]
                source = baseline if letter == "A" else run_directory(args.runs, architecture, 1.0, seed, arm)
                config = json.loads((source / "config.json").read_bytes())
                if any(config.get(key) != reference.get(key) for key in PAIRING):
                    raise ValueError(f"A paired training input or initialization disagrees at {source}")
                seal = json.loads((source / "seal.json").read_bytes())
                checkpoint = source / "best.pt"
                if (seal["epochs"] != reference_seal["epochs"] or seal["steps"] != reference_seal["steps"] or
                        sha256(checkpoint) != seal["best_sha256"]):
                    raise ValueError(f"The validation-selected checkpoint is not sealed at {source}")
                yield architecture, seed, letter, checkpoint, seal


@torch.inference_mode()
def score(args, architecture, checkpoint, signals, centers, interpolated=None):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_network(architecture)
    model.question_head = None
    model.load_state_dict({key: value for key, value in state["state"].items() if not key.startswith("question_head.")})
    model = model.to(args.device).eval()
    mean = state["mean"].to(args.device)
    scale = state["std"].to(args.device)
    normalized = (signals - mean) / scale
    time = torch.arange(signals.shape[-1], device=args.device)
    predictions = []
    for width in (0, *args.mask_samples):
        rows = []
        view = (interpolated[width] - mean) / scale if interpolated is not None and width else normalized
        for batch, midpoints in zip(view.split(args.batch_size), centers.split(args.batch_size), strict=True):
            if interpolated is None and width:
                start = midpoints[:, None] - width // 2
                region = (time >= start) & (time < start + width)
                batch = batch.masked_fill(region[:, None, :], 0)
            with torch.autocast(**batch_options(args)):
                rows.append(model(batch).float().cpu())
        predictions.append(torch.cat(rows).numpy())
    return np.stack(predictions)


def summarize(logits, truth, widths):
    decisions = logits.argmax(axis=-1)
    clean_correct = decisions[0] == truth
    rows = []
    for width, prediction in zip(widths, decisions, strict=True):
        rows.append(metrics(truth, prediction) | {"window_samples": width, "window_ms": width * 10,
                    "retained_initially_correct": float(np.mean(prediction[clean_correct] == truth[clean_correct]))
                    if clean_correct.any() else None})
    f1 = np.asarray([r["macro_f1"] for r in rows])
    positions = np.asarray(widths) / widths[-1]
    auc = float(((f1[:-1] + f1[1:]) * .5 * np.diff(positions)).sum())
    return {"levels": rows, "macro_f1_auc_actual_strength": auc,
            "relative_macro_f1": (f1 / f1[0]).tolist() if f1[0] > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "baselines", "runs", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--fold", choices=(9, 10), type=int, default=10)
    parser.add_argument("--missing", choices=("interpolation", "mask"), default="interpolation")
    parser.add_argument("--mask-samples", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=[17])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.mask_samples != sorted(set(args.mask_samples)) or args.mask_samples[0] <= 0:
        parser.error("Mask lengths must be unique positive numbers of samples in increasing order")
    torch.set_num_threads(4)
    models = list(sealed_runs(args))
    dataset, ids, patients, folds = load_data(args.data)
    signals, labels = dataset.tensors
    split, _ = cohort_splits(folds, patients, labels.numpy())
    indices = split["validation" if args.fold == 9 else "test"]
    ids, patients, truth = ids[indices], patients[indices], labels[indices].numpy()
    raw = signals[indices].numpy()
    signals = torch.as_tensor(raw, device=args.device)
    descriptors = parameters(ids, signals.shape[-1])
    centers = descriptors[:, 3].astype(np.int64)
    widths = (0, *args.mask_samples)
    if np.any(centers - widths[-1] // 2 < 0) or np.any(centers - widths[-1] // 2 + widths[-1] > signals.shape[-1]):
        raise ValueError("A specified window would not fit the frozen interior center")
    interpolated = None
    if args.missing == "interpolation":
        interpolated = {width: torch.as_tensor(changed_signal(raw, "gap", width, descriptors), device=args.device)
                        for width in args.mask_samples}
    args.out.mkdir(parents=True, exist_ok=True)
    freeze_json(args.out / "protocol.json", {"fold": args.fold, "sample_rate_hz": 100, "seed": PERTURBATION_SEED,
                "class_order": list(CLASSES), "leads": list(LEADS), "window_samples": list(widths),
                "ordered_ecg_ids_sha256": hash_values(ids.astype("<i8")),
                "mechanism": ("linear interpolation in raw mV using the two surrounding samples"
                              if interpolated is not None else "set to zero after train-only normalization"),
                "alignment": "one shared interior window across all leads in each record",
                "center": "same deterministic per-record center as the archived interpolation-gap protocol"})
    midpoints = torch.as_tensor(centers, device=args.device)
    all_rows = []
    for architecture, seed, letter, checkpoint, seal in models:
        destination = args.out / architecture / f"seed-{seed}" / letter
        destination.mkdir(parents=True, exist_ok=True)
        meta = {"fold": args.fold, "architecture": architecture, "seed": seed, "arm": letter,
                "selected_epoch": seal["best_epoch"], "selected_sha256": seal["best_sha256"], "missing": args.missing}
        array_path = destination / "predictions.npz"
        result_path = destination / "results.json"
        if result_path.is_file():
            report = json.loads(result_path.read_bytes())
            if any(report[key] != value for key, value in meta.items()) or sha256(array_path) != report["predictions_sha256"]:
                raise ValueError(f"The saved result disagrees with this frozen comparison: {result_path}")
        else:
            logits = score(args, architecture, checkpoint, signals, midpoints, interpolated)
            if not np.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite heldout logits at {checkpoint}")
            temporary = array_path.with_suffix(".part")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, ecg_id=ids, patient_id=patients, y_true=truth, logits=logits,
                                    window_center=centers, window_samples=np.asarray(widths), class_names=np.asarray(CLASSES))
            temporary.replace(array_path)
            report = meta | summarize(logits, truth, widths) | {"predictions_sha256": sha256(array_path)}
            freeze_json(result_path, report)
        all_rows.extend(meta | {key: level[key] for key in ("window_ms", "accuracy", "macro_f1", "balanced_accuracy", "retained_initially_correct")}
                        for level in report["levels"])
        print(f"{architecture} {seed} {letter} fold {args.fold}: clean macro F1={report['levels'][0]['macro_f1']:.4f}", flush=True)
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
