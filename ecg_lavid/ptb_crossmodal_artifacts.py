"""Score frozen PTB-XL classifiers under the separately specified acquisition artifacts."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch

from .teacher import write_json
from .train import load_data, sha256
from .train_ptb_crossmodal import ARCHITECTURES, ARMS, FRACTIONS, SEEDS, batch_options, freeze_json, infer, metrics, run_directory
from .ptb_crossmodal_eval import severity_auc, state_and_stats, validate_seals


SEED = 923211
LEVELS = {"baseline": (.025, .05, .1), "emg": (.0125, .025, .05)}


def noise_bases(ids, channel_count, width):
    seconds = np.arange(width, dtype=np.float64) / 100.
    baseline = np.empty((len(ids), channel_count, width), dtype=np.float32)
    emg = np.empty_like(baseline)
    band = butter(3, (25, 45), btype="bandpass", fs=100., output="sos")
    for i, ecg_id in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, int(ecg_id)]))
        freq = rng.uniform(.15, .40)
        phase = rng.uniform(0., 2. * np.pi) + rng.uniform(-.35, .35, size=(channel_count, 1))
        baseline[i] = np.sin(2. * np.pi * freq * seconds + phase)
        noise = sosfiltfilt(band, rng.standard_normal((channel_count, width)), axis=1)
        noise -= noise.mean(axis=1, keepdims=True)
        emg[i] = noise / np.sqrt(np.mean(np.square(noise), axis=1, keepdims=True))
    return {"baseline": baseline, "emg": emg}


def score(args, run, views, ids, truth):
    train_dir = run_directory(args.runs, *run)
    clean_dir = run_directory(args.clean, *run)
    out = run_directory(args.out, *run)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "artifact-results.json"
    if result_path.is_file():
        result = json.loads(result_path.read_bytes())
        if sha256(out / "artifact-predictions.npz") != result["predictions_sha256"]:
            raise ValueError(f"Saved predictions changed in {out}")
        return result
    reference = json.loads((clean_dir / "results.json").read_bytes())
    original = clean_dir / "test-predictions.npz"
    if sha256(original) != reference["predictions_sha256"]:
        raise ValueError(f"Original reference predictions changed under {clean_dir}")
    with np.load(original, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["ecg_id"], ids)
        np.testing.assert_array_equal(saved["y_true"], truth)
        assert saved["conditions"][0] == "clean"
        hit = saved["logits"][0].argmax(axis=1) == truth
    clean_f1 = reference["test_fold10"]["clean"]["macro_f1"]
    model, norm, checkpoint = state_and_stats(train_dir / "best.pt", run[0], args.device)
    if checkpoint["epoch"] != reference["selected_epoch"]:
        raise ValueError("Acquisition-artifact evaluation must use the original selected checkpoint")
    position = torch.arange(len(ids), device=args.device)
    robustness, cache = {}, []
    for condition, inputs in views.items():
        logits = infer(model, inputs, position, norm, 256, batch_options(args))
        if not np.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite perturbation outputs for {run} {condition}")
        prediction = logits.argmax(axis=1)
        values = metrics(truth, prediction)
        robustness[condition] = values | {"macro_f1_retention": values["macro_f1"] / clean_f1,
                                          "initial_correct_retention": float((prediction[hit] == truth[hit]).mean())}
        cache.append(logits)
    curves = {}
    for name in LEVELS:
        family = [robustness[f"{name}_{k}"] for k in range(1, 4)]
        curves[name] = {f"auc_{field}": severity_auc([clean] + [x[field] for x in family]) for field, clean in
                        (("macro_f1", clean_f1), ("macro_f1_retention", 1.), ("initial_correct_retention", 1.))}
    worst = min(robustness, key=lambda name: robustness[name]["macro_f1"])
    path = out / "artifact-predictions.npz"
    temp = out / "artifact-predictions.part.npz"
    np.savez_compressed(temp, ecg_id=ids, y_true=truth, conditions=np.asarray(list(views)), logits=np.stack(cache))
    temp.replace(path)
    result = {"selected_epoch": checkpoint["epoch"], "clean_macro_f1": clean_f1, "curves": curves, "conditions": robustness,
              "worst": {"name": worst} | robustness[worst], "predictions_sha256": sha256(path)}
    write_json(result_path, result)
    return result


def save_summary(args, outcomes):
    rows = []
    for architecture in ARCHITECTURES:
        for fraction in FRACTIONS:
            for name in LEVELS:
                for level in (1, 2, 3):
                    key = f"{name}_{level}"
                    for metric in ("macro_f1", "macro_f1_retention", "initial_correct_retention"):
                        a, h = (np.array([outcomes[architecture, fraction, seed, arm]["conditions"][key][metric] for seed in SEEDS])
                                for arm in ARMS)
                        rows.append({"architecture": architecture, "fraction": fraction, "family": name, "severity": level,
                                     "metric": metric, "A_mean": a.mean(), "H_mean": h.mean(), "H_minus_A": (h - a).mean(),
                                     "positive_pairs": int((h > a).sum()), "seed_deltas": "|".join(map(str, h - a))})
    with (args.out / "artifact-summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "clean", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--deadline", type=lambda stamp: datetime.fromisoformat(stamp).timestamp(), default=float("inf"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    runs = [(a, f, s, r) for a in ARCHITECTURES for f in FRACTIONS for s in SEEDS for r in ARMS]
    validate_seals(args, runs)
    freeze_json(args.out / "artifact-protocol.json", {"seed": SEED, "sample_rate_hz": 100,
                "baseline_peak_mv": list(LEVELS["baseline"]), "emg_rms_mv": list(LEVELS["emg"]),
                "baseline_frequency_hz": [.15, .4], "emg_band_hz": [25, 45], "emg_filter_order": 3,
                "baseline_relative_lead_phase_rad": [-.35, .35], "clean_predictions": "from the primary evaluation",
                "input": "perturbations added in raw mV before train-fitted per-lead normalization", "evaluation_fold": 10})
    original, ids, _, folds = load_data(args.data)
    test = np.flatnonzero(folds == 10)
    raw, truth = [t.numpy()[test] for t in original.tensors]
    ids = ids[test]
    del original
    noise = noise_bases(ids, raw.shape[1], raw.shape[2])
    views = {f"{name}_{i}": torch.as_tensor(raw + basis * strength, device=args.device)
             for name, basis in noise.items() for i, strength in enumerate(LEVELS[name], 1)}
    outcomes = {}
    for run in runs:
        if time() >= args.deadline:
            raise SystemExit(75)
        outcomes[run] = score(args, run, views, ids, truth)
        if len(outcomes) % 10 == 0:
            print(f"{len(outcomes)}/{len(runs)} acquisition-artifact evaluations complete", flush=True)
    save_summary(args, outcomes)
    write_json(args.out / "results.json", {"completed_runs": len(outcomes), "test_records": len(ids),
                                          "same_original_fold10_records_and_selected_checkpoints": True})


if __name__ == "__main__":
    main()
