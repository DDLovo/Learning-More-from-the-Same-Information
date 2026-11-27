"""Replay frozen RR auxiliary heads and score a fixed ordinal projection of the raw outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from . import CLASSES
from .rr_questions import FEATURES
from .teacher import write_json
from .train import ECGStudent, cohort_splits, load_data, make_loader, sha256, statistics


PROJECTION = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
METHODS = ("correct", "permuted")
SHAPE = (2, 5)


@torch.inference_mode()
def infer(model, loader, normalization, device, amp):
    model.eval()
    outputs, probabilities = [], []
    mean, std = (tensor.to(device) for tensor in normalization)
    for ecg, _ in loader:
        ecg = (ecg.to(device, non_blocking=True) - mean) / std
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            _, auxiliary = model(ecg, with_auxiliary=True)
        auxiliary = auxiliary.float()
        outputs.append(auxiliary.cpu().numpy())
        probabilities.append(auxiliary.softmax(-1).cpu().numpy())
    return np.concatenate(outputs), np.concatenate(probabilities)


def replay(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    dataset, ids, patients, folds = load_data(args.data)
    labels = dataset.tensors[1].numpy()
    splits, _ = cohort_splits(folds, patients, labels)
    model = ECGStudent(SHAPE).to(device)
    output_hashes = {}
    data_digest = sha256(args.data)
    for run_root in args.results:
        config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
        if config["waveforms_sha256"] != data_digest:
            raise ValueError(f"The training waveforms differ from the replay input at {run_root}")
        loaders = {split: make_loader(dataset, splits[split], config["batch_size"], device)
                   for split in ("validation", "test")}
        amp = config["precision"] == "bfloat16"
        for seed in config["seeds"]:
            run_subdir = f"fraction-{config['fraction']}_seed-{seed}"
            result_path = run_root / run_subdir
            output_path = args.out / run_subdir
            output_path.mkdir(parents=True, exist_ok=True)
            for method in METHODS:
                recorded = json.loads((result_path / f"{method}.json").read_text(encoding="utf-8"))
                checkpoint_path = result_path / f"{method}.pt"
                assert sha256(checkpoint_path) == recorded["checkpoint_sha256"]
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["model_state"])
                for split, loader in loaders.items():
                    idx = splits[split]
                    output, probs = infer(model, loader, checkpoint["normalization"], device, amp)
                    previous = result_path / f"{method}_{split}_predictions.npz"
                    with np.load(previous, allow_pickle=False) as archived:
                        np.testing.assert_array_equal(archived["ecg_id"], ids[idx])
                        np.testing.assert_array_equal(archived["array_index"], idx)
                        np.testing.assert_array_equal(archived["y_true"], labels[idx])
                        np.testing.assert_array_equal(archived["rr_aux"], probs)
                        values = archived["rr_measured"].copy()
                    filename = output_path / f"{method}_{split}_raw_aux.npz"
                    np.savez_compressed(filename, ecg_id=ids[idx], y_true=labels[idx], array_index=idx,
                                        class_codes=np.asarray(tuple(CLASSES)), feature_names=np.asarray(FEATURES),
                                        rr_raw_aux=output, projection=PROJECTION,
                                        ordinal_score=output @ PROJECTION, unig_reference_ms=values)
                    output_hashes[filename.relative_to(args.out).as_posix()] = {
                        "checkpoint_sha256": recorded["checkpoint_sha256"],
                        "original_predictions_sha256": sha256(previous),
                        "raw_aux_predictions_sha256": sha256(filename),
                    }
                print(run_subdir, method, "replayed; exact archived softmax", flush=True)
    write_json(args.out / "raw_predictions_manifest.json", {"prediction_sha256": output_hashes,
                                                               "projection": PROJECTION.tolist(),
                                                               "selection_changed": False})


def group_scores(actual, estimate, rhythm):
    result = {}
    for j, field in enumerate(FEATURES):
        per_rhythm = {}
        for k, code in enumerate(CLASSES):
            mask = (rhythm == k) & np.isfinite(actual[:, j])
            target, prediction = actual[mask, j], estimate[mask, j]
            if len(target) < 2 or np.unique(target).size < 2 or np.unique(prediction).size < 2:
                raise ValueError(f"Spearman undefined for {field}/{code}")
            per_rhythm[code] = {"valid_n": int(mask.sum()), "rho": float(spearmanr(target, prediction).statistic)}
        result[field] = {"macro_spearman": float(np.mean([row["rho"] for row in per_rhythm.values()])),
                         "by_rhythm": per_rhythm}
    return result


def score(args):
    independent = pd.read_csv(args.ecgdeli, float_precision="round_trip").set_index("ecg_id", verify_integrity=True)
    assert (independent["strat_fold"] == 10).all()
    results = {}
    for path in sorted(args.raw.glob("fraction-*_seed-*/*_test_raw_aux.npz")):
        fraction, _, seed = path.parent.name.removeprefix("fraction-").partition("_seed-")
        method = path.name.split("_")[0]
        with np.load(path, allow_pickle=False) as data:
            ids, labels = data["ecg_id"], data["y_true"]
            assert len(ids) == len(independent) and set(ids) == set(independent.index)
            aligned = independent.loc[ids]
            np.testing.assert_array_equal(aligned["class_code"], np.asarray(tuple(CLASSES))[labels])
            np.testing.assert_array_equal(aligned["array_index"], data["array_index"])
            np.testing.assert_array_equal(aligned[list(FEATURES)].to_numpy(), data["unig_reference_ms"])
            np.testing.assert_array_equal(data["projection"], PROJECTION)
            np.testing.assert_array_equal(data["rr_raw_aux"] @ PROJECTION, data["ordinal_score"])
            ecgdeli_values = aligned[["peak_mean_ms", "peak_sd_ms"]].to_numpy(dtype=np.float64, copy=True)
            ecgdeli_values[aligned["status"].to_numpy() != "ok"] = np.nan
            result = {source: group_scores(reference, data["ordinal_score"], labels)
                      for source, reference in (("Uni-G", data["unig_reference_ms"]), ("ECGDeli", ecgdeli_values))}
        results.setdefault(fraction, {}).setdefault(seed, {})[method] = result
    if not results:
        raise ValueError("No raw-score inputs")
    summary = {}
    for fraction, runs in results.items():
        if set(runs) != {"17", "29", "43"} or any(set(run) != set(METHODS) for run in runs.values()):
            raise ValueError(f"Missing pre-specified seed or matched arm for fraction {fraction}")
        summary[fraction] = {}
        for source in ("Uni-G", "ECGDeli"):
            summary[fraction][source] = {}
            for field in FEATURES:
                values = {method: [runs[seed][method][source][field]["macro_spearman"] for seed in ("17", "29", "43")]
                          for method in METHODS}
                summary[fraction][source][field] = {
                    **{method: statistics(scores) for method, scores in values.items()},
                    "correct_minus_permuted": statistics([a - b for a, b in zip(values["correct"], values["permuted"])]),
                }
    write_json(args.out, {"runs": results, "summary": summary, "ecgdeli_reference_sha256": sha256(args.ecgdeli),
                          "projection": PROJECTION.tolist(),
                          "interpretation": "Exploratory after the original held-out answer accuracy evaluation; no fitting or rhythm routing"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer_args = commands.add_parser("replay")
    infer_args.add_argument("--data", type=Path, required=True)
    infer_args.add_argument("--results", type=Path, nargs="+", required=True)
    infer_args.add_argument("--out", type=Path, required=True)
    infer_args.add_argument("--device", default="cuda")
    infer_args.set_defaults(action=replay)
    score_args = commands.add_parser("score")
    score_args.add_argument("--raw", type=Path, required=True)
    score_args.add_argument("--ecgdeli", type=Path, required=True)
    score_args.add_argument("--out", type=Path, required=True)
    score_args.set_defaults(action=score)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
