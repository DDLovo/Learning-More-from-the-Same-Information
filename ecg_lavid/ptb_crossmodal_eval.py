"""Evaluate sealed PTB-XL models on a shared clean fold and deterministic time perturbations."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter, time

import numpy as np
import torch

from . import CLASSES, LEADS
from .teacher import write_json
from .train import cohort_splits, load_data, score_predictions, sha256
from .train_ptb_crossmodal import ALL_ARCHITECTURES, ARCHITECTURES, ARMS, FRACTIONS, REFERENCE_ARMS, SEEDS
from .train_ptb_crossmodal import batch_options, build_network, freeze_json, infer, metrics, run_directory, save_checkpoint


LEVELS = {"shift": (2, 4, 8), "warp": (2, 4, 8), "gap": (5, 10, 20)}
PERTURBATION_SEED = 20261127


def parameters(ids, width):
    points = []
    for ecg_id in ids:
        rng = np.random.default_rng(np.random.SeedSequence([PERTURBATION_SEED, int(ecg_id)]))
        points.append((rng.choice((-1, 1)), rng.uniform(.75, 2.), rng.uniform(0., 2 * np.pi),
                       rng.integers(width // 5, 4 * width // 5)))
    return np.asarray(points, dtype=np.float64)


def changed_signal(x, family, severity, descriptor):
    sign, cycles, phase, center = descriptor.T
    width = x.shape[-1]
    position = np.arange(width)[None, :]
    if family == "shift":
        raw_index = position + sign[:, None].astype(np.int64) * severity
        raw_index = np.where(raw_index < 0, -raw_index, raw_index)
        reflect = np.where(raw_index >= width, 2 * (width - 1) - raw_index, raw_index)
        return np.take_along_axis(x, reflect[:, None, :], axis=2)
    if family == "warp":
        t = position / (width - 1)
        change = severity * np.sin(np.pi * t) * np.sin(2 * np.pi * cycles[:, None] * t + phase[:, None])
        raw_index = (position + change).clip(0, width - 1)
        if not (np.diff(raw_index, axis=1) > 0).all():
            raise ValueError("A local time perturbation is not monotone")
        left = np.floor(raw_index).astype(np.int64)
        right = np.minimum(left + 1, width - 1)
        frac = (raw_index - left).astype(np.float32)[:, None, :]
        a = np.take_along_axis(x, left[:, None, :], axis=2)
        b = np.take_along_axis(x, right[:, None, :], axis=2)
        return a + (b - a) * frac
    result = x.copy()
    weight = np.linspace(0, 1, severity + 2, dtype=np.float32)[1:-1]
    for row, midpoint in enumerate(center.astype(np.int64)):
        start = midpoint - severity // 2
        left, right = x[row, :, start - 1], x[row, :, start + severity]
        result[row, :, start:start + severity] = left[:, None] + (right - left)[:, None] * weight
    return result


def make_perturbations(values, ids, out, device):
    options = parameters(ids, values.shape[-1])
    views = {"clean": torch.as_tensor(values, device=device)}
    for name, strengths in LEVELS.items():
        for level, value in enumerate(strengths, 1):
            views[f"{name}_{level}"] = torch.as_tensor(changed_signal(values, name, value, options), device=device)
    definition = {
        "fold": 10, "sample_rate_hz": 100, "perturbation_seed": PERTURBATION_SEED,
        "strength_samples_at_100hz": {k: list(v) for k, v in LEVELS.items()},
        "shift": "record-specific signed shift of 20/40/80 ms with reflected edges",
        "warp": "monotone local time resampling with fixed phase and frequency per record, endpoint taper, 20/40/80 ms max envelope",
        "gap": "interior gap of 50/100/200 ms replaced by linear interpolation; one fixed center per record",
        "target": "original rhythm label retained under the synthetic timing nuisance",
        "alignment": "same transformation across leads of each record, and across all arms and seeds; operates in raw mV before train-fitted normalization",
        "retention": "macro F1 divided by the clean macro F1, and proportion of initially correct predictions remaining correct",
        "severity_auc": "trapezoidal mean over equally scaled severity index 0/1/2/3, including clean",
    }
    freeze_json(out / "perturbation-protocol.json", definition)
    np.savez_compressed(out / "perturbation-records.npz", ecg_id=ids, shift_sign=options[:, 0], warp_cycles=options[:, 1],
                        warp_phase_radians=options[:, 2], gap_center_sample=options[:, 3])
    return views


def state_and_stats(path, architecture, device, remove_aux=False):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_network(architecture)
    if remove_aux:
        model.question_head = None
    model.load_state_dict(saved["state"], strict=True)
    return model.to(device).eval(), tuple(saved[name].to(device) for name in ("mean", "std")), saved


def probabilities(logits):
    return torch.as_tensor(logits).softmax(-1).numpy()


def severity_auc(values):
    return float((values[0] / 2 + sum(values[1:-1]) + values[-1] / 2) / (len(values) - 1))


def robustness_summary(results):
    families = {}
    for family in LEVELS:
        curve = [results["clean"]] + [results[f"{family}_{step}"] for step in range(1, 4)]
        families[family] = {f"auc_{field}": severity_auc([row[field] for row in curve])
                            for field in ("macro_f1", "macro_f1_retention", "initial_correct_retention")}
    condition = min((key for key in results if key != "clean"), key=lambda key: results[key]["macro_f1"])
    return {"families": families, "worst_condition": {"condition": condition} | results[condition]}


@torch.inference_mode()
def export_and_time(args, model, checkpoint, wave, stats, path, measure):
    center, scale = stats
    wave = (wave[:args.batch_size] - center) / scale
    autocast = batch_options(args)
    with torch.autocast(**autocast):
        both, _ = model(wave, with_auxiliary=True)
        plain = model(wave)
    torch.testing.assert_close(plain, both, rtol=0, atol=0)

    def timed(with_auxiliary):
        if not measure:
            return None
        with torch.autocast(**autocast):
            for _ in range(5):
                model(wave, with_auxiliary=with_auxiliary)
            times = []
            for _ in range(15):
                if args.device.type == "cuda":
                    torch.cuda.synchronize(args.device)
                start = perf_counter()
                model(wave, with_auxiliary=with_auxiliary)
                if args.device.type == "cuda":
                    torch.cuda.synchronize(args.device)
                times.append((perf_counter() - start) * 1000)
        return {"median_ms_per_batch": float(np.median(times)), "p90_ms_per_batch": float(np.quantile(times, .9)),
                "examples_per_batch": len(wave)}

    latency_with = timed(True)
    count_with = sum(p.numel() for p in model.parameters())
    model.question_head = None
    with torch.autocast(**autocast):
        torch.testing.assert_close(model(wave), plain, rtol=0, atol=0)
    latency_without = timed(False)
    only_classifier = {k: v for k, v in checkpoint["state"].items() if not k.startswith("question_head.")}
    path.parent.mkdir(exist_ok=True, parents=True)
    save_checkpoint(path, checkpoint | {"state": only_classifier, "auxiliary_head_discarded": True})
    return {"parameters_with_auxiliary": count_with,
            "parameters_for_standalone_classifier": sum(p.numel() for p in model.parameters()),
            "auxiliary_head_discarded_and_logits_equal": True,
            "with_auxiliary": latency_with, "classifier_only": latency_without,
            "classifier_only_sha256": sha256(path),
            "classifier_only_bytes": path.stat().st_size}


def evaluate_arm(args, run, views, truth, ids, patients, folds):
    architecture, fraction, seed, arm = run
    trained = run_directory(args.runs, *run)
    out = run_directory(args.out, *run)
    out.mkdir(exist_ok=True, parents=True)
    result_file = out / "results.json"
    if result_file.is_file():
        result = json.loads(result_file.read_bytes())
        if sha256(out / "test-predictions.npz") != result["predictions_sha256"]:
            raise ValueError(f"The saved test predictions have changed: {out}")
        return result

    model, stats, checkpoint = state_and_stats(trained / "best.pt", architecture, args.device)
    eval_rows = torch.arange(len(ids), device=args.device)
    clean_hits, eval_results, logits_all, all_scores = None, {}, [], []
    started = perf_counter()
    for condition, data in views.items():
        logits = infer(model, data, eval_rows, stats, args.batch_size, batch_options(args))
        score = probabilities(logits)
        hit = score.argmax(1) == truth
        if condition == "clean":
            clean_hits = hit
            metrics_here = score_predictions(truth, score)
            base = metrics_here["macro_f1"]
        else:
            metrics_here = metrics(truth, score.argmax(1))
        if not np.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite test logits for {run} and {condition}")
        eval_results[condition] = metrics_here | {"macro_f1_retention": metrics_here["macro_f1"] / max(base, 1e-12),
                                                   "initial_correct_retention": float(hit[clean_hits].mean()) if clean_hits.any() else 0.}
        logits_all.append(logits)
        all_scores.append(score)
    final, final_stats, _ = state_and_stats(trained / "final.pt", architecture, args.device)
    final_logits = infer(final, views["clean"], eval_rows, final_stats, args.batch_size, batch_options(args))
    timing = export_and_time(args, model, checkpoint, views["clean"], stats, out / "standalone-classifier.pt",
                             measure=fraction == 1. and seed == 17)
    pred = out / "test-predictions.npz"
    tmp = out / "test-predictions.part.npz"
    np.savez_compressed(tmp, ecg_id=ids, patient_id=patients, fold=folds, y_true=truth, class_codes=np.asarray(CLASSES),
                        conditions=np.asarray(list(views)), logits=np.stack(logits_all), probabilities=np.stack(all_scores),
                        final_epoch_clean_logits=final_logits)
    tmp.replace(pred)
    history = json.loads((trained / "history.json").read_bytes())
    seal = json.loads((trained / "seal.json").read_bytes())
    result = {"architecture": architecture, "fraction": fraction, "seed": seed, "arm": arm,
              "model_selection": "best fold-9 macro F1 (earlier ties)", "selected_epoch": checkpoint["epoch"],
              "validation_fold9": history[checkpoint["epoch"] - 1]["validation"],
              "test_fold10": eval_results, "robustness": robustness_summary(eval_results),
              "fixed_final_epoch_clean_fold10": metrics(truth, final_logits.argmax(1)),
              "inference": timing, "training": seal, "evaluation_wall_seconds": perf_counter() - started,
              "predictions_sha256": sha256(pred)}
    write_json(result_file, result)
    print(f"{architecture} {fraction:g} {seed} {arm}: fold10 clean macro F1 {base:.4f}, worst "
          f"{result['robustness']['worst_condition']['macro_f1']:.4f}", flush=True)
    return result


def validate_seals(args, runs):
    expected = {}
    matching = {}
    for run in runs:
        trained = run_directory(args.runs, *run)
        seal_path = trained / "seal.json"
        seal = json.loads(seal_path.read_bytes())
        config = json.loads((trained / "config.json").read_bytes())
        for filename, field in (("best.pt", "best_sha256"), ("final.pt", "final_sha256"), ("history.json", "history_sha256")):
            if sha256(trained / filename) != seal[field]:
                raise ValueError(f"Training output changed after seal: {trained}/{filename}")
        if seal["epochs"] != config["epochs"]:
            raise ValueError(f"Model is not fully trained: {trained}")
        match = {k: config[k] for k in ("initial_state_sha256", "training_order_sha256", "train_patients", "epochs",
                                        "batch_size", "lr", "normalization", "weight_decay")}
        if matching.setdefault(run[:3], match) != match:
            raise ValueError(f"Model groups were not trained under matched conditions: {trained}")
        expected[trained.relative_to(args.runs).as_posix()] = sha256(seal_path)
    freeze_json(args.out / "sealed-before-fold10.json", expected)


def paired_summary(results, args):
    seed_rows, group_rows = [], []
    for run, entry in results.items():
        architecture, fraction, seed, arm = run
        clean = entry["test_fold10"]["clean"]
        worst = entry["robustness"]["worst_condition"]
        seed_rows.append({"architecture": architecture, "fraction": fraction, "seed": seed, "arm": arm,
                          "fold9_macro_f1": entry["validation_fold9"]["macro_f1"], "fold10_macro_f1": clean["macro_f1"],
                          "fold10_balanced_accuracy": clean["balanced_accuracy"],
                          "fold10_macro_auprc": clean["macro_auprc"], "worst_macro_f1": worst["macro_f1"],
                          "worst_macro_f1_retention": worst["macro_f1_retention"],
                          "mean_family_auc": float(np.mean([x["auc_macro_f1"] for x in entry["robustness"]["families"].values()])),
                          "best_epoch": entry["selected_epoch"], "training_seconds": entry["training"]["elapsed_seconds"]})
    index = {(r["architecture"], r["fraction"], r["seed"], r["arm"]): r for r in seed_rows}
    if set(ARMS).issubset(set(args.arms)):
        for architecture in args.architectures:
            for fraction in args.fractions:
                for metric in ("fold9_macro_f1", "fold10_macro_f1", "fold10_balanced_accuracy", "fold10_macro_auprc",
                               "worst_macro_f1", "worst_macro_f1_retention", "mean_family_auc", "training_seconds"):
                    a = np.array([index[architecture, fraction, seed, "A"][metric] for seed in args.seeds])
                    h = np.array([index[architecture, fraction, seed, "H"][metric] for seed in args.seeds])
                    diff = h - a
                    group_rows.append({"architecture": architecture, "fraction": fraction, "metric": metric,
                                       "a_mean": a.mean(), "h_mean": h.mean(), "paired_h_minus_a": diff.mean(),
                                       "paired_delta_sd": diff.std(ddof=1) if len(diff) > 1 else 0.,
                                       "paired_delta_min": diff.min(), "paired_delta_max": diff.max(),
                                       "positive_count": int((diff > 0).sum()), "seeds": len(diff)})
    for filename, rows in (("seed-results.csv", seed_rows), ("paired-results.csv", group_rows)):
        if rows:
            with (args.out / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", choices=ALL_ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--fractions", nargs="+", type=float, choices=FRACTIONS, default=list(FRACTIONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=ARMS + REFERENCE_ARMS, default=list(ARMS))
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--deadline", type=lambda stamp: datetime.fromisoformat(stamp).timestamp(), default=float("inf"))
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.out.mkdir(parents=True, exist_ok=True)
    runs = [(arch, frac, seed, arm) for arch in args.architectures for frac in args.fractions
            for seed in args.seeds for arm in args.arms]
    validate_seals(args, runs)
    dataset, ids, patients, folds = load_data(args.data)
    x, y = dataset.tensors
    truth = y.numpy()
    split, _ = cohort_splits(folds, patients, truth)
    test = split["test"]
    views = make_perturbations(x.numpy()[test], ids[test], args.out, args.device)
    del dataset, x
    results = {}
    for run in runs:
        if time() >= args.deadline:
            print("Deadline reached at the evaluation boundary. Existing prediction artifacts are sealed.", flush=True)
            raise SystemExit(75)
        results[run] = evaluate_arm(args, run, views, truth[test], ids[test], patients[test], folds[test])
    paired_summary(results, args)
    write_json(args.out / "results.json", {"test_fold": 10, "class_codes": list(CLASSES), "leads": list(LEADS),
                                          "test_records": len(test), "test_patients": int(np.unique(patients[test]).size),
                                          "completed_runs": len(results), "seeds": args.seeds,
                                          "model_selection_fold": 9,
                                          "completed_rows": [list(k) for k in results]})


if __name__ == "__main__":
    main()
