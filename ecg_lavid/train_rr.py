"""Compare record-level RR supervision, within-rhythm shuffled targets, and rhythm CE."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from . import CLASSES, LEADS, SAMPLE_RATE
from .rr_questions import FEATURES, validate_questions
from .teacher import LETTERS, write_json
from .train import (ECGStudent, cohort_splits, counts_for, describe_losses, fit_normalization,
                    load_data, make_loader, score_predictions, select_train_subset, sha256, state_hash,
                    statistics, write_subset_index)


METHODS = ("ce", "correct", "permuted")
QUANTILES = (0.2, 0.4, 0.6, 0.8)
RIDGE_ALPHA = 10.0
PERMUTATION_SEED = 104729
QUESTION_SHAPE = (len(FEATURES), len(LETTERS))
FEATURE_SOURCE = "https://physionet.org/files/ptb-xl-plus/1.0.1/features/unig_features.csv"


def load_measurements(path, ecg_ids):
    data = pd.read_csv(path, usecols=["ecg_id", *FEATURES]).set_index("ecg_id", verify_integrity=True)
    aligned = data.reindex(ecg_ids)[list(FEATURES)].to_numpy(dtype=np.float64, copy=True)
    aligned[~np.isfinite(aligned)] = np.nan
    aligned[aligned[:, 0] <= 0, 0] = np.nan
    aligned[aligned[:, 1] < 0, 1] = np.nan
    return aligned, int(np.isin(ecg_ids, data.index).sum())


def measure_bins(values, edges):
    bins = np.full(values.shape, -1, dtype=np.int64)
    for feature, boundaries in enumerate(edges):
        valid = np.isfinite(values[:, feature])
        bins[valid, feature] = np.searchsorted(boundaries, values[valid, feature], side="right")
    return bins


def permute_records(labels, patients, validity, seed):
    """Move complete target rows to different patients within each rhythm and validity pattern."""
    random = np.random.default_rng(np.random.SeedSequence([PERMUTATION_SEED, seed]))
    donor = np.arange(len(labels))
    active = validity.any(axis=1)
    groups = {}
    for row in np.flatnonzero(active):
        key = (int(labels[row]), *validity[row])
        groups.setdefault(key, {}).setdefault(patients[row], []).append(row)
    for blocks in groups.values():
        patient_order = random.permutation(sorted(blocks))
        order = np.concatenate([random.permutation(blocks[patient]) for patient in patient_order])
        shift = max(map(len, blocks.values()))
        if 2 * shift > len(order):
            active[order] = False
        else:
            donor[order] = np.roll(order, -shift)
    if (np.any(patients[active] == patients[donor[active]]) or
        not np.array_equal(labels, labels[donor]) or
        not np.array_equal(validity, validity[donor]) or len(np.unique(donor)) != len(donor)):
        raise RuntimeError("Within-rhythm whole-row permutation is invalid")
    return donor, active


def coverage(values, labels):
    return {
        code: {
            "records": int((labels == index).sum()),
            **{field: int(np.isfinite(values[labels == index, j]).sum()) for j, field in enumerate(FEATURES)},
        }
        for index, code in enumerate(CLASSES)
    }


def seed_targets(directory, seed, subset, values, questions, ecg_ids, patient_ids, labels, folds):
    train_values = values[subset]
    validity = np.isfinite(train_values)
    if not validity.any(axis=0).all():
        raise ValueError(f"Seed {seed} training subset is missing a complete RR concept")
    edges = np.nanquantile(train_values, QUANTILES, axis=0).T
    if not (np.diff(edges, axis=1) > 0).all():
        raise ValueError(f"Seed {seed} has repeated train quantiles and cannot form five distinct ranges")
    bins = measure_bins(train_values, edges)
    donor, active = permute_records(labels[subset], patient_ids[subset], validity, seed)
    active_mask = validity & active[:, None]
    if not active_mask.any(axis=0).all():
        raise ValueError(f"Seed {seed} has no matchable records for at least one RR concept")

    subset_path = directory / "train_indices.csv"
    assignment_path = directory / "rr_assignments.csv"
    write_subset_index(subset_path, subset, ecg_ids, patient_ids, labels, folds)
    with assignment_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ecg_id", "donor_ecg_id", "class_code", "active_aux", *(
            f"{field}_{suffix}" for field in FEATURES for suffix in ("value", "target", "donor_target")
        )])
        class_codes = list(CLASSES)
        for row, index in enumerate(subset):
            writer.writerow([ecg_ids[index], ecg_ids[subset[donor[row]]] if active[row] else "",
                             class_codes[labels[index]], int(active[row]), *(
                item for j in range(len(FEATURES)) for item in (
                    train_values[row, j] if validity[row, j] else "",
                    LETTERS[bins[row, j]] if validity[row, j] else "",
                    LETTERS[bins[donor[row], j]] if active_mask[row, j] else "",
                )
            )])
    different = active_mask & (bins != bins[donor])
    examples = np.flatnonzero(different.all(axis=1) & (labels[subset] == 0))
    example = None
    if len(examples):
        row = examples[0]
        example = {
            "class_code": next(iter(CLASSES)), "ecg_id": int(ecg_ids[subset[row]]),
            "donor_ecg_id": int(ecg_ids[subset[donor[row]]]),
            "features": {field: {"record_value": float(train_values[row, j]),
                                 "record_answer": LETTERS[bins[row, j]],
                                 "other_record_value": float(train_values[donor[row], j]),
                                 "permuted_answer": LETTERS[bins[donor[row], j]]}
                         for j, field in enumerate(FEATURES)},
        }
    feature_info = {}
    for j, field in enumerate(FEATURES):
        bounds = [f"{edge:.12g}" for edge in edges[j]]
        options = [f"Less than {bounds[0]} ms",
                   *(f"At least {lower} and less than {upper} ms" for lower, upper in zip(bounds, bounds[1:])),
                   f"At least {bounds[-1]} ms"]
        feature_info[field] = {
            **questions[field], "edges_ms": edges[j].tolist(), "options": dict(zip(LETTERS, options)),
            "training_valid": int(validity[:, j].sum()), "auxiliary_valid": int(active_mask[:, j].sum()),
            "auxiliary_answer_changed": int(different[:, j].sum()),
            "train_answer_counts": dict(zip(LETTERS, np.bincount(bins[validity[:, j], j], minlength=len(LETTERS)).tolist())),
        }
    threshold_report = {"features": feature_info, "example": example,
                        "valid_records_without_match": int((validity.any(axis=1) & ~active).sum()),
                        "training_coverage": coverage(train_values, labels[subset]),
                        "train_indices_sha256": sha256(subset_path), "rr_assignments_sha256": sha256(assignment_path)}
    write_json(directory / "thresholds.json", threshold_report)
    return edges, bins, donor, active_mask, threshold_report


@torch.inference_mode()
def infer(model, loader, normalization, device, amp, *, embed=False, with_aux=False):
    model.eval()
    mean, std = normalization
    probabilities, embeddings, auxiliary = [], [], []
    for signals, _, _ in loader:
        signals = (signals.to(device, non_blocking=True) - mean) / std
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            if embed:
                encoded = model.encoder(signals)
                pooled = torch.cat((encoded.mean(-1), encoded.amax(-1)), dim=1)
                logits = model.classifier(pooled)
                embeddings.append(pooled.float().cpu())
                if with_aux:
                    logits_aux = model.question_head(pooled).reshape(-1, *QUESTION_SHAPE)
                    auxiliary.append(logits_aux.float().softmax(dim=-1).cpu())
            else:
                logits = model(signals)
        probabilities.append(logits.float().softmax(dim=-1).cpu())
    return (torch.cat(probabilities).numpy(), torch.cat(embeddings).numpy() if embeddings else None,
            torch.cat(auxiliary).numpy() if auxiliary else None)


def train_one(args, model, initial, loader, val_loader, val_labels, normalization, target, mask, device, amp):
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mean, std = normalization
    history = []
    best_f1 = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = torch.zeros(2, device=device)
        num_steps = 0
        lr = optimizer.param_groups[0]["lr"]
        for signals, labels, rows in loader:
            signals = (signals.to(device, non_blocking=True) - mean) / std
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                output = model(signals, with_auxiliary=target is not None)
            logits = output if target is None else output[0]
            ce = F.cross_entropy(logits.float(), labels)
            mse = torch.zeros_like(ce)
            if target is not None:
                rows = rows.to(device, non_blocking=True)
                valid = mask[rows]
                squared = (output[1].float() - target[rows]).square().mean(dim=-1)
                mse = (squared * valid).sum() / valid.sum().clamp_min(1)
            (ce + args.lavid_weight * mse).backward()
            optimizer.step()
            totals += len(labels) * torch.stack((ce.detach(), mse.detach()))
            num_steps += 1
        scheduler.step()
        probabilities, _, _ = infer(model, val_loader, normalization, device, amp)
        validation = score_predictions(val_labels, probabilities)
        train_ce, train_mse = (totals / len(loader.dataset)).tolist()
        loss_values = describe_losses(train_ce, train_mse, args.lavid_weight)
        history.append({"epoch": epoch, "lr": lr, "optimizer_steps": num_steps, "train_ce": train_ce,
                        "train_mse": train_mse, "weighted_mse_loss_share": loss_values["weighted_mse_loss_share"],
                        "val_macro_f1": validation["macro_f1"]})
        if validation["macro_f1"] > best_f1:
            best_f1, best_epoch = validation["macro_f1"], epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(f"  epoch {epoch:02}/{args.epochs}  train CE={train_ce:.4f}  MSE={train_mse:.4f}  "
              f"val macro-F1={validation['macro_f1']:.4f}", flush=True)
    return best_state, {"best_epoch": best_epoch, "optimizer_steps": sum(row["optimizer_steps"] for row in history),
                        "history": history}


def average_defined(values):
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def concept_scores(actual, prediction, labels, *, auxiliary=False):
    features = {}
    for j, field in enumerate(FEATURES):
        per_class = {}
        for k, code in enumerate(CLASSES):
            valid = (labels == k) & (actual[:, j] >= 0) & np.isfinite(prediction[:, j])
            expected, observed = actual[valid, j], prediction[valid, j]
            count = len(expected)
            metrics = {"valid_n": count, "rhythm_support": int((labels == k).sum())}
            if auxiliary:
                metrics.update(accuracy=float(np.mean(expected == observed)) if count else None,
                               bin_mae=float(np.abs(observed - expected).mean()) if count else None)
            else:
                rho = None
                if count >= 2 and len(np.unique(expected)) > 1 and len(np.unique(observed)) > 1:
                    rho = float(spearmanr(expected, observed).statistic)
                metrics.update(spearman=rho, mae_ms=float(np.abs(observed - expected).mean()) if count else None)
            per_class[code] = metrics
        names = ("accuracy", "bin_mae") if auxiliary else ("spearman", "mae_ms")
        features[field] = {
            "per_class": per_class, **{f"macro_{name}": average_defined(row[name] for row in per_class.values())
                                       for name in names},
            **{f"macro_{name}_n_classes": sum(row[name] is not None for row in per_class.values()) for name in names},
        }
    return {"features": features}


def fit_probes(embedding, measured, labels):
    probes = {}
    for i in range(len(CLASSES)):
        for j in range(len(FEATURES)):
            valid = (labels == i) & np.isfinite(measured[:, j])
            if valid.sum() >= 2:
                scaler = StandardScaler().fit(embedding[valid])
                ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(embedding[valid]), measured[valid, j])
                probes[i, j] = (scaler, ridge)
    return probes


def apply_probes(probes, embedding, labels):
    estimate = np.full((len(labels), len(FEATURES)), np.nan, dtype=np.float64)
    for (i, j), (scaler, ridge) in probes.items():
        group = labels == i
        if group.any():
            estimate[group, j] = ridge.predict(scaler.transform(embedding[group]))
    return estimate


def save_checkpoint(path, state, normalization, question_shape=QUESTION_SHAPE):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"model_state": state, "question_shape": question_shape, "class_codes": tuple(CLASSES),
                "leads": LEADS, "sample_rate": SAMPLE_RATE,
                "normalization": tuple(t.cpu() for t in normalization)}, temporary)
    temporary.replace(path)


def summarize(runs):
    if not runs:
        return {}
    by_method = {method: {run["seed"]: run["test"] for run in runs if run["method"] == method} for method in METHODS}
    summary = {}
    for field in FEATURES:
        summary[field] = {}
        for metric in ("macro_spearman", "macro_mae_ms"):
            scores = {method: {seed: row["probe"]["features"][field][metric] for seed, row in tests.items()}
                      for method, tests in by_method.items()}
            available = {method: [score for score in vals.values() if score is not None]
                         for method, vals in scores.items()}
            means = {method: statistics(vals) for method, vals in available.items() if vals}
            paired = {}
            for comparator in ("ce", "permuted"):
                seeds = sorted(scores["correct"].keys() & scores[comparator].keys())
                delta = [scores["correct"][seed] - scores[comparator][seed] for seed in seeds
                         if scores["correct"][seed] is not None and scores[comparator][seed] is not None]
                if delta:
                    paired[f"correct_minus_{comparator}"] = statistics(delta)
            summary[field][metric] = {"by_method": means, "paired": paired}
    return summary


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    amp = args.precision == "bfloat16" or (
        args.precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    data, ecg_ids, patient_ids, folds = load_data(args.data)
    labels = data.tensors[1].numpy()
    splits, overlap = cohort_splits(folds, patient_ids, labels)
    dataset = TensorDataset(*data.tensors, torch.arange(len(labels)))
    values, matched = load_measurements(args.features, ecg_ids)
    questions = validate_questions(json.loads(args.questions.read_text(encoding="utf-8-sig")))
    config = {
        "waveforms_sha256": sha256(args.data), "features_sha256": sha256(args.features),
        "feature_source": FEATURE_SOURCE, "questions_sha256": sha256(args.questions), "cohort_rows_in_features": matched,
        "class_codes": list(CLASSES), "feature_names": list(FEATURES), "answer_labels": list(LETTERS),
        "cross_split_patient_overlap": overlap, "seeds": args.seeds, "fraction": args.fraction,
        "epochs": args.epochs, "batch_size": args.batch_size, "cpu_threads": args.cpu_threads, "lr": args.lr,
        "weight_decay": args.weight_decay, "lavid_weight": args.lavid_weight, "ridge_alpha": RIDGE_ALPHA,
        "quantiles": list(QUANTILES), "permutation_seed": PERMUTATION_SEED, "methods": list(METHODS),
        "precision": "bfloat16" if amp else "float32", "torch_version": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_selection": "maximum fold 9 rhythm macro-F1, earliest epoch on ties",
        "primary_test_metric": "unweighted macro average of within-rhythm Spearman by concept",
        "split_counts": {name: counts_for(index, labels, patient_ids) for name, index in splits.items()},
        "measurement_coverage": {name: coverage(values[index], labels[index]) for name, index in splits.items()},
    }
    config_path = args.out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory contains a different experiment configuration")
    write_json(config_path, config)
    for name in ("validation", "test"):
        write_subset_index(args.out / f"{name}_indices.csv", splits[name], ecg_ids, patient_ids, labels, folds)
    validation_loader = make_loader(dataset, splits["validation"], args.batch_size, device)
    eval_queue = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = ECGStudent(QUESTION_SHAPE)
        initial = copy.deepcopy(model.state_dict())
        model.to(device)
        subset = select_train_subset(splits["train"], labels, patient_ids, args.fraction, seed)
        directory = args.out / f"fraction-{args.fraction}_seed-{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        edges, bins, donor, active, definition = seed_targets(
            directory, seed, subset, values, questions, ecg_ids, patient_ids, labels, folds,
        )
        normalization = fit_normalization(dataset.tensors[0].numpy(), subset, device)
        info = {"seed": seed, "fraction": args.fraction, "initial_state_sha256": state_hash(initial),
                "train_indices_sha256": definition["train_indices_sha256"],
                "rr_assignments_sha256": definition["rr_assignments_sha256"],
                "train_counts": counts_for(subset, labels, patient_ids)}
        indices = torch.from_numpy(subset).to(device)
        mask = torch.zeros((len(labels), len(FEATURES)), dtype=torch.bool, device=device)
        mask[indices] = torch.from_numpy(active).to(device)
        target = torch.zeros((len(labels), *QUESTION_SHAPE), device=device)
        one_hot = np.eye(len(LETTERS), dtype=np.float32)[bins.clip(min=0)]
        for name in METHODS:
            training_path, checkpoint_path = directory / f"{name}.training.json", directory / f"{name}.pt"
            print(f"fraction={args.fraction}  seed={seed}  method={name}  training ECGs={len(subset)}", flush=True)
            target[indices] = torch.from_numpy(one_hot[donor] if name == "permuted" else one_hot).to(device)
            if training_path.is_file() and checkpoint_path.is_file():
                trained = json.loads(training_path.read_text(encoding="utf-8"))
                if any(trained[key] != value for key, value in info.items()) or sha256(checkpoint_path) != trained["checkpoint_sha256"]:
                    raise ValueError(f"Existing training output differs from the current run: {training_path}")
            else:
                loader = make_loader(dataset, subset, args.batch_size, device, seed)
                state, trained = train_one(args, model, initial, loader, validation_loader, labels[splits["validation"]],
                                           normalization, None if name == "ce" else target, mask, device, amp)
                save_checkpoint(checkpoint_path, state, normalization)
                trained.update(info, method=name, checkpoint_sha256=sha256(checkpoint_path))
                write_json(training_path, trained)
            eval_queue.append((directory, name, subset, edges, trained))

    # Every checkpoint is chosen before the first forward pass on fold 10.
    test_loader = make_loader(dataset, splits["test"], args.batch_size, device)
    runs = []
    for directory, name, subset, edges, trained in eval_queue:
        result_path = directory / f"{name}.json"
        prediction_paths = {split: directory / f"{name}_{split}_predictions.npz"
                            for split in ("validation", "test")}
        if result_path.exists() and all(path.exists() for path in prediction_paths.values()):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["checkpoint_sha256"] != trained["checkpoint_sha256"]:
                raise ValueError(f"Existing evaluation uses a different checkpoint: {result_path}")
        else:
            checkpoint = torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            normalization = tuple(t.to(device) for t in checkpoint["normalization"])
            loader = make_loader(dataset, subset, args.batch_size, device)
            _, embeddings, _ = infer(model, loader, normalization, device, amp, embed=True)
            probes = fit_probes(embeddings, values[subset], labels[subset])
            result = dict(trained)
            for split, loader in (("validation", validation_loader), ("test", test_loader)):
                index = splits[split]
                probabilities, embeddings, auxiliary = infer(
                    model, loader, normalization, device, amp, embed=True, with_aux=name != "ce",
                )
                predictions = apply_probes(probes, embeddings, labels[index])
                result[split] = {
                    "classification": score_predictions(labels[index], probabilities),
                    "probe": concept_scores(values[index], predictions, labels[index]),
                    "auxiliary": concept_scores(measure_bins(values[index], edges), auxiliary.argmax(axis=-1),
                                                labels[index], auxiliary=True) if auxiliary is not None else None,
                }
                predictions_path = prediction_paths[split]
                temporary = predictions_path.with_suffix(".tmp.npz")
                np.savez(temporary, ecg_id=ecg_ids[index], array_index=index, y_true=labels[index],
                         class_codes=np.array(tuple(CLASSES)), feature_names=np.array(FEATURES),
                         probabilities=probabilities, rr_measured=values[index], rr_probe=predictions,
                         rr_aux=auxiliary if auxiliary is not None else np.empty((0, *QUESTION_SHAPE)))
                temporary.replace(predictions_path)
            write_json(result_path, result)
        runs.append(result)
        write_json(args.out / "results.json", {"runs": runs, "primary_probe_summary": summarize(runs)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--lavid-weight", type=float, default=0.4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.cpu_threads) < 1 or not 0 < args.fraction <= 1:
        parser.error("Epochs, batch size and threads must be positive, and the fraction in (0, 1]")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    if (not np.isfinite([args.lr, args.weight_decay, args.lavid_weight]).all() or args.lr <= 0 or
        args.lavid_weight <= 0 or args.weight_decay < 0):
        parser.error("Finite positive learning rate and RR weight, and finite nonnegative weight decay required")
    run(args)


if __name__ == "__main__":
    main()
