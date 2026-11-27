"""Audit the matched optimization and re-score archived visual student predictions."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def load(path):
    return json.loads(path.read_bytes())


def hash_file(path):
    with path.open("rb") as stream:
        checksum = sha256()
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def check_predictions(path, cohort, n, result):
    with np.load(path, allow_pickle=False) as evidence:
        ids, label, score = (evidence[key] for key in ("image_id", "y_true", "log_probability"))
    np.testing.assert_array_equal(ids, cohort["heldout_ids"])
    np.testing.assert_array_equal(label, cohort["heldout_labels"])
    if score.shape != (len(label), n) or not np.isfinite(score).all():
        raise ValueError(f"Invalid prediction shape or scores in {path}")
    np.testing.assert_allclose(logsumexp(score.astype(np.float64), axis=1), 0., atol=3e-6, rtol=0.)
    predicted = score.argmax(axis=1)
    correct = label == predicted
    counts = np.bincount(label, minlength=n) + np.bincount(predicted, minlength=n)
    observed = np.array([correct.mean(), (2 * np.bincount(label[correct], minlength=n) / counts).mean(),
                         -score[np.arange(len(label)), label].astype(np.float64).mean()])
    difference = observed - [result[key] for key in ("top1", "macro_f1", "nll")]
    if abs(difference).max() > 1e-12:
        raise ValueError(f"Metric mismatch in {path}")
    return float(abs(difference).max())


def audit_seed(root, arms, expected_epochs, weights):
    cohort_path = root / "cohort.json"
    cohort = load(cohort_path)
    training_n = len(cohort["training_ids"])
    n = len(set(cohort["training_labels"]))
    if set(cohort["training_ids"]) & set(cohort["heldout_ids"]) or len(set(cohort["training_ids"])) != training_n:
        raise ValueError("Official training IDs and held-out IDs must be disjoint without duplicates")
    codebook_path = root / "targets.json"
    codebook = load(codebook_path)
    mapping = codebook["donor_class_index_zero_based"]
    eligible = codebook.get("eligible_index_zero_based", list(range(n)))
    donors = [mapping[index] for index in eligible]
    if sorted(donors) != sorted(eligible) or any(a == b for a, b in zip(eligible, donors, strict=True)):
        raise ValueError("Expected a derangement within the supervised class set")
    if any(mapping[index] is not None for index in set(range(n)) - set(eligible)):
        raise ValueError("An excluded class has an auxiliary donor")
    masked = "masked_scope_sha256" in codebook
    if masked:
        coverage = {}
        for terms, split in (("training", "train"), ("heldout", "heldout")):
            chosen = np.isin(cohort[f"{terms}_labels"], eligible)
            ids = np.asarray(cohort[f"{terms}_ids"])[chosen].astype("<i8")
            spec = codebook["masked_train_and_heldout"][split]
            if len(ids) != spec["eligible"] or sha256(ids.tobytes()).hexdigest() != spec["selected_ids_original_order_sha256_le_i64"]:
                raise ValueError("Saved auxiliary mask differs from the fixed IDs")
            coverage[split] = len(ids)
    if set(arms).intersection(("ecoc", "ecoc-hard")):
        ecoc = np.asarray(codebook["ecoc_symbols_zero_based"])
        if not np.array_equal(np.bincount(ecoc.ravel(), minlength=5), np.full(5, 10 * n)):
            raise ValueError("Expected globally balanced five-way ECOC symbols")
    records = {}
    shared = None
    checks = {"prediction_files": 0, "checkpoint_files": 0, "maximum_metric_absolute_error": 0.}
    for arm in arms:
        run = root / arm
        config, metrics, epochs = (load(run / name) for name in ("config.json", "results.json", "history.json"))
        if config["cohort_sha256"] != hash_file(cohort_path) or config["epochs"] != expected_epochs:
            raise ValueError(f"Unexpected cohort or schedule in {run}")
        if masked and config.get("masked_scope_sha256") != codebook["masked_scope_sha256"]:
            raise ValueError(f"Different frozen auxiliary scope in {run}")
        if len(epochs) != expected_epochs or metrics["epochs"] != expected_epochs:
            raise ValueError(f"Incomplete training in {run}")
        selection = [row for row in epochs if row["epoch"] % 5 == 0]
        best = min(selection, key=lambda row: row["author_eval_split"]["nll"])
        if metrics["author_selection"] != {"epoch": best["epoch"], "validation_nll": best["author_eval_split"]["nll"]}:
            raise ValueError(f"The recorded author selection does not minimize held-out NLL in {run}")
        manifest = {"common_config": {key: value for key, value in config.items() if
                                      key not in ("arm", "target_float64_sha256", "muse_signature_sha256")},
                    "initial": metrics["initial_state_sha256"],
                    "seed_first_loss": epochs[0]["first_batch"]["ce"],
                    "seed_first_gradient": epochs[0]["first_batch"]["ce_first_layer_grad_norm"],
                    "batch_and_augment_plan": [(row["ordered_index_and_augmentation_seed_sha256"],
                                                 row["first_batch"]["first_augmented_batch_sha256"]) for row in epochs]}
        if shared is None:
            shared = manifest
        elif shared != manifest:
            raise ValueError(f"Unpaired run at {run}")
        for position, row in enumerate(epochs, 1):
            expected_lr = .01 * (.1 ** sum(position > x for x in (150, 180, 210)))
            if (row["epoch"] != position or row["records"] != training_n or
                    row["updates"] != math.ceil(training_n / config["batch_size"])
                    or not math.isclose(row["lr"], expected_lr, rel_tol=1e-12)):
                raise ValueError(f"Unexpected records, optimizer updates, or learning rate in {run} epoch {position}")
            if masked and (row["auxiliary_records"] != coverage["train"] or
                           row["auxiliary_denominator_coordinates"] != training_n * 250):
                raise ValueError(f"Auxiliary mask or reduction denominator changed during {run} epoch {position}")
        if metrics["optimizer_steps"] != sum(row["updates"] for row in epochs):
            raise ValueError(f"Total optimizer updates differ from the archived history in {run}")
        for name, digest in metrics["artifact_sha256"].items():
            path = run / name
            if path.suffix == ".pt":
                if not path.is_file() and not weights:
                    continue
                checks["checkpoint_files"] += 1
            if hash_file(path) != digest:
                raise ValueError(f"Unmatched file hash {path}")
        for prefix, metric_key in (("final", "fixed_final_fp32"), ("author_best", "author_selected_fp32")):
            error = check_predictions(run / f"{prefix}_predictions.npz", cohort, n, metrics[metric_key])
            checks["maximum_metric_absolute_error"] = max(checks["maximum_metric_absolute_error"], error)
            checks["prediction_files"] += 1
        records[arm] = {"top1_fixed_final": metrics["fixed_final_fp32"]["top1"],
                        "top1_author_selected": metrics["author_selected_fp32"]["top1"],
                        "author_selected_epoch": metrics["author_selection"]["epoch"],
                        "training_diagnostics": [
                            {"epoch": row["epoch"], "mean_ce": row["ce"], "mean_mse": row["mse"],
                             "weighted_mse_loss_share": row["weighted_mse_loss_share"],
                             "first_batch": row["first_batch"]}
                            for row in epochs if row["epoch"] in (1, 120, 240)
                        ]}
    projection_error = None
    if {"qwen", "qwen-span", "qwen-complement"}.issubset(records):
        initial_losses = {arm: records[arm]["training_diagnostics"][0]["first_batch"]["mse"]
                          for arm in ("qwen", "qwen-span", "qwen-complement")}
        projection_error = abs(initial_losses["qwen"] - initial_losses["qwen-span"] - initial_losses["qwen-complement"])
        if projection_error > 1e-6:
            raise ValueError("The matched initial Qwen projections do not decompose the original MSE")
    return {"train_images": training_n, "heldout_images": len(cohort["heldout_ids"]), "classes": n,
            "cohort_sha256": hash_file(cohort_path), "codebook_sha256": hash_file(codebook_path),
            "paired_training_verified": True if len(arms) > 1 else None,
            "optimizer_steps": metrics["optimizer_steps"],
            "initial_batch_projection_mse_absolute_error": projection_error,
            "masked_auxiliary_records": coverage if masked else None,
            "arms": records, **checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-results", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--require-weights", action="store_true")
    args = parser.parse_args()
    if args.epochs < 5:
        parser.error("Author checkpoint selection starts at epoch 5")
    report = {str(seed): audit_seed(args.dataset_results / f"seed-{seed}", args.arms, args.epochs, args.require_weights)
              for seed in args.seeds}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
