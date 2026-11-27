"""Score the original frozen RR probe predictions against ECGDeli R-peak intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES
from .rr_questions import FEATURES
from .teacher import write_json
from .train import sha256, statistics
from .train_rr import concept_scores


def compare(results, records):
    config = json.loads((results / "config.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(records).set_index("ecg_id", verify_integrity=True)
    if not (reference["strat_fold"] == 10).all():
        raise ValueError("ECGDeli references must come from fold 10")
    runs = []
    prediction_hashes = {}
    for seed in config["seeds"]:
        setting = results / f"fraction-{config['fraction']}_seed-{seed}"
        for method in config["methods"]:
            path = setting / f"{method}_test_predictions.npz"
            with np.load(path, allow_pickle=False) as prediction:
                ids, labels = prediction["ecg_id"], prediction["y_true"]
                if (tuple(prediction["feature_names"]) != FEATURES or
                    tuple(prediction["class_codes"]) != tuple(CLASSES) or
                    len(ids) != len(reference) or set(ids) != set(reference.index)):
                    raise ValueError(f"Unexpected class, concept, or ECG ID set: {path}")
                aligned = reference.loc[ids]
                if (not np.array_equal(aligned["class_code"], np.array(tuple(CLASSES))[labels]) or
                    not np.array_equal(aligned["array_index"], prediction["array_index"]) or
                    not np.array_equal(aligned[list(FEATURES)].to_numpy(), prediction["rr_measured"], equal_nan=True)):
                    raise ValueError(f"ECGDeli audit and prediction record metadata disagree: {path}")
                independent = aligned[["peak_mean_ms", "peak_sd_ms"]].to_numpy(dtype=np.float64, copy=True)
                independent[aligned["status"].to_numpy() != "ok"] = np.nan
                scores = concept_scores(independent, prediction["rr_probe"], labels)["features"]
            runs.append({
                "seed": seed, "method": method, "features": {
                    field: {"macro_spearman": item["macro_spearman"],
                            "macro_spearman_n_classes": item["macro_spearman_n_classes"],
                            "per_class": {code: {key: row[key] for key in ("spearman", "valid_n", "rhythm_support")}
                                          for code, row in item["per_class"].items()}}
                    for field, item in scores.items()
                },
            })
            prediction_hashes[path.relative_to(results).as_posix()] = sha256(path)
    summary = {}
    for field in FEATURES:
        scores = {method: {run["seed"]: run["features"][field]["macro_spearman"]
                           for run in runs if run["method"] == method}
                  for method in config["methods"]}
        summary[field] = {
            "by_method": {method: statistics([value for value in values.values() if value is not None])
                          for method, values in scores.items() if any(value is not None for value in values.values())},
            "paired_correct_minus": {
                method: statistics([scores["correct"][seed] - values[seed] for seed in config["seeds"]
                                    if scores["correct"][seed] is not None and values[seed] is not None])
                for method, values in scores.items() if method != "correct"
                and any(scores["correct"][seed] is not None and values[seed] is not None for seed in config["seeds"])
            },
        }
    return {
        "comparison": "Fixed fold 10 probe predictions against ECGDeli global R-peak interval measurements",
        "comparison_records": len(reference), "reference_sha256": sha256(records),
        "expected_columns": dict(zip(FEATURES, ["peak_mean_ms", "peak_sd_ms"])),
        "prediction_sha256": prediction_hashes, "runs": runs, "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, compare(args.results, args.records))


if __name__ == "__main__":
    main()
