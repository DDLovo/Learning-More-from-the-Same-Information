"""Report paired nine-view top-1 accuracy and the difference from the CE checkpoint."""
import argparse
import csv
import json
from hashlib import file_digest
from pathlib import Path

import numpy as np


def read_scores(path):
    summary = json.loads(path.with_suffix(".json").read_bytes())
    with path.open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != summary["prediction_sha256"]:
            raise ValueError(f"Prediction file does not match its summary: {path}")
    with np.load(path, allow_pickle=False) as data:
        cohort = {key: data[key] for key in ("image_id", "donor_image_id", "y_true", "variants")}
        scores = data["log_probability"]
    views = cohort["variants"].tolist()
    n = len(cohort["image_id"])
    if (n != summary["scored_records"] or n == 0 or len(np.unique(cohort["image_id"])) != n
            or len(set(views)) != 9 or scores.shape[:2] != (len(views), n)
            or scores.ndim != 3 or not np.isfinite(scores).all()):
        raise ValueError(f"Unexpected nine-view cohort or scores: {path}")
    counts = (scores.argmax(axis=2) == cohort["y_true"]).sum(axis=1)
    np.testing.assert_allclose(counts / n, [summary["top1"][view] for view in views], rtol=0, atol=1e-8)
    return cohort, summary["manifest_sha256"], counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ce", type=Path, required=True, help="A/CE intervention prediction NPZ")
    parser.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME", "NPZ"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cohort, manifest, baseline = read_scores(args.ce)
    n = len(cohort["image_id"])
    results = {"A": baseline}
    for arm, filename in args.arm:
        if arm in results:
            parser.error(f"Repeated arm: {arm}")
        other, other_manifest, counts = read_scores(Path(filename))
        if other_manifest != manifest:
            raise ValueError(f"Segmentation manifest differs for {arm}")
        for key in cohort:
            np.testing.assert_array_equal(cohort[key], other[key], err_msg=f"{arm}: {key}")
        results[arm] = counts
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("arm", "view", "n_images", "correct", "top1_percent", "delta_from_ce_pp"))
        for arm, counts in results.items():
            for view, count, reference in zip(cohort["variants"], counts, baseline, strict=True):
                writer.writerow((arm, view, n, count, 100 * count / n, 100 * (count - reference) / n))


if __name__ == "__main__":
    main()
