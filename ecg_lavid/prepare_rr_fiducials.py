"""Derive an independent R–R reference from released ECGDeli global peaks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from .prepare import download_file
from .rr_questions import FEATURES
from .train import sha256
from .train_rr import load_measurements


SOURCE = "https://physionet-open.s3.amazonaws.com/ptb-xl-plus/1.0.1/"


def measure(record_id, cache, fetch_missing):
    relative = f"fiducial_points/ecgdeli/{record_id // 1000 * 1000:05d}/{record_id:05d}_points_global.atr"
    path = cache / relative
    row = {"ecg_id": record_id, "feature_source": relative}
    if fetch_missing:
        download_file(cache, relative, base_url=SOURCE, allow_missing=True)
    if not path.is_file():
        return row | {"status": "missing_file"}
    annotation = wfdb.rdann(str(path.with_suffix("")), "atr")
    if annotation.fs != 500:
        raise ValueError(f"Expected released 500 Hz annotations for ECG {record_id}")
    peaks = np.asarray(annotation.sample)[np.asarray(annotation.aux_note) == "R peak"]
    row |= {"r_peak_count": len(peaks), "annotation_sha256": sha256(path)}
    if len(peaks) < 3:
        return row | {"status": "fewer_than_three_peaks"}
    if (np.diff(peaks) <= 0).any():
        return row | {"status": "nonincreasing_peaks"}
    if ((peaks < 0) | (peaks >= 5000)).any():
        return row | {"status": "out_of_record_peaks"}
    intervals = np.diff(peaks) * 1000 / annotation.fs
    return row | {"status": "ok", "peak_mean_ms": intervals.mean(), "peak_sd_ms": intervals.std(ddof=1)}


def run(args):
    fold = pd.read_csv(args.test_index)
    if fold.empty or not fold.ecg_id.is_unique or (fold.ecg_id <= 0).any() or set(fold.strat_fold) != {10}:
        raise ValueError("Use the unique fold-10 index produced by the RR trainer")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        observations = list(pool.map(partial(measure, cache=args.cache, fetch_missing=args.download), fold.ecg_id))
    measurements = pd.DataFrame(observations)
    if not measurements.status.eq("ok").any():
        raise ValueError("No complete ECGDeli records. Supply the released annotation directory or use --download.")
    table = fold.merge(measurements, on="ecg_id", validate="one_to_one", sort=False)
    unig, _ = load_measurements(args.features, table.ecg_id.to_numpy())
    table[list(FEATURES)] = unig
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draft = args.out.with_suffix(".partial")
    table.to_csv(draft, index=False, lineterminator="\n")
    draft.replace(args.out)
    print(table.status.value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="The Uni-G feature CSV used to train the RR runs")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--download", action="store_true", help="Fetch missing official ECGDeli annotations")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
