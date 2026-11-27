"""Make a fixed external lead-II manual interval reference and 100 Hz waveforms."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import wfdb

from . import LEADS
from .prepare import download_file

BASE = "https://physionet-open.s3.amazonaws.com/ludb/1.0.1/"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(identifier, cache, manifest):
    record = f"data/{identifier}"
    for ext in ("hea", "dat", "ii"):
        name = f"{record}.{ext}"
        path = cache / name
        download_file(cache, name, base_url=BASE)
        if sha(path) != manifest[name]:
            raise ValueError(f"Incorrect published checksum for {name}")
    signal, meta = wfdb.rdsamp(str(cache / record))
    assert signal.shape == (5000, 12) and meta["fs"] == 500
    assert list(map(str.lower, meta["sig_name"])) == list(map(str.lower, LEADS))
    assert all(unit == "mV" for unit in meta["units"]) and np.isfinite(signal).all()
    annotation = wfdb.rdann(str(cache / record), "ii")
    symbols = np.asarray(annotation.symbol)
    peaks = np.asarray(annotation.sample)[symbols == "N"]
    assert set(symbols).issubset({"(", ")", "p", "N", "t"})
    assert np.all(np.diff(peaks) > 0) and np.all((peaks >= 0) & (peaks < 5000))
    row = {"id": identifier, "manual_qrs_count": len(peaks), "manual_rr_count": max(0, len(peaks) - 1),
           "manual_qrs_samples": " ".join(str(p) for p in peaks),
           "annotation_start_s": annotation.sample[0] / 500, "annotation_end_s": annotation.sample[-1] / 500,
           "peak_start_s": peaks[0] / 500 if len(peaks) else None,
           "peak_end_s": peaks[-1] / 500 if len(peaks) else None}
    if len(peaks) >= 3:
        rr = np.diff(peaks) * 2.0
        row.update(rr_mean_ms=rr.mean(), rr_sd_ms=rr.std(ddof=1), valid=True)
    else:
        row.update(rr_mean_ms=np.nan, rr_sd_ms=np.nan, valid=False)
    waveform = resample_poly(signal.T, 1, 5, axis=1, window=("kaiser", 5), padtype="constant").astype(np.float32)
    assert waveform.shape == (12, 1000)
    return row, waveform, Counter(symbols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    (args.cache / "data").mkdir(exist_ok=True, parents=True)
    args.out.mkdir(exist_ok=True, parents=True)
    for name in ("SHA256SUMS.txt", "RECORDS", "ludb.csv", "README", "ANNOTATORS"):
        download_file(args.cache, name, base_url=BASE)
    manifest = {line.split()[1]: line.split()[0]
                for line in (args.cache / "SHA256SUMS.txt").read_text().splitlines()}
    for name in ("RECORDS", "ludb.csv", "README", "ANNOTATORS"):
        assert sha(args.cache / name) == manifest[name]
    ids = [int(Path(x).name) for x in (args.cache / "RECORDS").read_text().splitlines()]
    assert ids == list(range(1, 201))
    rows, waveforms, counts = [], [], Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row, signal, symbols in executor.map(lambda i: prepare(i, args.cache, manifest), ids):
            rows.append(row)
            waveforms.append(signal)
            counts.update(symbols)
    metadata = pd.read_csv(args.cache / "ludb.csv")
    table = pd.DataFrame(rows).merge(metadata, left_on="id", right_on="ID", validate="one_to_one")
    assert len(table) == len(ids) and np.array_equal(table.id, ids)
    table["sinus_subgroup"] = (table["Rhythms"].str.strip().eq("Sinus rhythm")
                               & table["Extrasystolies"].fillna("").str.strip().eq("")
                               & table["Cardiac pacing"].fillna("").str.strip().eq(""))
    table["peak_span_s"] = table.peak_end_s - table.peak_start_s
    table["annotation_span_s"] = table.annotation_end_s - table.annotation_start_s
    table.to_csv(args.out / "manual_reference.csv", index=False, lineterminator="\n")
    np.savez_compressed(args.out / "waveforms.npz", record_id=np.array(ids), leads=np.array(LEADS),
                        waveforms=np.stack(waveforms), sample_rate=np.array(100))
    summary = {"source": "https://physionet.org/content/ludb/1.0.1/", "records": len(table),
               "valid": int(table.valid.sum()), "valid_sinus_subgroup": int((table.valid & table.sinus_subgroup).sum()),
               "zero_sd": int(table.rr_sd_ms.eq(0).sum()), "annotation_symbol_counts": dict(counts),
               "wfdb_version": wfdb.__version__, "release_manifest_sha256": sha(args.cache / "SHA256SUMS.txt"),
               "verified_release_files": len(ids)*3+4, "source_metadata_sha256": manifest["ludb.csv"],
               "waveforms_sha256": sha(args.out / "waveforms.npz"), "reference_sha256": sha(args.out / "manual_reference.csv"),
               "distribution": {name: table[name].quantile([0, .25, .5, .75, 1]).to_dict() for name in (
                   "manual_qrs_count", "peak_start_s", "peak_end_s", "peak_span_s", "annotation_span_s", "rr_mean_ms", "rr_sd_ms"
               )}}
    (args.out / "source_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
