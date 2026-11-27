"""Build the single-rhythm dataset from PTB-XL 1.0.3 WFDB records."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import wfdb

from . import CLASSES, LEADS, N_SAMPLES, SAMPLE_RATE


SOURCE = "https://physionet.org/files/ptb-xl/1.0.3/"
METADATA_SHA256 = {
    "ptbxl_database.csv": "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp_statements.csv": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}


def download_file(root, relative, *, base_url=SOURCE, allow_missing=False):
    destination = root / relative
    if destination.is_file() and destination.stat().st_size:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    for delay in (0, 1, 3, 8):
        time.sleep(delay)
        try:
            with urlopen(base_url + relative, timeout=25) as response:
                content = response.read()
            if not content:
                raise OSError("Empty download")
        except HTTPError as error:
            if allow_missing and error.code == 404:
                return False
            failure = error
        except (URLError, OSError) as error:
            failure = error
        else:
            with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            temporary_path.replace(destination)
            return True
    raise OSError(f"Could not download {relative}") from failure


def make_cohort(root):
    columns = ["ecg_id", "patient_id", "strat_fold", "scp_codes", "filename_lr"]
    records = pd.read_csv(root / "ptbxl_database.csv", usecols=columns)
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    rhythms = set(statements.index[statements["rhythm"] == 1])
    if not set(CLASSES) <= rhythms:
        raise ValueError("Expected the official rhythm flags for all four target classes")
    if records[columns].isna().any().any() or not records["ecg_id"].is_unique:
        raise ValueError("Missing PTB-XL metadata or duplicate ECG IDs")
    if not records["strat_fold"].isin(range(1, 11)).all():
        raise ValueError("PTB-XL strat_fold must be in 1..10")
    if (records.groupby("patient_id")["strat_fold"].nunique() > 1).any():
        raise ValueError("At least one PTB-XL patient appears in multiple official folds")

    def label(codes):
        matched = rhythms.intersection(ast.literal_eval(codes))
        if len(matched) == 1:
            code = next(iter(matched))
            if code in CLASSES:
                return code

    records["label"] = records["scp_codes"].map(label)
    cohort = records.loc[records["label"].notna(), ["ecg_id", "patient_id", "strat_fold", "label", "filename_lr"]]
    cohort = cohort.sort_values("ecg_id").reset_index(drop=True)
    cohort["patient_id"] = cohort["patient_id"].astype(np.int64)
    cohort["split"] = cohort["strat_fold"].map({**{k: "train" for k in range(1, 9)}, 9: "validation", 10: "test"})
    return cohort


def load_record(root, relative, download):
    if download:
        download_file(root, relative + ".hea")
        download_file(root, relative + ".dat")
    record = wfdb.rdrecord(str(root / relative), physical=True, return_res=32)
    names = [lead.upper() for lead in record.sig_name]
    if record.fs != SAMPLE_RATE or record.p_signal.shape != (N_SAMPLES, len(LEADS)):
        raise ValueError(f"Unexpected sample rate or signal shape for {relative}")
    if set(names) != set(LEADS) or any(unit.lower() != "mv" for unit in record.units):
        raise ValueError(f"Unexpected lead names or units for {relative}")
    signal = record.p_signal[:, [names.index(lead) for lead in LEADS]].T
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite ECG values in {relative}")
    return signal


def prepare(args):
    if args.download_workers:
        for filename in METADATA_SHA256:
            download_file(args.ptbxl, filename)

    metadata_hashes = {}
    for filename in METADATA_SHA256:
        with (args.ptbxl / filename).open("rb") as stream:
            metadata_hashes[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    official_metadata = metadata_hashes == METADATA_SHA256
    if not official_metadata:
        print("PTB-XL metadata do not byte-match the official 1.0.3 release. Recording the actual hashes.", flush=True)
    cohort = make_cohort(args.ptbxl)
    counts = {}
    for split in ("train", "validation", "test"):
        split_cohort = cohort.loc[cohort["split"] == split]
        per_class = split_cohort["label"].value_counts().reindex(CLASSES, fill_value=0)
        if not (per_class > 0).all():
            raise ValueError(f"Missing classes in {split}: {per_class.to_dict()}")
        counts[split] = dict(records=len(split_cohort), patients=split_cohort["patient_id"].nunique(),
                             per_class=per_class.to_dict())
    print(json.dumps(counts), flush=True)

    waveforms = np.empty((len(cohort), len(LEADS), N_SAMPLES), dtype=np.float32)

    def read(index):
        waveforms[index] = load_record(args.ptbxl, filenames[index], args.download_workers > 0)

    filenames = cohort["filename_lr"].tolist()
    with ThreadPoolExecutor(max_workers=args.download_workers or args.read_workers) as pool:
        for number, _ in enumerate(pool.map(read, range(len(cohort))), start=1):
            if number % 1_000 == 0:
                print(f"Read {number:,}/{len(cohort):,} ECGs", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.out.parent, delete=False) as temporary:
        np.savez(
            temporary, waveforms=waveforms, labels=cohort["label"].map({code: idx for idx, code in enumerate(CLASSES)}).to_numpy(np.int64),
            ecg_id=cohort["ecg_id"].to_numpy(np.int64), patient_id=cohort["patient_id"].to_numpy(np.int64),
            folds=cohort["strat_fold"].to_numpy(np.int64), class_codes=np.array(tuple(CLASSES)), leads=np.array(LEADS),
        )
        temporary_path = Path(temporary.name)
    temporary_path.replace(args.out)
    cohort.to_csv(args.out.with_suffix(".cohort.csv"), index_label="array_index")
    args.out.with_suffix(".cohort.json").write_text(
        json.dumps({"reference_source": SOURCE, "metadata_sha256": metadata_hashes,
                    "matches_official_metadata": official_metadata, "counts": counts}, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl", type=Path, required=True, help="PTB-XL 1.0.3 dataset root")
    parser.add_argument("--out", type=Path, required=True, help="Output NPZ path")
    parser.add_argument("--download-workers", type=int, default=0, help="Fetch missing cohort files from PhysioNet")
    parser.add_argument("--read-workers", type=int, default=4, help="Parallel readers for local WFDB files")
    args = parser.parse_args()
    if args.download_workers < 0 or args.read_workers < 1:
        parser.error("--download-workers must be nonnegative and --read-workers must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
