"""Frozen author image populations and language/label code provenance for the Table 1 suite."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from .teacher import write_json
from .train import sha256
from .train_visual_lavid import RAW_SOURCES, SHAPE, VisualDataset, centered_energy, read_signatures, transforms
from .train_visual_lavid_teacher import FlowerTrainingImages


RELEASES = {
    "flowers": (102, 102, 2040, 6149),
    "cub200": (200, 200, 5994, 5794),
    "fgvc_aircraft": (100, 102, 6667, 3333),
    "stanfordcars": (196, 196, 8144, 8041),
    "oxfordpets": (37, 37, 3680, 3669),
    "caltech101": (101, 101, 4310, 4367),
}
ARCHITECTURES = ("resnet18", "mobilenetv2", "shufflenetv2")
LOSS_WEIGHTS = {
    "flowers": (50., 30., 30.),
    "cub200": (40., 40., 40.),
    "fgvc_aircraft": (50., 20., 10.),
    "stanfordcars": (40., 30., 20.),
    "oxfordpets": (50., 20., 30.),
    "caltech101": (20., 20., 20.),
}
HISTORICAL_TARGETS = {
    "flowers": "7382445c2a58afec39409e8ae5a536c17f226f952e6888a9e6b3ccd6b1eafce9",
    "cub200": "85c87693ac3e85ae2ce426f1ea398b10aa7671c48698a878b17b37fce0e291a5",
}
QWEN_RELEASE_SHA = {
    "flowers": "fdb051c89eabc1c51ab80947251fb7176f64893bbf0641e229028a9490b6d74d",
    "cub200": "6ad69aa1da6e11e5e3b45a2ca78fe5574e7ed603ce6cdd6ca6bd7899d4eebef2",
    "fgvc_aircraft": "2fc11b54601bc249571aba57664f535281d86b62fc6c66a0c57a3188233f3dfc",
    "stanfordcars": "0ca8ed8d568e22cfc341a52e6107b71fc64bcc070265e0a11860f9beb8f3b15a",
    "oxfordpets": "52f087eb73ba8784e2b7c39bcd0079cef7746122f92f26dfcb1ca48982788008",
}


class ManifestImages(VisualDataset):
    def __init__(self, root, rows, training):
        self.ids = np.array([int(row["image_id"]) for row in rows], dtype=np.int64)
        self.labels = np.array([int(row["class_id"]) for row in rows], dtype=np.int64)
        paths = [PurePosixPath(row["image_path"]) for row in rows]
        if any(path.is_absolute() or ".." in path.parts for path in paths):
            raise ValueError("Manifest image paths must be relative to the released data")
        self.paths = [root.joinpath(*path.parts) for path in paths]
        self.transform = transforms(training)


def load_names(dataset, author, cohort=None):
    classes, outputs, _, _ = RELEASES[dataset]
    if cohort is None:
        if dataset not in RAW_SOURCES:
            raise ValueError(f"{dataset} requires a locked cohort CSV and sidecar JSON")
        names = json.loads((author / f"lavid/data/classes/{dataset}_classes.json").read_bytes())
        qwen_names = names
    else:
        metadata = json.loads(cohort.with_suffix(".json").read_bytes())
        names = metadata["class_names"]
        qwen_names = metadata.get("qwen_names", names)
        if metadata.get("classifier_outputs", metadata.get("output_classes")) != outputs:
            raise ValueError(f"Incorrect author classifier size in {cohort}")
    if len(names) != classes or len(set(names)) != classes or len(qwen_names) != classes:
        raise ValueError(f"Unexpected classifier or language class index for {dataset}")
    return names, qwen_names


def populations(dataset, images, cohort=None, rgb_cache=None, val_cache=None):
    true_classes, _, train_count, test_count = RELEASES[dataset]
    if cohort is None:
        if dataset not in RAW_SOURCES:
            raise ValueError(f"{dataset} requires a frozen cohort CSV")
        train = FlowerTrainingImages(images, rgb_cache) if dataset == "flowers" else VisualDataset(images, dataset, True)
        test = VisualDataset(images, dataset, False)
        metadata = {name: sha256(images / name) for name in RAW_SOURCES[dataset]}
    else:
        if rgb_cache:
            raise ValueError("This original-resolution RGB cache is bound to the official Flowers IDs")
        sidecar = cohort.with_suffix(".json")
        cohort_sha = sha256(cohort)
        if json.loads(sidecar.read_bytes())["cohort_sha256"] != cohort_sha:
            raise ValueError(f"Images and classes must match the released cohort: {cohort}")
        with cohort.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != train_count + test_count or {row["split"] for row in rows} != {"train", "test"}:
            raise ValueError(f"Unexpected split names or population size at {cohort}")
        train, test = (ManifestImages(images, [r for r in rows if r["split"] == split], split == "train")
                       for split in ("train", "test"))
        metadata = {"cohort_csv": cohort_sha, "cohort_json": sha256(sidecar)}
    ids = np.concatenate((train.ids, test.ids))
    if (len(train) != train_count or len(test) != test_count or len(np.unique(ids)) != len(ids) or
            set(train.labels) != set(range(true_classes)) or set(test.labels) != set(range(true_classes))):
        raise ValueError(f"Unexpected source IDs, class support or split counts for {dataset}")
    if val_cache:
        from .visual_validation_cache import CachedVisualEval

        test = CachedVisualEval(val_cache, test, dataset=dataset)
    return train, test, metadata


def soft_code(manifest_path, dataset, names):
    archive = json.loads(manifest_path.read_bytes())
    if archive["class_names"] != names or archive["ecoc_seed"] != 271828:
        raise ValueError(f"Code class rows do not follow the frozen {dataset} order")
    symbols = np.asarray(archive["ecoc_symbols_zero_based"])
    scale = archive["ecoc_scale_about_uniform"]
    if (symbols.shape != (len(names), SHAPE[0]) or not np.issubdtype(symbols.dtype, np.integer) or
            (symbols < 0).any() or (symbols >= SHAPE[1]).any() or not 0 < scale <= 1):
        raise ValueError(f"Invalid five-way target format in {manifest_path}")
    if not np.array_equal(symbols, balanced_symbols(len(names))):
        raise ValueError(f"The declared soft code was not built from the frozen balanced codebook: {manifest_path}")
    rows = .2 + scale * (np.eye(SHAPE[1], dtype=np.float64)[symbols] - .2)
    digest = hashlib.sha256(rows.tobytes()).hexdigest()
    if digest != HISTORICAL_TARGETS.get(dataset, archive.get("target_float64_sha256")):
        raise ValueError(f"The stored code does not match the frozen target bytes for {dataset}")
    return rows


def language_target(dataset, signature, qwen_names):
    expected = QWEN_RELEASE_SHA.get(dataset)
    if expected and sha256(signature) != expected:
        raise ValueError(f"Use the fixed author's released Qwen2.5-7B scores for {dataset}")
    return read_signatures(signature, qwen_names)


def balanced_symbols(classes):
    rng = np.random.default_rng(271828)
    return np.column_stack([rng.permutation((np.arange(classes) + q) % SHAPE[1]) for q in range(SHAPE[0])])


def make_code(dataset, names, qwen_names, signature, output):
    qwen = language_target(dataset, signature, qwen_names)
    symbols = balanced_symbols(len(names))
    hard = np.eye(SHAPE[1], dtype=np.float64)[symbols]
    scale = math.sqrt(centered_energy(qwen) / centered_energy(hard))
    if not 0 < scale <= 1:
        raise ValueError("The original variance-matching rule would give invalid soft targets")
    rows = .2 + scale * (hard - .2)
    digest = hashlib.sha256(rows.tobytes()).hexdigest()
    if dataset in HISTORICAL_TARGETS and digest != HISTORICAL_TARGETS[dataset]:
        raise ValueError(f"The regenerated code differs from the historical {dataset} target")
    manifest = {"dataset": dataset, "class_names": names, "ecoc_seed": 271828,
                "ecoc_symbols_zero_based": symbols.tolist(), "ecoc_scale_about_uniform": float(scale),
                "scale_reference_file": signature.name, "scale_reference_sha256": sha256(signature),
                "qwen_centered_energy": centered_energy(qwen), "code_centered_energy": centered_energy(rows),
                "target_float64_sha256": digest}
    if output.exists() and json.loads(output.read_bytes()) != manifest:
        raise ValueError(f"Refusing to replace an existing frozen {dataset} code")
    write_json(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=RELEASES, required=True)
    for name in ("author-source", "cohort", "signature", "out"):
        parser.add_argument(f"--{name}", type=Path, required=name != "cohort")
    args = parser.parse_args()
    names, qwen_names = load_names(args.dataset, args.author_source, args.cohort)
    make_code(args.dataset, names, qwen_names, args.signature, args.out)


if __name__ == "__main__":
    main()
