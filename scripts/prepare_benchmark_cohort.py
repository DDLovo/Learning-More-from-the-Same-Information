"""Freeze the ordered image IDs and original annotations for the author vision grid."""
import argparse
import ast
from collections import Counter
import csv
from hashlib import file_digest
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from scipy.io import loadmat

SIZES = {"flowers": (2040, 6149, 102, 102), "cub200": (5994, 5794, 200, 200),
         "fgvc_aircraft": (6667, 3333, 100, 102), "stanfordcars": (8144, 8041, 196, 196),
         "oxfordpets": (3680, 3669, 37, 37), "caltech101": (4310, 4367, 101, 101)}


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def table(path):
    rows = [line.split(maxsplit=1) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed = {int(key): value for key, value in rows}
    if len(parsed) != len(rows):
        raise ValueError(f"Duplicate IDs in the source table: {path}")
    return parsed


def flowers(root):
    raw = loadmat(root / "setid.mat")
    labels = loadmat(root / "imagelabels.mat")["labels"].ravel() - 1
    train = np.concatenate((raw["trnid"].ravel(), raw["valid"].ravel()))
    test = raw["tstid"].ravel()
    data = [(int(i), split, int(labels[i - 1]), f"jpg/image_{i:05d}.jpg") for split, seq in (("train",train),("test",test)) for i in seq]
    return data, [root / "setid.mat", root / "imagelabels.mat"]


def cub200(root):
    files = table(root / "images.txt")
    classes = table(root / "image_class_labels.txt")
    split = table(root / "train_test_split.txt")
    if set(files) != set(classes) or set(split) != set(files):
        raise ValueError("Original CUB metadata files disagree")
    data = [(i, key, int(classes[i]) - 1, f"images/{name}") for key,v in (("train","1"),("test","0"))
            for i,name in files.items() if split[i] == v]
    return data, [root / x for x in ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")]


def caltech101(root, author):
    import torch

    # Use the directory names in the pinned loader, including both face classes.
    tree = ast.parse((author / "lavid/data/custom_loaders.py").read_bytes())
    loader = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CustomCaltechLoader")
    folders = next(ast.literal_eval(node.value) for node in ast.walk(loader)
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Attribute) and target.attr == "classes" for target in node.targets))
    catalogue = []
    for label, folder in enumerate(folders):
        directory = root / "101_ObjectCategories" / folder
        images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
        catalogue.extend((label, path.relative_to(root).as_posix()) for path in images)
    train_count, test_count, _, _ = SIZES["caltech101"]
    if len(catalogue) != train_count + test_count:
        raise ValueError("Incorrect image population in the Caltech-101 source")
    order = torch.randperm(len(catalogue), generator=torch.Generator().manual_seed(27)).tolist()
    return [(i + 1, "train" if offset < train_count else "test", *catalogue[i]) for offset, i in enumerate(order)], []


def from_torchvision(root, name):
    from torchvision.datasets import FGVCAircraft, OxfordIIITPet, StanfordCars
    records = []
    for split, image_ids in (("train", 1), ("test", 0)):
        if name == "fgvc_aircraft":
            ds = FGVCAircraft(root=root, split="trainval" if image_ids else "test", annotation_level="variant", download=False)
            images = zip(ds._image_files, ds._labels, strict=True)
        elif name == "stanfordcars":
            ds = StanfordCars(root=root, split=split, download=False)
            images = ds._samples
        else:
            ds = OxfordIIITPet(root=root, split="trainval" if image_ids else "test", download=False)
            images = zip(ds._images, ds._labels, strict=True)
        for path, target in images:
            file = Path(path)
            row_id = int(file.stem) if name == "fgvc_aircraft" else len(records) + 1
            records.append((row_id, split, int(target), file.relative_to(root).as_posix()))
    manifests = {
        "fgvc_aircraft": ["fgvc-aircraft-2013b/data/" + n for n in ("images_variant_trainval.txt","images_variant_test.txt","variants.txt")],
        "stanfordcars": ["stanford_cars/" + n for n in ("devkit/cars_train_annos.mat","cars_test_annos_withlabels.mat","devkit/cars_meta.mat")],
        "oxfordpets": ["oxford-iiit-pet/annotations/" + n for n in ("trainval.txt","test.txt")],
    }
    return records, [root / n for n in manifests[name]]


def prepare(args):
    root, author, out = args.images.resolve(strict=True), args.author.resolve(strict=True), args.out
    true_train, true_test, nclasses, outputs = SIZES[args.dataset]
    if args.dataset in ("flowers", "cub200"):
        source = flowers if args.dataset == "flowers" else cub200
        records, metadata = source(root)
    elif args.dataset == "caltech101":
        records, metadata = caltech101(root, author)
    else:
        records, metadata = from_torchvision(root, args.dataset)
    sizes = Counter(r[1] for r in records)
    if sizes != Counter(train=true_train, test=true_test):
        raise ValueError(f"Unexpected true cohort sizes: {sizes}")
    ids = [r[0] for r in records]
    paths = [r[3] for r in records]
    if len(set(paths)) != len(paths) or len(set(ids)) != len(ids) or min(ids) < 1:
        raise ValueError("Official cohorts must have unique split-aware image IDs and unique paths")
    for key in ("train", "test"):
        if set(r[2] for r in records if r[1] == key) != set(range(nclasses)):
            raise ValueError(f"Unexpected class IDs in the {key} fold")
    for value in paths:
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or not (root / value).is_file():
            raise ValueError(f"Unsafe or missing official source image: {value}")
    samples = []
    for idx in (0, true_train // 2, true_train - 1, true_train, len(records) - 1):
        file = root / paths[idx]
        with Image.open(file) as image:
            image.convert("RGB").load()
            samples.append({"image_id": ids[idx], "image_path": paths[idx], "size": list(image.size), "sha256": digest(file)})
    names = author / f"lavid/data/classes/{args.dataset}_classes.json"
    classes = json.loads(names.read_bytes())
    if isinstance(classes, dict):
        classes = list(classes.values())
    qwen_names = classes
    if args.dataset == "caltech101":
        qwen_names = json.loads((author / "lavid/data/classes/caltech101_classes_duplicate_face.json").read_bytes())
        if len(classes) != nclasses - 1 or qwen_names != [classes[0], *classes]:
            raise ValueError("Expected the shared face signature for labels 0 and 1")
        classes = ["face (Faces)", "face (Faces_easy)", *classes[1:]]
    if len(classes) != nclasses or len(set(classes)) != nclasses:
        raise ValueError(f"The author class list has {len(classes)} rows rather than the observed {nclasses}")
    out.mkdir(parents=True, exist_ok=True)
    draft = out / "cohort.csv.partial"
    with draft.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id","split","class_id","image_path"))
        writer.writerows(records)
    csv_path = out / "cohort.csv"
    cohort = {"dataset": args.dataset, "train_records": true_train, "test_records": true_test,
              "classifier_outputs": outputs, "observed_classes": nclasses, "class_names": classes, "qwen_names": qwen_names,
              "cohort_sha256": digest(draft), "author_class_names_sha256": digest(names),
              "official_metadata": {p.relative_to(root).as_posix(): digest(p) for p in metadata}, "samples": samples}
    if args.dataset == "caltech101":
        cohort["split_seed"] = 27
    json_file = out / "cohort.json"
    if json_file.is_file() and json.loads(json_file.read_bytes()) != cohort:
        raise ValueError("This data path already has an incompatible frozen cohort")
    draft.replace(csv_path)
    temporary = json_file.with_suffix(".partial")
    temporary.write_text(json.dumps(cohort, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(json_file)
    print(f"Frozen {args.dataset}: {true_train}/{true_test}, {nclasses} true classes, {outputs} classifier outputs, {cohort['cohort_sha256']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=tuple(SIZES), required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--author", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    prepare(p.parse_args())
