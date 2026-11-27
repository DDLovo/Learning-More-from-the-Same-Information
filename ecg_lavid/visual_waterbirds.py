"""Read the official Waterbirds-95 splits and score bird/background groups."""

import csv
from pathlib import PurePosixPath

import numpy as np
import torch

from .train_visual_lavid import VisualDataset, metrics, transforms


DATASET = "waterbirds"
PRIMARY_CLASSES = 2
SPECIES = 200
SPLIT_SIZES = {"train": 4795, "val": 1199, "test": 5794}
GROUP_NAMES = ("landbird_land", "landbird_water", "waterbird_land", "waterbird_water")


class WaterbirdsDataset(VisualDataset):
    def __init__(self, root, split):
        fields = {"img_id", "img_filename", "split", "y", "place"}
        split_number = tuple(SPLIT_SIZES).index(split)
        with (root / "metadata.csv").open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not fields <= set(reader.fieldnames or ()):
                raise ValueError("Use the official Waterbirds metadata.csv with CUB IDs")
            rows = list(reader)
        split_numbers = np.asarray([int(row["split"]) for row in rows])
        all_ids = [int(row["img_id"]) for row in rows]
        all_paths = [row["img_filename"] for row in rows]
        if (set(all_ids) != set(range(1, sum(SPLIT_SIZES.values()) + 1))
                or len(set(all_paths)) != len(rows)
                or set(split_numbers) != {0, 1, 2}
                or tuple(np.bincount(split_numbers)) != tuple(SPLIT_SIZES.values())):
            raise ValueError("Waterbirds split counts, file names or global CUB IDs differ from the official release")
        selected = [row for row, number in zip(rows, split_numbers, strict=True) if number == split_number]
        relative = [PurePosixPath(row["img_filename"]) for row in selected]
        if any(len(path.parts) != 2 or path.is_absolute() or ".." in path.parts for path in relative):
            raise ValueError("Expected an official bird species directory and image filename in each record")
        self.ids = np.asarray([int(row["img_id"]) for row in selected], dtype=np.int64)
        self.paths = [root.joinpath(*path.parts) for path in relative]
        self.labels = np.asarray([int(row["y"]) for row in selected], dtype=np.int64)
        self.backgrounds = np.asarray([int(row["place"]) for row in selected], dtype=np.int64)
        self.auxiliary_labels = np.asarray([int(path.parts[0].split(".", 1)[0]) - 1 for path in relative], dtype=np.int64)
        if (set(self.labels) != {0, 1} or set(self.backgrounds) != {0, 1}
                or self.auxiliary_labels.min() < 0 or self.auxiliary_labels.max() >= SPECIES):
            raise ValueError("Expected binary bird/background IDs and CUB species directories numbered 001–200")
        self.groups = 2 * self.labels + self.backgrounds
        if set(self.groups) != set(range(len(GROUP_NAMES))):
            raise ValueError("Each official split must contain all four bird/background groups")
        self.transform = transforms(split == "train")


@torch.inference_mode()
def evaluate_waterbirds(model, loader, device, amp, training_group_counts, path=None):
    model.eval()
    outputs = []
    for images, _, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images.to(device, non_blocking=True))
        outputs.append(logits.float().log_softmax(-1).cpu())
    logs = torch.cat(outputs).numpy()
    cohort = loader.dataset
    labels = cohort.labels
    predicted = logs.argmax(axis=1)
    counts = np.bincount(cohort.groups, minlength=len(GROUP_NAMES))
    correct = np.bincount(cohort.groups[predicted == labels], minlength=len(GROUP_NAMES))
    accuracy = correct / counts
    if path is not None:
        np.savez_compressed(path, image_id=cohort.ids, y_true=labels, log_probability=logs,
                            group_id=cohort.groups, background_id=cohort.backgrounds,
                            species_id=cohort.auxiliary_labels)
    return metrics(logs, labels, PRIMARY_CLASSES) | {
        "group_macro_top1": float(accuracy.mean()),
        "worst_group_top1": float(accuracy.min()),
        "best_group_top1": float(accuracy.max()),
        "train_distribution_adjusted_top1": float(np.average(accuracy, weights=training_group_counts)),
        "groups": {group: {"total": int(n), "correct": int(k), "top1": float(value)}
                   for group, n, k, value in zip(GROUP_NAMES, counts, correct, accuracy, strict=True)},
    }
