"""Build and verify a read-only uint8 archive of a deterministic author evaluation crop."""

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from .audit_visual_lavid import hash_file
from .teacher import write_json
from .train_visual_lavid import BATCH_SIZE, DATASETS, VisualDataset, evaluate


PIXELS_SHAPE = (3, 224, 224)
CROP = v2.Compose([v2.ToImage(), v2.Resize(256), v2.CenterCrop(224)])
NORMALIZE = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])


class CachedVisualEval(Dataset):
    def __init__(self, path, original, require_verified=True, dataset="flowers"):
        self.path = path
        self.ids, self.labels = original.ids, original.labels
        self.cache = None
        if require_verified and not path.with_suffix(".json").is_file():
            raise ValueError("Use a cache with a completed pixel-equality audit")
        if path.with_suffix(".json").is_file():
            self.metadata = json.loads(path.with_suffix(".json").read_bytes())
            if (self.metadata["dataset"] != dataset or self.metadata["image_ids"] != self.ids.tolist()
                    or self.metadata["labels"] != self.labels.tolist()
                    or not self.metadata["normalized_tensor_equality_all_records"]):
                raise ValueError("Cache metadata differs from the original image order or transform")
            if require_verified and hash_file(path) != self.metadata["sha256"]:
                raise ValueError("Cached evaluation bytes differ from the verified archive")
        else:
            self.metadata = None
        probe = np.load(path, mmap_mode="r", allow_pickle=False)
        if probe.shape != (len(self.ids), *PIXELS_SHAPE) or probe.dtype != np.uint8:
            raise ValueError("Unexpected cache shape or pixels")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.cache is None:
            self.cache = np.load(self.path, mmap_mode="r", allow_pickle=False)
        pixels = torch.from_numpy(np.array(self.cache[index], copy=True))
        return NORMALIZE(pixels), int(self.labels[index]), index


def verify_pixels(cached, raw, workers):
    loaders = [DataLoader(data, batch_size=64, num_workers=workers) for data in (cached, raw)]
    checked = 0
    for first, second in zip(*loaders, strict=True):
        for a, b in zip(first, second, strict=True):
            if not torch.equal(a, b):
                raise ValueError("Cached normalized pixel tensors differ from the author loader")
        checked += len(first[1])
    return checked


def build(args, raw):
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if not args.cache.exists():
        estimated_bytes = len(raw) * int(np.prod(PIXELS_SHAPE))
        if shutil.disk_usage(args.cache.parent).free < estimated_bytes + 8 * 1024 ** 2:
            raise ValueError("Insufficient space for the verified cache")
        source = VisualDataset(args.images, args.dataset, False)
        source.transform = CROP
        temporary = args.cache.with_name(f"{args.cache.stem}.{os.getpid()}.tmp.npy")
        images = np.lib.format.open_memmap(temporary, mode="w+", shape=(len(raw), *PIXELS_SHAPE), dtype=np.uint8)
        for x, y, indices in DataLoader(source, batch_size=64, num_workers=args.workers):
            np.testing.assert_array_equal(y.numpy(), source.labels[indices.numpy()])
            images[indices.numpy()] = x.numpy()
        images.flush()
        del images
        temporary.replace(args.cache)
    dataset = CachedVisualEval(args.cache, raw, require_verified=False, dataset=args.dataset)
    count = verify_pixels(dataset, raw, args.workers)
    return {"dataset": args.dataset, "cache_bytes": args.cache.stat().st_size, "sha256": hash_file(args.cache),
            "image_ids": dataset.ids.tolist(), "labels": dataset.labels.tolist(), "images_compared": count,
            "normalized_tensor_equality_all_records": count == len(raw),
            "pixels": "author eval short-edge 256 and center 224, RGB uint8; converts and normalizes on read"}


def check_model(args, raw, cached):
    device = torch.device(args.device)
    classes = DATASETS[args.dataset][0]
    if args.architecture == "resnet18":
        from .visual_probe_extract import load_frozen

        model, state, _ = load_frozen(args.author_source, args.checkpoint, args.dataset, device)
    else:
        from .train_visual_loss_matrix import LossStudent

        model = LossStudent(args.architecture, classes, args.author_source).to(device)
        state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(state["student"], strict=True)
    batch = 64 if args.architecture == "vit_small_patch16" else BATCH_SIZE
    loaders = [DataLoader(data, batch_size=batch, num_workers=args.workers, pin_memory=device.type == "cuda")
               for data in (raw, cached)]
    results = {}
    with TemporaryDirectory() as temporary:
        for precision in ("bfloat16", "float32"):
            scores = []
            for key, loader in zip(("author", "cache"), loaders, strict=True):
                path = Path(temporary) / f"{key}.npz"
                start = time.perf_counter()
                metrics = evaluate(model, loader, device, precision == "bfloat16", classes, path)
                seconds = time.perf_counter() - start
                with np.load(path, allow_pickle=False) as saved:
                    scores.append((metrics, seconds, saved["log_probability"]))
            first, second = scores
            if first[0] != second[0] or not np.array_equal(first[2], second[2]):
                raise ValueError(f"The cache changed full-split {precision} checkpoint outputs or metrics")
            results[precision] = {"metrics": second[0], "author_seconds_including_output": first[1],
                                  "cache_seconds_including_output": second[1], "all_log_probabilities_identical": True}
    return {"checked_checkpoint_sha256": hash_file(args.checkpoint), "checked_checkpoint_epoch": state["epoch"],
            "scores_identical": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="flowers")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--author-source", type=Path)
    parser.add_argument("--architecture", choices=("resnet18", "mobilenetv2", "vit_small_patch16"), default="resnet18")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.cache.suffix != ".npy" or (args.checkpoint and not args.author_source):
        parser.error("Supply a .npy cache and, for model equivalence, the author source path")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    raw = VisualDataset(args.images, args.dataset, False)
    evidence = build(args, raw)
    if args.checkpoint:
        evidence.update(check_model(args, raw, CachedVisualEval(args.cache, raw, require_verified=False, dataset=args.dataset)))
    write_json(args.cache.with_suffix(".json"), evidence)
    print(f"Verified {evidence['images_compared']} original/cached evaluation tensors", flush=True)


if __name__ == "__main__":
    main()
