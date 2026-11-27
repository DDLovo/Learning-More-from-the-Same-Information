"""Extract the frozen author's 512-D pooled representation on the official image splits."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .teacher import write_json
from .train import sha256
from .train_visual_lavid import DATASETS, VisualDataset, transforms
from .train_visual_lavid import NUM_OUTPUTS, state_hash, write_checkpoint
from .visual_reference import AUTHOR_REVISION, author_model


def load_frozen(author_source, checkpoint, dataset, device):
    n, _, _, _ = DATASETS[dataset]
    source = author_source / "lavid/models/resnet.py"
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model = author_model(source, n)
    model.load_state_dict(state["student"], strict=True)
    return model.to(device).eval(), state, source


def clean_dataset(root, dataset, train):
    data = VisualDataset(root, dataset, train)
    data.transform = transforms(False)
    return data


@torch.inference_mode()
def representations(model, data, device, batch_size, workers):
    loader = DataLoader(data, batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    codes = []
    for inputs, labels, positions in loader:
        features, logits = model(inputs.to(device, non_blocking=True), is_feat=True)
        if features[-1].shape != (len(labels), 512) or features[-1].dtype != torch.float32:
            raise ValueError("Expected the clean FP32 512-D penultimate pooled representation")
        np.testing.assert_array_equal(labels, data.labels[positions.numpy()])
        codes.append(features[-1].cpu().numpy())
    return np.concatenate(codes)


def reconstruct_initialization(args):
    reference = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    torch.manual_seed(args.initialization_seed)
    n = DATASETS[args.dataset][0]
    student = author_model(args.author_source / "lavid/models/resnet.py", n)
    auxiliary = torch.nn.Linear(512, NUM_OUTPUTS)
    digest = state_hash(student, auxiliary)
    if digest != reference["initial_state_sha256"]:
        raise ValueError("Reconstructed epoch 0 does not match the trained model's recorded shared initialization")
    args.save_initialization.parent.mkdir(parents=True, exist_ok=True)
    write_checkpoint(args.save_initialization, {
        "epoch": 0, "student": student.state_dict(), "auxiliary": auxiliary.state_dict(),
        "initial_state_sha256": digest, "verified_against_checkpoint_sha256": sha256(args.checkpoint),
    })
    return args.save_initialization


def run(args):
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    source_checkpoint = reconstruct_initialization(args) if args.initialization_seed is not None else args.checkpoint
    model, state, source = load_frozen(args.author_source, source_checkpoint, args.dataset, device)
    arrays = {}
    for part, training in (("train", True), ("test", False)):
        data = clean_dataset(args.images, args.dataset, training)
        arrays[f"{part}_ids"] = data.ids
        arrays[f"{part}_y"] = data.labels
        arrays[f"{part}_features"] = representations(model, data, device, args.batch_size, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(args.out)
    write_json(args.out.with_suffix(".json"), {
        "dataset": args.dataset, "author_revision": AUTHOR_REVISION, "author_resnet_sha256": sha256(source),
        "checkpoint_name": source_checkpoint.name, "checkpoint_sha256": sha256(source_checkpoint),
        "checkpoint_epoch": state["epoch"], "initial_state_sha256": state["initial_state_sha256"],
        "features": "512-D penultimate global average pooled features, before classification/auxiliary heads",
        "image_inputs": "RGB, original author evaluation resize-short-edge 256, center crop 224, normalization; both splits",
        "precision": "FP32, no TF32 convolution", "batch_size": args.batch_size,
        "reconstructed_initialization_reference_sha256": state.get("verified_against_checkpoint_sha256"),
        "rows": {part: len(arrays[f"{part}_ids"]) for part in ("train", "test")},
        "ordered_ids_sha256": {part: hashlib.sha256(arrays[f"{part}_ids"].astype("<i8").tobytes()).hexdigest()
                               for part in ("train", "test")},
        "archive_sha256": sha256(args.out),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("author-source", "images", "checkpoint", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initialization-seed", type=int)
    parser.add_argument("--save-initialization", type=Path)
    args = parser.parse_args()
    if args.out.suffix != ".npz" or min(args.batch_size, args.cpu_threads) < 1 or args.workers < 0:
        parser.error("Expected an NPZ output and positive loader/runtime settings")
    if (args.initialization_seed is None) != (args.save_initialization is None):
        parser.error("Initialization verification requires both a seed and a separate output checkpoint")
    if args.save_initialization and args.save_initialization.resolve() == args.checkpoint.resolve():
        parser.error("The reference checkpoint and initialization output must be distinct")
    run(args)


if __name__ == "__main__":
    main()
