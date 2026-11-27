"""Instrument the first Flowers-102 epoch of the published LaViD training recipe."""

import argparse
from importlib.metadata import version
import importlib.util
from itertools import chain, islice
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .teacher import write_json
from .train import describe_losses, sha256


AUTHOR_REVISION = "fa689286d168eb7daa9899239461cfffc5517de1"
LAVID_WEIGHT = 50.0
BATCH_SIZE = 16
QUESTION_SHAPE = (50, 5)


class FlowerTrainingImages(Dataset):
    def __init__(self, root):
        split = loadmat(root / "setid.mat")
        self.image_ids = np.concatenate((split["trnid"].ravel(), split["valid"].ravel()))
        labels = loadmat(root / "imagelabels.mat")["labels"].ravel() - 1
        self.labels = labels[self.image_ids - 1].astype(np.int64)
        if len(self.image_ids) != 2040 or len(np.unique(self.image_ids)) != 2040:
            raise ValueError("Expected the author repository's 2,040-image Flowers-102 training split")
        if not np.array_equal(np.bincount(self.labels, minlength=102), np.full(102, 20)):
            raise ValueError("Expected 20 training examples for each of the 102 flower categories")
        self.paths = [root / "jpg" / f"image_{image_id:05d}.jpg" for image_id in self.image_ids]
        self.transform = v2.Compose([
            v2.ToImage(), v2.RandomResizedCrop(224), v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.labels[index])


def author_model(path, classes=102):
    spec = importlib.util.spec_from_file_location("original_lavid_resnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resnet18(pretrained=False, num_classes=classes)


def author_probabilities(classes_path, signatures_path):
    names = list(json.loads(classes_path.read_text(encoding="utf-8")))
    signatures = json.loads(signatures_path.read_text(encoding="utf-8"))
    if len(names) != 102 or set(names) != set(signatures):
        raise ValueError("The author signature bank does not match the author Flowers-102 class order")
    targets = torch.tensor([signatures[name] for name in names], dtype=torch.float32)
    if targets.shape != (102, *QUESTION_SHAPE) or (targets < 0).any() or not torch.isfinite(targets).all():
        raise ValueError("Expected 102 finite nonnegative author signatures with shape 50 by 5")
    if not torch.allclose(targets.sum(-1), torch.ones((102, 50)), atol=0.005):
        raise ValueError("The published target rows are not option probabilities")
    return targets.flatten(1)


def run(args):
    device = torch.device(args.device)
    amp = args.precision == "bfloat16" or (args.precision == "auto" and device.type == "cuda")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    files = {
        "student_source": args.author_source / "lavid/models/resnet.py",
        "classes": args.author_source / "lavid/data/classes/flowers_classes.json",
        "signatures": args.author_source / "questions/flowers/logits/qwen2.5-7b_chat_50-5.json",
        "labels": args.flowers / "imagelabels.mat",
        "split": args.flowers / "setid.mat",
    }
    data = FlowerTrainingImages(args.flowers)
    class_targets = author_probabilities(files["classes"], files["signatures"]).to(device)
    model = author_model(files["student_source"]).to(device).train()
    head = nn.Linear(512, math.prod(QUESTION_SHAPE)).to(device).train()
    optimizer = torch.optim.SGD(chain(model.parameters(), head.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))

    started = time.perf_counter()
    steps = []
    for step, (images, labels) in enumerate(islice(loader, args.steps), start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            features, logits = model(images, is_feat=True)
            auxiliary = head(features[-1])
        ce_loss = F.cross_entropy(logits.float(), labels)
        mse_loss = F.mse_loss(auxiliary.float(), class_targets[labels])
        ce_gradient, = torch.autograd.grad(ce_loss, model.conv1.weight, retain_graph=True)
        aux_gradient, = torch.autograd.grad(mse_loss, model.conv1.weight, retain_graph=True)
        ce_norm = ce_gradient.norm().item()
        aux_norm = aux_gradient.norm().item()
        (ce_loss + LAVID_WEIGHT * mse_loss).backward()
        optimizer.step()

        losses = describe_losses(ce_loss.item(), mse_loss.item(), LAVID_WEIGHT)
        steps.append({"step": step, "records": len(labels), **losses,
                      "ce_encoder_gradient_norm": ce_norm, "aux_encoder_gradient_norm": aux_norm,
                      "weighted_aux_to_ce_encoder_grad_norm_ratio": LAVID_WEIGHT * aux_norm / ce_norm if ce_norm else None})
        if step == 1 or step % 25 == 0:
            print(f"step={step} CE={losses['ce']:.4f} MSE={losses['mse']:.4f} "
                  f"weighted auxiliary loss share={losses['weighted_mse_loss_share']:.4f}", flush=True)

    records = sum(row["records"] for row in steps)
    mean_ce = sum(row["records"] * row["ce"] for row in steps) / records
    mean_mse = sum(row["records"] * row["mse"] for row in steps) / records
    gradient_ratios = [row["weighted_aux_to_ce_encoder_grad_norm_ratio"] for row in steps
                       if row["weighted_aux_to_ce_encoder_grad_norm_ratio"] is not None]
    write_json(args.out, {
        "scope": "Instrumented rerun using the author repository recipe and targets. Only training batches were evaluated.",
        "author_repository": "https://github.com/lliangthomas/lavid",
        "author_revision": AUTHOR_REVISION,
        "input_sha256": {name: sha256(path) for name, path in files.items()},
        "train_split": "Oxford Flowers-102 trnid + valid, as implemented by the author repository",
        "available_train_images": len(data),
        "optimizer": {"name": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
        "student": "Authors' ResNet-18, initialized from scratch",
        "teacher": "Authors' published qwen2.5-7b_chat_50-5 signature bank",
        "lavid_weight": LAVID_WEIGHT, "batch_size": BATCH_SIZE, "question_shape": list(QUESTION_SHAPE),
        "precision": "bfloat16 mixed" if amp else "float32", "device": str(device), "seed": args.seed,
        "software": {name: version(name) for name in ("torch", "torchvision", "scipy")},
        "elapsed_seconds": time.perf_counter() - started,
        "summary": {"processed_images": records, "steps": len(steps),
                    "losses_record_weighted_mean": describe_losses(mean_ce, mean_mse, LAVID_WEIGHT),
                    "median_weighted_aux_to_ce_encoder_grad_norm_ratio": float(np.median(gradient_ratios))
                    if gradient_ratios else None},
        "batches": steps,
    })
    print(f"Saved {len(steps)} training batches to {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-source", type=Path, required=True, help=f"LaViD source at commit {AUTHOR_REVISION}")
    parser.add_argument("--flowers", type=Path, required=True, help="Oxford Flowers-102 root with jpg/ and both original MAT files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, help="Stop after this many training batches")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if (args.steps is not None and args.steps < 1) or args.seed < 0 or args.workers < 0 or args.cpu_threads < 1:
        parser.error("Step and CPU thread counts must be positive, and seed and worker count must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
