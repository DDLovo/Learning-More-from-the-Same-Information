"""Train the matched ViT-B/16 five-way class-code arm on Flowers-102."""

import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path

import numpy as np
import torch
import torchvision

import train as pair
from ecg_lavid.visual_benchmark_data import make_code


DATASET = "flowers"
ARM = "ecoc"
recipe = pair.recipe


def read(path):
    return json.loads(path.read_bytes())


def flowers_code(author_source, path):
    dataset = pair.PROVENANCE["datasets"][DATASET]
    qwen = dataset["qwen_target"]
    class_order = author_source / qwen["class_order"]
    if recipe.sha256(class_order) != qwen["class_order_sha256"]:
        raise ValueError("Use the Flowers class order from the pinned LaViD checkout")
    names = read(class_order)
    make_code(DATASET, names, names, author_source / qwen["source"], path)
    target = recipe.frozen_code(path, dataset["classes"], DATASET, author_source)
    digest = hashlib.sha256(target.astype("<f8", copy=False).tobytes()).hexdigest()
    if digest != dataset["five_way_code"]["matrix_float64_le_sha256"]:
        raise ValueError("The regenerated Flowers code differs from the fitted target")
    return target, digest


def measure_h(args, student, auxiliary, training, target, device):
    loader, order = recipe.training_loader(args, training, device)
    order.epoch = 1
    recipe.seed_epoch(args.seed, 1)
    student.train()
    auxiliary.train()
    fingerprint = hashlib.sha256()
    grad_ce = torch.zeros_like(student.input_weight)
    grad_mse = torch.zeros_like(student.input_weight)
    losses = torch.zeros(2, dtype=torch.float64, device=device)
    count = 0
    for images, labels, _ in islice(loader, args.effective_batch // args.physical_batch):
        fingerprint.update(images.numpy().tobytes())
        ce, mse, _ = recipe.batch_losses(
            student, auxiliary, images.to(device, non_blocking=True), labels.to(device, non_blocking=True),
            target, ARM, True,
        )
        fraction = len(labels) / args.effective_batch
        losses += torch.stack((ce.detach(), mse.detach())).double() * fraction
        (ce_grad,) = torch.autograd.grad(ce, student.input_weight, retain_graph=True)
        (mse_grad,) = torch.autograd.grad(mse, student.input_weight)
        grad_ce.add_(ce_grad.detach(), alpha=fraction)
        grad_mse.add_(mse_grad.detach(), alpha=fraction)
        count += len(labels)
    first = recipe.first_observation(losses, grad_ce, grad_mse, 1.0, fingerprint.hexdigest())
    if (count != args.effective_batch or not first["finite"]
            or min(first["input_weight_grad_ce"], first["input_weight_grad_auxiliary"]) <= 0):
        raise FloatingPointError("Expected finite, nonzero H gradients from the complete first effective batch")
    return first | {"records": count}


def calibrate_h(args, classes, training, target, device, binding, ce_dir, reference):
    path = args.initial_state.with_name("calibration.json")
    ce_state, ce_first = (read(ce_dir / name) for name in ("initial-state.json", "first_batch.json"))
    if path.is_file():
        saved = read(path)
        if (saved["binding"] != binding or saved["initial_state"] != ce_state
                or saved["reference_gradient_ratio"] != reference
                or not pair.matches_first_batch(saved["raw_initial_h"], ce_first)):
            raise ValueError("H calibration belongs to a different CE pair")
        return saved
    student, auxiliary = pair.init_models(args, classes, device)
    if read(args.initial_state) != ce_state:
        raise ValueError("H initialization differs from its CE pair")
    first = measure_h(args, student, auxiliary, training, target, device)
    if first["records"] != ce_first["records"] or not pair.matches_first_batch(first, ce_first):
        raise ValueError("The pre-update H batch differs from its CE pair")
    weight = reference / first["weighted_gradient_ratio"]
    saved = {"binding": binding, "initial_state": ce_state, "raw_initial_h": first,
             "reference_gradient_ratio": reference, "lambda": weight}
    recipe.write_json(path, saved)
    print(f"Flowers H: fixed weight {weight:.10g}, reference gradient ratio {reference:.10g}", flush=True)
    return saved


def verify_full_pair(case, references, epochs):
    fields = ("epoch", "updates", "records", "lr", "ordered_index_and_augmentation_seed_sha256")

    def plan(directory):
        return [tuple(row[key] for key in fields) for row in read(directory / "history.json")]

    history = plan(case)
    result = read(case / "results.json")
    initial = result["initial_state_sha256"]
    if len(history) != epochs or any(history != plan(other) for other in references):
        raise ValueError("Flowers H, CE and Qwen have different training schedules")
    if any(read(other / "results.json")["initial_state_sha256"] != initial for other in references):
        raise ValueError("Flowers H, CE and Qwen have different initialization hashes")
    recipe.write_json(case / "h-verified.json", {"epochs": len(history), "steps": sum(row[1] for row in history),
                                               "paired_initial_state_sha256": initial,
                                               "results_sha256": recipe.sha256(case / "results.json")})


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The published training recipe requires a CUDA GPU with bfloat16 support")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    provenance = pair.PROVENANCE
    settings = provenance["training"]
    dataset = provenance["datasets"][DATASET]
    args.dataset, args.architecture, args.precision = DATASET, pair.ARCHITECTURE, "bfloat16"
    args.seed, args.epochs, args.effective_batch = settings["seed"], settings["epochs"], settings["effective_batch_size"]
    args.gradient_calibration = None
    sibling = args.out / DATASET / pair.ARCHITECTURE / f"seed-{args.seed}"
    case, ce_dir, qwen_dir = (sibling / name for name in (ARM, "ce", "qwen"))
    args.initial_state = case / "initial-state.json"

    data_hashes = {name: recipe.sha256(args.images / name) for name in recipe.RAW_SOURCES[DATASET]}
    if data_hashes != dataset["metadata_sha256"]:
        raise ValueError("Dataset metadata hashes differ from the official Flowers files")
    training = recipe.VisualDataset(args.images, DATASET, True)
    heldout = recipe.VisualDataset(args.images, DATASET, False)
    if np.intersect1d(training.ids, heldout.ids).size:
        raise ValueError("Training and held-out image IDs overlap")
    if args.validation_cache:
        heldout = pair.CachedVisualEval(args.validation_cache, heldout, dataset=DATASET)
    expected = {
        "dataset": DATASET, "architecture": pair.ARCHITECTURE, "seed": args.seed, "epochs": args.epochs,
        "precision": args.precision, "physical_batch_size": args.physical_batch,
        "effective_batch_size": args.effective_batch, "metadata_sha256": data_hashes,
        "pretrained_sha256": provenance["initialization"]["sha256"], "optimizer": settings["optimizer"],
        "torch": str(torch.__version__), "torchvision": torchvision.__version__,
        "validation_cache_sha256": heldout.metadata["sha256"] if args.validation_cache else None,
    }
    for directory in (ce_dir, qwen_dir):
        result_path = directory / "results.json"
        if not result_path.is_file() or read(result_path)["epochs"] != args.epochs:
            raise ValueError(f"Complete the paired Flowers {directory.name} run in {directory} first")
        config = read(directory / "config.json")
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Flowers H training and evaluation settings differ from its {directory.name} pair")

    case.mkdir(parents=True, exist_ok=True)
    target, digest = flowers_code(args.author_source, case / "class-code.json")
    target = torch.as_tensor(target, dtype=torch.float32, device=device)
    pair.ensure_pretrained_weights()
    binding = expected | {
        "author_revision": recipe.AUTHOR_REVISION,
        "training_source_sha256": recipe.sha256(Path(recipe.__file__)),
        "wrapper_sha256": recipe.sha256(Path(__file__)),
        "paired_reference_wrapper_sha256": read(ce_dir / "config.json")["wrapper_sha256"],
        "capture_input_parameter": "conv_proj.weight",
        "class_code_sha256_float64": digest,
        "auxiliary_shape": list(target.shape[1:]),
    }
    reference = dataset["five_way_code"]["resnet18_h_first_batch"]["weighted_gradient_ratio"]
    calibration = calibrate_h(args, dataset["classes"], training, target, device, binding, ce_dir, reference)
    args.auxiliary_weight = calibration["lambda"]
    binding |= {"h_calibration_sha256": recipe.sha256(case / "calibration.json"), "reference_gradient_ratio": reference}

    observe = recipe.first_observation
    checked = (case / "last.pt").is_file()

    def verify_first(*values):
        nonlocal checked
        first = observe(*values)
        if not checked:
            if not pair.matches_first_batch(first, calibration["raw_initial_h"], reference):
                raise ValueError("The first formal H update differs from its calibration")
            checked = True
        return first

    original_models = recipe.initial_models
    recipe.initial_models = pair.init_models
    recipe.first_observation = verify_first
    try:
        recipe.fit(args, ARM, dataset["classes"], training, heldout, device, target, binding, None)
    finally:
        recipe.initial_models = original_models
        recipe.first_observation = observe
    if (case / "results.json").is_file():
        verify_full_pair(case, (ce_dir, qwen_dir), args.epochs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Root of the unpacked Flowers-102 dataset")
    parser.add_argument("--out", type=Path, required=True, help="Output root of the completed Flowers CE and Qwen pair")
    parser.add_argument("--author-source", type=Path, required=True, help="Pinned LaViD checkout")
    parser.add_argument("--validation-cache", type=Path, help="Optional verified cache shared with the two reference arms")
    parser.add_argument("--physical-batch", type=int, choices=(16, 32, 64),
                        default=pair.PROVENANCE["training"]["physical_batch_size"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-epochs", type=int, help="Limit this invocation and resume with the same command")
    args = parser.parse_args()
    if args.workers < 0 or (args.max_new_epochs is not None and args.max_new_epochs < 1):
        parser.error("--workers must be nonnegative and --max-new-epochs must be positive when given")
    run(args)


if __name__ == "__main__":
    main()
