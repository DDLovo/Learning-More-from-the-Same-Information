"""Train the paired ViT-B/16 cross-entropy and Qwen auxiliary-loss arms."""

import argparse
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models import ViT_B_16_Weights, vit_b_16

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ecg_lavid import train_visual_loss_matrix as recipe
from ecg_lavid.train_visual_lavid import read_signatures
from ecg_lavid.visual_validation_cache import CachedVisualEval


PROVENANCE = json.loads(Path(__file__).with_name("provenance.json").read_bytes())
ARCHITECTURE = "vit_b_16"
WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1


class TorchVisionStudent(nn.Module):
    mobile = False

    def __init__(self, classes):
        super().__init__()
        self.encoder = vit_b_16(weights=WEIGHTS)
        self.encoder.heads = nn.Linear(self.encoder.hidden_dim, classes)

    @property
    def classifier(self):
        return self.encoder.heads

    @property
    def input_weight(self):
        return self.encoder.conv_proj.weight

    def training_outputs(self, images):
        tokens = self.encoder._process_input(images)
        prefix = self.encoder.class_token.expand(len(tokens), -1, -1)
        features = self.encoder.encoder(torch.cat((prefix, tokens), dim=1))[:, 0]
        return features, features, self.classifier(features)

    def forward(self, images):
        return self.encoder(images)


def ensure_pretrained_weights():
    reference = PROVENANCE["initialization"]
    if WEIGHTS.url != reference["url"]:
        raise ValueError("TorchVision's ViT-B/16 initialization URL differs from the published recipe")
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS.url.rsplit("/", 1)[1]
    if not checkpoint.is_file():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS.url, str(checkpoint), hash_prefix=reference["sha256"])
    if recipe.sha256(checkpoint) != reference["sha256"]:
        raise ValueError(f"SHA-256 mismatch for the ImageNet-1K initialization at {checkpoint}")


def read_qwen_targets(author_source, dataset):
    qwen = PROVENANCE["datasets"][dataset]["qwen_target"]
    signature, class_order = (author_source / qwen[key] for key in ("source", "class_order"))
    for path, key in ((signature, "source_sha256"), (class_order, "class_order_sha256")):
        if recipe.sha256(path) != qwen[key]:
            raise ValueError(f"SHA-256 mismatch at {path}. Use the pinned LaViD checkout.")
    probability = read_signatures(signature, json.loads(class_order.read_bytes()))
    digest = hashlib.sha256(probability.astype("<f8", copy=False).tobytes()).hexdigest()
    if list(probability.shape) != qwen["shape"] or digest != qwen["matrix_float64_le_sha256"]:
        raise ValueError("The Qwen probability matrix differs from the published class order")
    return probability


def init_models(args, classes, device):
    torch.manual_seed(args.seed)
    student = TorchVisionStudent(classes).to(device)
    auxiliary = nn.Linear(student.classifier.in_features, recipe.NUM_OUTPUTS).to(device)
    state = {"seed": args.seed, "student_and_auxiliary_sha256": recipe.state_hash(student, auxiliary)}
    if args.initial_state.is_file() and json.loads(args.initial_state.read_bytes()) != state:
        raise ValueError("Initialization changed on restart")
    recipe.write_json(args.initial_state, state)
    return student, auxiliary


def matches_first_batch(observed, reference, gradient_ratio=None):
    if not observed.get("finite") or not reference.get("finite"):
        return False
    keys = ("ce", "input_weight_grad_ce")
    if gradient_ratio is not None:
        keys += ("raw_auxiliary", "input_weight_grad_auxiliary")
    return (
        observed["augmented_batch_sha256"] == reference["augmented_batch_sha256"]
        and all(math.isclose(observed[key], reference[key], rel_tol=1e-5, abs_tol=1e-8) for key in keys)
        and (gradient_ratio is None
             or math.isclose(observed["weighted_gradient_ratio"], gradient_ratio, rel_tol=1e-7, abs_tol=1e-8))
    )


def measure_qwen(args, student, auxiliary, training, target, device):
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
            target, "qwen", True,
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
        raise FloatingPointError("Expected finite, nonzero gradients from the complete first effective batch")
    return first | {"records": count}


def calibrate_qwen(args, classes, training, target, device, common_binding, qwen_binding):
    path = args.initial_state.with_name("calibration.json")
    ce_dir = args.initial_state.parent.parent / "ce"
    required = [ce_dir / name for name in ("config.json", "initial-state.json", "first_batch.json")]
    if not all(candidate.is_file() for candidate in required):
        raise ValueError(f"Run the paired CE arm in the same output root to produce its first-batch evidence at {ce_dir}")
    ce_config, ce_state, ce_first = (json.loads(candidate.read_bytes()) for candidate in required)
    if any(ce_config.get(key) != value for key, value in common_binding.items()):
        raise ValueError("The CE first batch belongs to a different training configuration")
    reference = PROVENANCE["datasets"][args.dataset]["qwen_target"]["reference_gradient_ratio"]
    binding = common_binding | qwen_binding
    if path.is_file():
        saved = json.loads(path.read_bytes())
        if (saved["binding"] != binding or saved["initial_state"] != ce_state
                or saved["reference_gradient_ratio"] != reference
                or not matches_first_batch(saved["raw_initial_qwen"], ce_first)):
            raise ValueError("Qwen calibration belongs to a different CE pair")
        return saved

    model, head = init_models(args, classes, device)
    if json.loads(args.initial_state.read_bytes()) != ce_state:
        raise ValueError("The initialized Qwen model differs from its CE pair")
    first = measure_qwen(args, model, head, training, target, device)
    if first["records"] != ce_first["records"] or not matches_first_batch(first, ce_first):
        raise ValueError("The pre-update Qwen batch differs from its CE pair")
    weight = reference / first["weighted_gradient_ratio"]
    saved = {
        "binding": binding, "initial_state": ce_state, "raw_initial_qwen": first,
        "reference_gradient_ratio": reference, "lambda": weight,
    }
    recipe.write_json(path, saved)
    print(f"{args.dataset}: fixed Qwen weight {weight:.10g}, reference gradient ratio {reference:.10g}", flush=True)
    return saved


def verify_formal_first_batch(calibration):
    observe = recipe.first_observation
    checked = False

    def verify_once(*values):
        nonlocal checked
        first = observe(*values)
        if not checked:
            if not matches_first_batch(first, calibration["raw_initial_qwen"], calibration["reference_gradient_ratio"]):
                raise ValueError("The first formal Qwen update differs from its calibration")
            checked = True
        return first

    recipe.first_observation = verify_once


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The published training recipe requires a CUDA GPU with bfloat16 support")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    dataset = PROVENANCE["datasets"][args.dataset]
    data_hashes = {name: recipe.sha256(args.images / name) for name in recipe.RAW_SOURCES[args.dataset]}
    if data_hashes != dataset["metadata_sha256"]:
        raise ValueError(f"Dataset metadata hashes differ from the official {args.dataset} files recorded in provenance.json")
    training = recipe.VisualDataset(args.images, args.dataset, True)
    heldout = recipe.VisualDataset(args.images, args.dataset, False)
    if np.intersect1d(training.ids, heldout.ids).size:
        raise ValueError("Training and held-out image IDs overlap")
    if args.validation_cache:
        heldout = CachedVisualEval(args.validation_cache, heldout, dataset=args.dataset)

    target = None
    qwen_binding = {}
    if args.arm == "qwen":
        probability = read_qwen_targets(args.author_source, args.dataset)
        target = torch.as_tensor(probability, dtype=torch.float32, device=device)
        qwen = dataset["qwen_target"]
        qwen_binding = {
            "qwen_source_sha256": qwen["source_sha256"],
            "qwen_float64_matrix_sha256": qwen["matrix_float64_le_sha256"],
            "class_order_sha256": qwen["class_order_sha256"],
            "auxiliary_shape": qwen["shape"][1:],
        }

    ensure_pretrained_weights()
    settings = PROVENANCE["training"]
    args.seed = settings["seed"]
    args.epochs = settings["epochs"]
    args.effective_batch = settings["effective_batch_size"]
    args.precision = PROVENANCE["reference_environment"]["precision"]
    args.architecture = ARCHITECTURE
    args.gradient_calibration = None
    args.initial_state = args.out / args.dataset / ARCHITECTURE / f"seed-{args.seed}" / args.arm / "initial-state.json"
    args.initial_state.parent.mkdir(parents=True, exist_ok=True)
    binding = {
        "dataset": args.dataset,
        "architecture": ARCHITECTURE,
        "author_revision": recipe.AUTHOR_REVISION,
        "pretrained_sha256": PROVENANCE["initialization"]["sha256"],
        "training_source_sha256": recipe.sha256(Path(recipe.__file__)),
        "wrapper_sha256": recipe.sha256(Path(__file__)),
        "metadata_sha256": data_hashes,
        "precision": args.precision,
        "capture_input_parameter": "conv_proj.weight",
        "physical_batch_size": args.physical_batch,
        "effective_batch_size": args.effective_batch,
        "epochs": args.epochs,
        "optimizer": settings["optimizer"],
        "torchvision": torchvision.__version__,
        "torch": str(torch.__version__),
    }

    recipe.initial_models = init_models
    args.auxiliary_weight = 0.0
    if args.arm == "qwen":
        recipe.ARMS = recipe.ARMS | {"qwen": "Qwen"}
        recipe.CODE_MSE = (*recipe.CODE_MSE, "qwen")
        recipe.WEIGHTED = (*recipe.WEIGHTED, "qwen")
        calibration = calibrate_qwen(args, dataset["classes"], training, target, device, binding, qwen_binding)
        args.auxiliary_weight = calibration["lambda"]
        recipe.DATASETS = recipe.DATASETS | {args.dataset: (*recipe.DATASETS[args.dataset][:3], args.auxiliary_weight)}
        binding |= qwen_binding | {
            "qwen_calibration_sha256": recipe.sha256(args.initial_state.with_name("calibration.json")),
            "reference_gradient_ratio": calibration["reference_gradient_ratio"],
        }
        if not args.initial_state.with_name("last.pt").is_file():
            verify_formal_first_batch(calibration)

    recipe.fit(args, args.arm, dataset["classes"], training, heldout, device, target, binding, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=PROVENANCE["datasets"], required=True)
    parser.add_argument("--arm", choices=("ce", "qwen"), required=True)
    parser.add_argument("--images", type=Path, required=True, help="Root of the unpacked official image dataset")
    parser.add_argument("--out", type=Path, required=True, help="Common output root for both paired arms")
    parser.add_argument("--author-source", type=Path, help="Pinned LaViD checkout, required for Qwen")
    parser.add_argument("--validation-cache", type=Path, help="Optional verified .npy cache of held-out images")
    parser.add_argument("--physical-batch", type=int, choices=(16, 32, 64),
                        default=PROVENANCE["training"]["physical_batch_size"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-epochs", type=int, help="Limit this invocation and resume with the same command")
    args = parser.parse_args()
    if args.arm == "qwen" and args.author_source is None:
        parser.error("--author-source is required for --arm qwen")
    if args.workers < 0 or (args.max_new_epochs is not None and args.max_new_epochs < 1):
        parser.error("--workers must be nonnegative and --max-new-epochs must be positive when given")
    run(args)


if __name__ == "__main__":
    main()
