"""Train A–I and the gradient-matched G★ on the original visual cohorts."""

import argparse
import hashlib
from importlib import metadata, util
from itertools import chain, islice
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .teacher import write_json
from .train import sha256
from .train_visual_lavid import (DATASETS, MILESTONES, NUM_OUTPUTS, PURE_SQUARE, RAW_SOURCES, SHAPE, PairedOrder,
                                VisualDataset, class_head_term, evaluate, same_training_config, state_hash, write_checkpoint)
from .visual_reference import AUTHOR_REVISION


ARMS = {"ce": "A", "raw-onehot": "B", "prob-onehot": "C", "ce-same-onehot": "D", "squentropy": "E",
        "ce-independent-onehot": "F", "ecoc-soft-ce": "G", "ecoc": "H", "ecoc-constant": "I",
        "ecoc-soft-ce-gradmatch": "G★"}
WEIGHTED = ("ce-same-onehot", "ce-independent-onehot", "ecoc-soft-ce", "ecoc", "ecoc-constant")
GRAD_MATCH = "ecoc-soft-ce-gradmatch"
SOFT_ARMS = ("ecoc-soft-ce", GRAD_MATCH)
CODE_MSE = ("ecoc", "ecoc-constant")
ARCHITECTURES = ("mobilenetv2", "vit_small_patch16")
VIT_WEIGHTS_SHA256 = "79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa"


class LossStudent(nn.Module):
    def __init__(self, architecture, classes, author, weights=None):
        super().__init__()
        self.mobile = architecture == "mobilenetv2"
        if self.mobile:
            source = author / "lavid/models/mobilenetv2.py"
            spec = util.spec_from_file_location("lavid_author_mobilenetv2", source)
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.encoder = module.mobilenet_v2(pretrained=False, num_classes=classes)
        else:
            import timm

            self.encoder = timm.create_model("vit_small_patch16_224", pretrained=False)
            if weights:
                from safetensors.torch import load_file

                self.encoder.load_state_dict(load_file(str(weights)), strict=True)
            self.encoder.reset_classifier(classes)

    @property
    def classifier(self):
        return self.encoder.classifier[-1] if self.mobile else self.encoder.head

    @property
    def input_weight(self):
        return self.encoder.features[0][0].weight if self.mobile else self.encoder.patch_embed.proj.weight

    def training_outputs(self, x):
        if self.mobile:
            for layer in self.encoder.features:
                x = layer(x)
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            class_input = self.encoder.classifier[0](pooled)
        else:
            pooled = self.encoder.forward_head(self.encoder.forward_features(x), pre_logits=True)
            class_input = pooled
        return pooled, class_input, self.classifier(class_input)

    def forward(self, x):
        return self.encoder(x)


def frozen_code(path, classes, dataset, author):
    stored = json.loads(path.read_bytes())
    symbols = np.asarray(stored["ecoc_symbols_zero_based"], dtype=np.int64)
    names = json.loads((author / f"lavid/data/classes/{dataset}_classes.json").read_bytes())
    if (symbols.shape != (classes, SHAPE[0]) or (symbols < 0).any() or (symbols >= SHAPE[1]).any()
            or stored["class_names"] != names):
        raise ValueError("The frozen class code must match the official author class order")
    scale = stored["ecoc_scale_about_uniform"]
    if type(scale) not in (int, float) or not 0. < scale <= 1.:
        raise ValueError("The published class-code scale must give valid probabilities")
    return .2 + scale * (np.eye(SHAPE[1])[symbols] - .2)


def initial_models(args, classes, device):
    torch.manual_seed(args.seed)
    model = LossStudent(args.architecture, classes, args.author_source, args.weights).to(device)
    auxiliary = nn.Linear(model.classifier.in_features, NUM_OUTPUTS).to(device)
    return model, auxiliary


def training_loader(args, training, device):
    order = PairedOrder(training, args.seed)
    loader = DataLoader(training, sampler=order, batch_size=args.physical_batch, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(args.seed))
    return loader, order


def seed_epoch(seed, epoch):
    torch.manual_seed(1000003 * int(seed) + epoch)


def soft_code_ce(projected, target):
    log_options = projected.float().reshape(-1, *SHAPE).log_softmax(dim=-1)
    return -(target.reshape(-1, *SHAPE) * log_options).sum(dim=-1).mean()


def project_independent(auxiliary, features, classes):
    return F.linear(features, auxiliary.weight[:classes], auxiliary.bias[:classes])


def batch_losses(model, auxiliary, x, y, code, arm, amp):
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
        pooled, class_input, logits = model.training_outputs(x)
        projected = project_independent(auxiliary, class_input, logits.size(1)) if arm == "ce-independent-onehot" else None
        if arm in (*SOFT_ARMS, *CODE_MSE):
            projected = auxiliary(pooled)
    logits = logits.float()
    ce = F.cross_entropy(logits, y)
    if arm == "ce":
        other = ce.new_zeros(())
    elif arm in SOFT_ARMS:
        other = soft_code_ce(projected, code[y])
    elif arm in CODE_MSE:
        target = code.reshape(1, -1) if arm == "ecoc-constant" else code[y].flatten(1)
        other = (projected.float() - target).square().mean()
    else:
        other = class_head_term(arm, logits, y, projected.float() if projected is not None else None)
    return ce, other, (logits.argmax(dim=1) == y).sum()


def calibrate_gradmatch(args, training, classes, device, code, binding):
    path = args.gradient_calibration
    if path.exists():
        stored = json.loads(path.read_bytes())
        if stored["binding"] != binding:
            raise ValueError("The frozen G★ calibration belongs to another model/data recipe")
        return stored
    model, auxiliary = initial_models(args, classes, device)
    initial = state_hash(model, auxiliary)
    model.train()
    loader, order = training_loader(args, training, device)
    order.epoch = 1
    seed_epoch(args.seed, 1)
    amount = min(args.effective_batch, len(training))
    losses = torch.zeros(3, dtype=torch.float64, device=device)
    names = ("primary_ce", "G_soft_ce", "H_raw_mse")
    gradients = [torch.zeros_like(model.input_weight) for _ in names]
    fingerprint = hashlib.sha256()
    total = 0
    for images, labels, _ in islice(loader, (amount + args.physical_batch - 1) // args.physical_batch):
        fingerprint.update(images.numpy().tobytes())
        y = labels.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pooled, _, logits = model.training_outputs(images.to(device))
            projected = auxiliary(pooled)
        ce = F.cross_entropy(logits.float(), y)
        target = code[y]
        g = soft_code_ce(projected, target)
        h = (projected.float() - target.reshape(len(y), -1)).square().mean()
        terms = (ce, g, h)
        losses += torch.stack([t.detach() for t in terms]).double() * (len(y) / amount)
        for i, (destination, term) in enumerate(zip(gradients, terms, strict=True)):
            (value,) = torch.autograd.grad(term, model.input_weight, retain_graph=i < len(terms) - 1)
            destination.add_(value.detach(), alpha=len(y) / amount)
        total += len(y)
    raw_norms = [float(value.norm()) for value in gradients]
    primary, norm_g, norm_h = raw_norms
    raw_losses = losses.tolist()
    if total != amount or not all(np.isfinite(raw_norms + raw_losses)) or min(raw_norms) <= 0:
        raise ValueError("Cannot calibrate a non-finite or zero first-effective-batch gradient")
    beta_h = DATASETS[args.dataset][3]
    beta_g = beta_h * norm_h / norm_g
    result = {"binding": binding, "calibration_seed": 17, "records": total, "initial_state_sha256": initial,
              "augmented_batch_sha256": fingerprint.hexdigest(),
              "raw_loss": dict(zip(names, raw_losses, strict=True)),
              "raw_gradient_norm": dict(zip(names, raw_norms, strict=True)),
              "reference_H_weight": beta_h, "G_star_weight": beta_g,
              "weighted_G_star_to_H_gradient_ratio": beta_g * norm_g / (beta_h * norm_h),
              "weighted_H_to_primary_gradient_ratio": beta_h * norm_h / primary}
    write_json(path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def first_observation(losses, grad_ce, grad_aux, beta, fingerprint):
    ce, other = losses.tolist()
    norm_ce, norm_aux = float(grad_ce.norm()), float(grad_aux.norm())
    if not all(np.isfinite((ce, other, norm_ce, norm_aux))):
        return {"augmented_batch_sha256": fingerprint, "finite": False}
    return {"augmented_batch_sha256": fingerprint, "finite": True, "ce": ce, "raw_auxiliary": other,
            "weighted_auxiliary": beta * other, "weighted_to_ce_loss_ratio": beta * other / ce if ce else None,
            "input_weight_grad_ce": norm_ce, "input_weight_grad_auxiliary": norm_aux,
            "weighted_gradient_ratio": beta * norm_aux / norm_ce if norm_ce else None,
            "gradient_cosine": float(F.cosine_similarity(grad_ce.flatten(), grad_aux.flatten(), dim=0)) if norm_aux and norm_ce else None}


def train_epoch(args, model, auxiliary, loader, optimizer, device, code, arm, beta, epoch, first_path=None):
    model.train()
    auxiliary.train()
    seed_epoch(args.seed, epoch)
    records = len(loader.dataset)
    total = updates = 0
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    first_losses = torch.zeros(2, dtype=torch.float64, device=device)
    grad_ce = torch.zeros_like(model.input_weight)
    grad_aux = torch.zeros_like(model.input_weight)
    first_batch_hash = hashlib.sha256()
    first = {}
    observe = epoch in (1, args.epochs // 2, args.epochs)
    for images, labels, _ in loader:
        if total % args.effective_batch == 0:
            group = min(args.effective_batch, records - total)
            optimizer.zero_grad(set_to_none=True)
        n = len(labels)
        portion = n / group
        ce, other, correct = batch_losses(model, auxiliary, images.to(device, non_blocking=True),
                                          labels.to(device, non_blocking=True), code, arm, args.precision == "bfloat16")
        objective = other if arm in PURE_SQUARE else ce + beta * other
        if observe and total < args.effective_batch:
            first_batch_hash.update(images.numpy().tobytes())
            first_losses += torch.stack((ce.detach(), other.detach())).double() * portion
            (ce_gradient,) = torch.autograd.grad(ce, model.input_weight, retain_graph=True)
            grad_ce.add_(ce_gradient.detach(), alpha=portion)
            if arm != "ce":
                (aux_gradient,) = torch.autograd.grad(other, model.input_weight, retain_graph=True)
                grad_aux.add_(aux_gradient.detach(), alpha=portion)
        (objective if n == group else objective * portion).backward()
        total += n
        if total % args.effective_batch == 0 or total == records:
            if observe and updates == 0:
                first = first_observation(first_losses, grad_ce, grad_aux, beta, first_batch_hash.hexdigest())
                if epoch == 1 and first_path:
                    write_json(first_path, first | {"epoch": epoch, "records": total, "auxiliary_weight": beta})
                if not first["finite"]:
                    raise FloatingPointError(f"Non-finite first effective batch at epoch {epoch} for {arm}")
            optimizer.step()
            updates += 1
        sums += torch.stack((ce.detach() * n, other.detach() * n, objective.detach() * n, correct)).double()
    losses = (sums / total).tolist()
    if total != records or not all(np.isfinite(losses)):
        raise FloatingPointError(f"Epoch {epoch} did not produce a finite complete training result for {arm}")
    ce, other, objective, acc = losses
    return {"updates": updates, "records": total, "train_top1": acc, "ce": ce, "raw_auxiliary": other,
            "weighted_auxiliary": beta * other, "train_objective": objective, "first_effective_batch": first}


def fit(args, arm, n, training, heldout, device, codes, binding, gradmatch):
    out = args.out / args.dataset / args.architecture / f"seed-{args.seed}" / arm
    out.mkdir(parents=True, exist_ok=True)
    config_path, results, latest = (out / name for name in ("config.json", "results.json", "last.pt"))
    beta = 0. if arm == "ce" else 1.
    if arm in WEIGHTED:
        beta = DATASETS[args.dataset][3]
    if arm == GRAD_MATCH:
        beta = gradmatch["G_star_weight"]
    config = binding | {"arm": arm, "loss_letter": ARMS[arm], "seed": args.seed, "auxiliary_weight": beta,
                        "calibration_sha256": sha256(args.gradient_calibration) if arm == GRAD_MATCH else None,
                        "validation_cache_sha256": heldout.metadata["sha256"] if args.validation_cache else None}
    if config_path.exists() and not same_training_config(json.loads(config_path.read_bytes()), config):
        raise ValueError(f"Cannot resume with a different layout or training recipe at {out}")
    if results.exists():
        print(f"Reuse sealed result {out}", flush=True)
        return
    write_json(config_path, config)
    model, auxiliary = initial_models(args, n, device)
    initial = state_hash(model, auxiliary)
    if arm == GRAD_MATCH and args.seed == 17 and initial != gradmatch["initial_state_sha256"]:
        raise ValueError("The real training initialization differs from the frozen G★ calibration")
    method_initial = initial
    if arm == "ce-independent-onehot":
        with torch.no_grad():
            auxiliary.weight[:n].copy_(model.classifier.weight)
            auxiliary.bias[:n].copy_(model.classifier.bias)
        method_initial = state_hash(model, auxiliary)
    optimizer = torch.optim.SGD(chain(model.parameters(), auxiliary.parameters()), lr=.01, momentum=.9, weight_decay=.0005)
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.1) if model.mobile else
                 torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs))
    loader, order = training_loader(args, training, device)
    test = DataLoader(heldout, batch_size=args.physical_batch, num_workers=args.workers,
                      persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    history, best = [], {"epoch": None, "validation_nll": float("inf")}
    if latest.exists():
        recovered = torch.load(latest, weights_only=True, map_location=device)
        if recovered["initial_state_sha256"] != initial:
            raise ValueError("Initial shared state changed during resume")
        model.load_state_dict(recovered["student"])
        auxiliary.load_state_dict(recovered["auxiliary"])
        optimizer.load_state_dict(recovered["optimizer"])
        scheduler.load_state_dict(recovered["scheduler"])
        history, best = recovered["history"], recovered["author_best"]
        del recovered
    limit = min(args.epochs, len(history) + args.max_new_epochs) if args.max_new_epochs else args.epochs
    started = time.perf_counter()
    chosen_path = out / "author-best.pt"
    for epoch in range(len(history) + 1, limit + 1):
        order.epoch = epoch
        t0 = time.perf_counter()
        values = train_epoch(args, model, auxiliary, loader, optimizer, device, codes, arm, beta, epoch, out / "first_batch.json")
        values.update(epoch=epoch, lr=optimizer.param_groups[0]["lr"], training_seconds=time.perf_counter() - t0,
                      ordered_index_and_augmentation_seed_sha256=hashlib.sha256(order.plan().tobytes()).hexdigest())
        observation = values["first_effective_batch"]
        if epoch == 1 and args.seed == 17 and arm == GRAD_MATCH:
            if observation["augmented_batch_sha256"] != gradmatch["augmented_batch_sha256"]:
                raise ValueError("The actual first augmented batch differs from the frozen G★ calibration")
            observation["weighted_G_star_to_H_gradient_ratio_observed"] = beta * observation["input_weight_grad_auxiliary"] / (
                gradmatch["reference_H_weight"] * gradmatch["raw_gradient_norm"]["H_raw_mse"])
        scheduler.step()
        state = {"epoch": epoch, "student": model.state_dict(), "auxiliary": auxiliary.state_dict(),
                 "initial_state_sha256": initial, "method_initial_state_sha256": method_initial}
        if epoch == 1 or epoch % 5 == 0:
            before = time.perf_counter()
            validation = evaluate(model, test, device, args.precision == "bfloat16", n)
            if not all(np.isfinite(list(validation.values()))):
                raise FloatingPointError(f"Non-finite author evaluation at epoch {epoch}")
            values.update(author_eval_split=validation, validation_seconds=time.perf_counter() - before)
            if args.validation_cache:
                values["validation_cache_sha256"] = config["validation_cache_sha256"]
            if epoch % 5 == 0 and validation["nll"] < best["validation_nll"]:
                best = {"epoch": epoch, "validation_nll": validation["nll"]}
                write_checkpoint(chosen_path, state)
        history.append(values)
        write_checkpoint(latest, state | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                         "history": history, "author_best": best})
        write_json(out / "history.json", history)
        if epoch <= 3 or epoch % 5 == 0:
            print(f"{args.dataset}/{args.architecture}/{arm} seed={args.seed} epoch={epoch}/{args.epochs} "
                  f"steps={values['updates']} train_sec={values['training_seconds']:.2f} wall={time.perf_counter() - t0:.2f}", flush=True)
    if len(history) != args.epochs:
        return
    write_checkpoint(out / "final.pt", {"epoch": args.epochs, "student": model.state_dict(), "auxiliary": auxiliary.state_dict(),
                                        "initial_state_sha256": initial, "method_initial_state_sha256": method_initial})
    final = evaluate(model, test, device, False, n, out / "final_predictions.npz")
    selected = torch.load(chosen_path, weights_only=True, map_location=device)
    if selected["epoch"] != best["epoch"]:
        raise ValueError("The selected checkpoint did not match the author selection history")
    model.load_state_dict(selected["student"])
    author_best = evaluate(model, test, device, False, n, out / "author_best_predictions.npz")
    write_json(results, {"arm": arm, "initial_state_sha256": initial, "method_initial_state_sha256": method_initial,
                         "epochs": len(history), "optimizer_steps": sum(r["updates"] for r in history),
                         "author_selection": best, "fixed_final_fp32": final, "author_selected_fp32": author_best,
                         "artifact_sha256": {name: sha256(out / name) for name in ("final.pt", "author-best.pt",
                                                                                  "final_predictions.npz", "author_best_predictions.npz")},
                         "seconds_this_invocation": time.perf_counter() - started})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    for name in ("author-source", "images", "out", "code"):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in ("weights", "gradient-calibration", "validation-cache"):
        p.add_argument(f"--{name}", type=Path)
    p.add_argument("--arms", nargs="+", choices=ARMS)
    p.add_argument("--calibrate-gradmatch", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--physical-batch", type=int)
    p.add_argument("--max-new-epochs", type=int)
    args = p.parse_args()
    mobile = args.architecture == "mobilenetv2"
    args.epochs = 240 if mobile else 500
    args.effective_batch = 16 if mobile else 512
    args.physical_batch = args.physical_batch or (16 if mobile else 64)
    if (args.physical_batch < 1 or args.effective_batch % args.physical_batch or (mobile and args.physical_batch != 16)
            or args.workers < 0 or args.cpu_threads < 1 or not 0 <= args.seed < 2**31):
        p.error("Use MobileNet batch 16 or a positive ViT physical batch dividing 512, and valid worker/seed settings")
    if args.max_new_epochs is not None and args.max_new_epochs < 1:
        p.error("A partial training invocation requires at least one epoch")
    if mobile and args.weights or not mobile and (not args.weights or sha256(args.weights) != VIT_WEIGHTS_SHA256):
        p.error("Use the author's random MobileNetV2 or the already frozen ViT-S/16 weights")
    if args.calibrate_gradmatch and (args.arms or args.seed != 17 or not args.gradient_calibration):
        p.error("Run G★ gradient calibration separately at seed 17 and specify its output JSON")
    if not args.calibrate_gradmatch and (not args.arms or (GRAD_MATCH in args.arms) != bool(args.gradient_calibration)):
        p.error("Pass at least one arm and provide a first-batch gradient JSON only for G★")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    args.precision = "bfloat16" if device.type == "cuda" else "float32"
    n = DATASETS[args.dataset][0]
    target = frozen_code(args.code, n, args.dataset, args.author_source)
    codes = torch.as_tensor(target, dtype=torch.float32, device=device)
    train = VisualDataset(args.images, args.dataset, True)
    binding = {"dataset": args.dataset, "architecture": args.architecture, "author_revision": AUTHOR_REVISION,
               "class_code_sha256_float64": hashlib.sha256(target.tobytes()).hexdigest(),
               "precision": args.precision, "capture_input_parameter": "stem conv" if mobile else "patch projection",
               "pretrained_sha256": None if mobile else VIT_WEIGHTS_SHA256,
               "backbone_source": {"author_mobilenetv2_sha256": sha256(args.author_source / "lavid/models/mobilenetv2.py")} if mobile else
                                  {"timm": metadata.version("timm")},
               "optimizer": {"lr": .01, "momentum": .9, "weight_decay": .0005,
                             "scheduler": "milestones 150/180/210, gamma 0.1" if mobile else "epoch cosine, T_max=500"},
               "epochs": args.epochs, "physical_batch_size": args.physical_batch, "effective_batch_size": args.effective_batch,
               "input_preprocessing": "author RGB 224 and ImageNet channel normalization",
               "metadata_sha256": {name: sha256(args.images / name) for name in RAW_SOURCES[args.dataset]}}
    if args.calibrate_gradmatch:
        args.gradient_calibration.parent.mkdir(parents=True, exist_ok=True)
        calibrate_gradmatch(args, train, n, device, codes, binding)
        return
    gradmatch = json.loads(args.gradient_calibration.read_bytes()) if args.gradient_calibration else None
    if gradmatch and gradmatch["binding"] != binding:
        raise ValueError("G★ calibration does not match the frozen model, data, weights or effective batch")
    heldout = VisualDataset(args.images, args.dataset, False)
    if np.intersect1d(train.ids, heldout.ids).size:
        raise ValueError("Official training and held-out images overlap")
    if args.validation_cache:
        from .visual_validation_cache import CachedVisualEval

        heldout = CachedVisualEval(args.validation_cache, heldout, dataset=args.dataset)
        if "scores_identical" not in heldout.metadata:
            raise ValueError("Expected the original verified author evaluation cache")
    args.out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        if arm == "ecoc-constant":
            mean = target.mean(axis=0)
            constant = torch.as_tensor(mean, dtype=torch.float32, device=device)
            fixed = binding | {"auxiliary_target": "equal class mean", "target_float64_sha256": hashlib.sha256(mean.tobytes()).hexdigest()}
            fit(args, arm, n, train, heldout, device, constant, fixed, None)
        else:
            fit(args, arm, n, train, heldout, device, codes, binding, gradmatch)


if __name__ == "__main__":
    main()
