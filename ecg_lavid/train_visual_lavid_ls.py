"""Run the published Flowers ResNet-18 LaViD+LS path with a fixed visual teacher."""

import argparse
import hashlib
import json
import random
from itertools import chain
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .teacher import write_json
from .train import sha256
from .train_visual_lavid import BATCH_SIZE, MILESTONES, NUM_OUTPUTS, PairedOrder, evaluate, read_signatures, state_hash, write_checkpoint
from .train_visual_lavid_teacher import CLASSES, EPOCHS, FlowerTrainingImages, flower_evaluation, resnet_from_author
from .visual_teacher_logits import VisualTeacherLogits
from .visual_reference import AUTHOR_REVISION


SOURCE_HASHES = {
    "lavid/models/resnet.py": "f32099bb2a6632404f1ee677bbc32a399293b35f997bc7245afbed4547d13363",
    "lavid/models/distill/mlkd.py": "a05b11b82637fe2e22ce4d24cec2f9f3c573b1335518887994acb7736705e70f",
    "lavid/models/distill/_base.py": "a109b34802f0e3c8f2ec139ba0f809a5ba45e294e8dd2c2f682e4a37dbf30116",
    "lavid/data/base_dataset.py": "9333d0a657a964af86ebf92957a81a9289a44f5ac8d389f6b419f56cf76477c1",
    "lavid/data/mlkd_augment.py": "6106cd1a4e0d3e14c6851225257e819b54ac70d84351cf71f9934887387253df",
}
TERMS = ("loss_ce", "loss_kd", "loss_cc", "loss_bc")
PAIRED_EPOCH_FIELDS = ("epoch", "updates", "records", "lr", "first_augmented_batch_sha256",
                       "ordered_index_and_augmentation_seed_sha256")
BETA = 50.
QWEN_SHA256 = "fdb051c89eabc1c51ab80947251fb7176f64893bbf0641e229028a9490b6d74d"


def author_components(root):
    if any(sha256(root / name) != digest for name, digest in SOURCE_HASHES.items()):
        raise ValueError("A pinned author model, objective or augmentation source has changed")
    # Only these upstream submodules are needed. Their package initializers also
    # import Lightning and CLIP for unrelated training routes.
    for name in ("lavid", "lavid.data", "lavid.models"):
        location = root.joinpath(*name.split("."))
        if name in sys.modules:
            if list(sys.modules[name].__path__) != [str(location)]:
                raise ValueError("Another author package is already imported")
        else:
            package = ModuleType(name)
            package.__path__ = [str(location)]
            sys.modules[name] = package

    from lavid.data.base_dataset import BaseDataset
    from lavid.models.distill.mlkd import MLKD

    return BaseDataset._setup_mlkd, MLKD


class HybridFlowers(FlowerTrainingImages):
    def __init__(self, root, author_view_setup, rgb_cache=None):
        super().__init__(root, rgb_cache)
        author_view_setup(self)

    def seed_transform(self, seed):
        super().seed_transform(seed)
        np.random.seed(int(seed))
        random.seed(int(seed))


def load_target(path, kind, names):
    if kind == "qwen":
        if sha256(path) != QWEN_SHA256:
            raise ValueError("Expected the original released Flowers Qwen2.5-7B signature")
        target = read_signatures(path, names)
    else:
        stored = json.loads(path.read_bytes())
        if stored["class_names"] != names or stored["ecoc_seed"] != 271828:
            raise ValueError("Expected the archived Flowers code and its original class order")
        symbols = np.asarray(stored["ecoc_symbols_zero_based"])
        scale = stored["ecoc_scale_about_uniform"]
        if symbols.shape != (CLASSES, 50) or not np.issubdtype(symbols.dtype, np.integer) or not 0 < scale <= 1:
            raise ValueError("Unexpected original five-way code shape or scale")
        if (symbols < 0).any() or (symbols >= 5).any():
            raise ValueError("Original five-way symbols outside the valid range")
        target = .2 + scale * (np.eye(5, dtype=np.float64)[symbols] - .2)
        if hashlib.sha256(target.tobytes()).hexdigest() != stored["target_float64_sha256"]:
            raise ValueError("Archived hard symbols and scale do not reconstruct the locked prior soft target")
    return target


def hybrid_epoch(model, projection, distiller, loader, optimizer, target, device, amp, epoch, teacher_logits=None,
                 beta=BETA):
    distiller.train()
    projection.train()
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    seen = 0
    for step, ((weak, strong), labels, rows) in enumerate(loader, 1):
        if step == 1:
            first_batch = {name: hashlib.sha256(view.numpy().tobytes()).hexdigest()
                           for name, view in (("weak", weak), ("strong", strong))}
        if teacher_logits:
            teacher_logits.open_batch(seen, rows.numpy(), first_batch if step == 1 else None)
        weak = weak.to(device, non_blocking=True)
        strong = strong.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            features, logits = model(weak, is_feat=True)
            _, terms = distiller.forward_train(weak, strong, labels, epoch=epoch - 1)
            mse = F.mse_loss(projection(features[-1]), target[labels])
            visual = sum(terms.values())
            loss = visual + beta * mse
        if teacher_logits:
            teacher_logits.close_batch()
        loss.backward()
        optimizer.step()
        record = [terms[name] for name in TERMS] + [mse, visual, loss]
        totals += torch.stack([x.detach() * len(labels) for x in record] + [(logits.argmax(1) == labels).sum()]).double()
        seen += len(labels)
    means = (totals / seen).tolist()
    if seen != len(loader.dataset) or not all(np.isfinite(means)):
        raise FloatingPointError(f"Non-finite or incomplete hybrid training epoch {epoch}")
    ce, kd, cc, bc, mse, visual, objective, accuracy = means
    result = {"records": seen, "updates": step, "ce": ce, "mlkd_kl": kd, "mlkd_cc": cc, "mlkd_bc": bc,
              "mlkd_total": visual, "raw_auxiliary": mse, "weighted_auxiliary": beta * mse,
              "train_objective": objective, "train_top1": accuracy, "first_augmented_batch_sha256": first_batch}
    if teacher_logits:
        result["teacher_logit_cache"] = teacher_logits.end_epoch()
    return result


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    amp = device.type == "cuda"
    author_views, mlkd_class = author_components(args.author_source)
    names = json.loads((args.author_source / "lavid/data/classes/flowers_classes.json").read_bytes())
    raw_target = load_target(args.target_file, args.target_kind, names)
    target = torch.as_tensor(raw_target.reshape(CLASSES, NUM_OUTPUTS), dtype=torch.float32, device=device)
    training = HybridFlowers(args.images, author_views, args.training_rgb_cache)
    heldout = flower_evaluation(args.images, args.validation_cache)
    order = PairedOrder(training, args.seed)
    loader_options = {"batch_size": BATCH_SIZE, "num_workers": args.workers,
                      "persistent_workers": args.workers > 0, "pin_memory": amp}
    batches = DataLoader(training, sampler=order, **loader_options)
    test = DataLoader(heldout, **loader_options)

    source = args.author_source / "lavid/models/resnet.py"
    teacher_result = json.loads((args.teacher_checkpoint.parent / "results.json").read_bytes())
    teacher_sha = sha256(args.teacher_checkpoint)
    if teacher_result["epochs"] != EPOCHS or teacher_sha != teacher_result["artifact_sha256"]["author-best.pt"]:
        raise ValueError("The teacher does not match the completed selected ResNet-50 checkpoint")
    teacher_config = json.loads((args.teacher_checkpoint.parent / "config.json").read_bytes())
    if (teacher_config["training_image_ids"] != training.ids.tolist()
            or teacher_config["evaluation_image_ids"] != heldout.ids.tolist()):
        raise ValueError("Teacher and student use different official image ID populations")
    model = resnet_from_author(source, 18).to(device)
    teacher = resnet_from_author(source, 50)
    teacher_state = torch.load(args.teacher_checkpoint, weights_only=True, map_location="cpu")
    teacher.load_state_dict(teacher_state["student"], strict=True)
    teacher = teacher.to(device).requires_grad_(False).eval()
    teacher_logits = VisualTeacherLogits(teacher) if args.teacher_logit_cache else None
    distiller = mlkd_class(model, teacher_logits if teacher_logits is not None else teacher,
                          SimpleNamespace(T=1, CE_WEIGHT=.5, ALPHA=.5, LOGIT_STAND=True))
    projection = nn.Linear(512, NUM_OUTPUTS).to(device)
    initial = state_hash(model, projection)
    optimizer = torch.optim.SGD(chain(model.parameters(), projection.parameters()), lr=.01, momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.1)
    output = args.out / "flowers" / "resnet18" / f"seed-{args.seed}" / f"lavid-ls-{args.target_kind}"
    output.mkdir(parents=True, exist_ok=True)
    config = {"author_revision": AUTHOR_REVISION, "source_sha256": SOURCE_HASHES, "dataset": "flowers",
              "architecture": "resnet18", "seed": args.seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
              "precision": "bfloat16-mixed" if amp else "float32", "gamma": 1., "auxiliary_weight": BETA,
              "target_kind": args.target_kind, "target_source_sha256": sha256(args.target_file),
              "target_float64_sha256": hashlib.sha256(raw_target.tobytes()).hexdigest(),
              "teacher_sha256": teacher_sha, "teacher_selected_epoch": teacher_state["epoch"],
              "initial_state_sha256": initial, "metadata_sha256": teacher_config["metadata_sha256"],
              "optimizer": {"lr": .01, "momentum": .9, "weight_decay": .0005,
                            "milestones": list(MILESTONES), "gamma": .1},
              "mlkd": {"default_temperature": 1, "logit_standardization": True, "ce_coefficient": .5, "alpha": .5}}
    del teacher_state
    cfg_path, last, best_path = (output / name for name in ("config.json", "last.pt", "author-best.pt"))
    if cfg_path.exists() and json.loads(cfg_path.read_bytes()) != config:
        raise ValueError(f"The hybrid recipe changed at {output}")
    if (output / "results.json").is_file():
        print(f"Reuse the sealed hybrid student at {output}", flush=True)
        return
    reference_history = None
    if args.target_kind == "ecoc":
        reference = output.parent / "lavid-ls-qwen"
        complete = json.loads((reference / "results.json").read_bytes())
        previous = json.loads((reference / "config.json").read_bytes())
        target_keys = {"target_kind", "target_source_sha256", "target_float64_sha256"}
        reference_history = json.loads((reference / "history.json").read_bytes())
        if (complete["epochs"] != EPOCHS or len(reference_history) != EPOCHS or
                {k: v for k, v in previous.items() if k not in target_keys} !=
                {k: v for k, v in config.items() if k not in target_keys}):
            raise ValueError("Five-way-code training requires a completed Qwen pair with identical configuration")
    write_json(cfg_path, config)
    write_json(output / "input-cache.json", {
        "train_rgb": str(args.training_rgb_cache) if args.training_rgb_cache else None,
        "validation_crop_sha256": heldout.metadata["sha256"] if args.validation_cache else None,
        "visual_teacher_logit_replay_root": str(args.teacher_logit_cache) if args.teacher_logit_cache else None,
    })
    history, best = [], {"epoch": None, "validation_nll": float("inf")}
    if last.exists():
        state = torch.load(last, weights_only=True, map_location=device)
        if state["initial_state_sha256"] != initial:
            raise ValueError("Hybrid seed produced a different initial state")
        model.load_state_dict(state["student"], strict=True)
        projection.load_state_dict(state["auxiliary"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best = state["history"], state["author_best"]
        del state

    for epoch in range(len(history) + 1, EPOCHS + 1):
        order.epoch = epoch
        started = time.perf_counter()
        plan_sha = hashlib.sha256(order.plan().tobytes()).hexdigest()
        if teacher_logits:
            cache_path = args.teacher_logit_cache / f"seed-{args.seed}" / f"epoch-{epoch:03d}.npz"
            binding = {"seed": args.seed, "epoch": epoch, "teacher_sha256": teacher_sha, "batch": BATCH_SIZE,
                       "data_metadata_sha256": config["metadata_sha256"], "paired_order_sha256": plan_sha,
                       "author_augmentation_sha256": SOURCE_HASHES["lavid/data/mlkd_augment.py"],
                       "author_view_setup_sha256": SOURCE_HASHES["lavid/data/base_dataset.py"],
                       "precision": config["precision"], "device": torch.cuda.get_device_name(device),
                       "torch": torch.__version__}
            teacher_logits.begin_epoch(cache_path, "record" if args.target_kind == "qwen" else "replay",
                                      binding, len(training), CLASSES, device)
        row = hybrid_epoch(model, projection, distiller, batches, optimizer, target, device, amp, epoch, teacher_logits)
        row.update(epoch=epoch, lr=optimizer.param_groups[0]["lr"], training_seconds=time.perf_counter() - started,
                   ordered_index_and_augmentation_seed_sha256=plan_sha)
        if reference_history is not None:
            qwen = reference_history[epoch - 1]
            if any(row[key] != qwen[key] for key in PAIRED_EPOCH_FIELDS):
                raise ValueError(f"The paired Qwen and code batches or training schedule differed at epoch {epoch}")
        scheduler.step()
        state = {"epoch": epoch, "student": model.state_dict(), "auxiliary": projection.state_dict(),
                 "initial_state_sha256": initial}
        if epoch == 1 or epoch % 5 == 0:
            started_eval = time.perf_counter()
            scores = evaluate(model, test, device, amp, CLASSES)
            if not all(np.isfinite(list(scores.values()))):
                raise FloatingPointError(f"Non-finite hybrid evaluation at epoch {epoch}")
            row.update(author_eval_split=scores, validation_seconds=time.perf_counter() - started_eval)
            if epoch % 5 == 0 and scores["nll"] < best["validation_nll"]:
                best = {"epoch": epoch, "validation_nll": scores["nll"]}
                write_checkpoint(best_path, state)
        history.append(row)
        write_checkpoint(last, state | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                        "history": history, "author_best": best})
        write_json(output / "history.json", history)
        if epoch <= 3 or epoch % 5 == 0:
            print(f"Flowers hybrid {args.target_kind} seed={args.seed} epoch={epoch}/{EPOCHS} steps={row['updates']} "
                  f"CE={row['ce']:.5f} MLKD={row['mlkd_total']:.5f} QWEN_OR_CODE_MSE={row['raw_auxiliary']:.5f} "
                  f"wall={time.perf_counter() - started:.2f}", flush=True)

    end = {"epoch": EPOCHS, "student": model.state_dict(), "auxiliary": projection.state_dict(),
           "initial_state_sha256": initial}
    write_checkpoint(output / "final.pt", end)
    final = evaluate(model, test, device, False, CLASSES, output / "final_predictions.npz")
    selected = torch.load(best_path, weights_only=True, map_location=device)
    if selected["epoch"] != best["epoch"]:
        raise ValueError("Selected hybrid checkpoint did not match the logged validation result")
    model.load_state_dict(selected["student"], strict=True)
    selected_metrics = evaluate(model, test, device, False, CLASSES, output / "author_best_predictions.npz")
    files = ("final.pt", "author-best.pt", "final_predictions.npz", "author_best_predictions.npz")
    write_json(output / "results.json", {"target_kind": args.target_kind, "epochs": EPOCHS, "teacher_sha256": teacher_sha,
                                         "initial_state_sha256": initial, "author_selection": best,
                                         "optimizer_steps": sum(row["updates"] for row in history),
                                         "fixed_final_fp32": final, "author_selected_fp32": selected_metrics,
                                         "artifact_sha256": {name: sha256(output / name) for name in files}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "author-source", "out", "teacher-checkpoint", "target-file"):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in ("training-rgb-cache", "validation-cache", "teacher-logit-cache"):
        p.add_argument(f"--{name}", type=Path)
    p.add_argument("--target-kind", choices=("qwen", "ecoc"), required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.workers < 0:
        p.error("Use a nonnegative worker count")
    if args.teacher_logit_cache and torch.device(args.device).type != "cuda":
        p.error("The native BF16 teacher logit cache is available for CUDA runs")
    run(args)


if __name__ == "__main__":
    main()
