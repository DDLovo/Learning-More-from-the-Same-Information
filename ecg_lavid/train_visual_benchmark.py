"""Train a frozen Table 1 student with the author hybrid or the historical single-view code loss."""

import argparse
import hashlib
from importlib import util
from itertools import chain
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .teacher import write_json
from .train import sha256
from .train_visual_lavid import BATCH_SIZE, MILESTONES, NUM_OUTPUTS, PairedOrder, evaluate, state_hash, write_checkpoint
from .train_visual_lavid_ls import PAIRED_EPOCH_FIELDS, SOURCE_HASHES, author_components, hybrid_epoch
from .train_visual_lavid_teacher import EPOCHS, resnet_from_author
from .visual_benchmark_data import ARCHITECTURES, LOSS_WEIGHTS, RELEASES, language_target, load_names, populations, soft_code
from .visual_reference import AUTHOR_REVISION
from .visual_teacher_logits import VisualTeacherLogits


METHODS = ("lavid-ls-qwen", "lavid-ls-ecoc", "ecoc-only")
MODEL_SOURCES = {
    "resnet18": ("resnet.py", "resnet18", 512, SOURCE_HASHES["lavid/models/resnet.py"]),
    "mobilenetv2": ("mobilenetv2.py", "mobilenet_v2", 1280, "c341a42c5f7b80b0f992d8b5a0454b1ef231fd6c2584e6a6887028edc4e32b37"),
    "shufflenetv2": ("shufflenetv2.py", "shufflenet_v2_x1_0", 1024, "329bd9f5c4b050a11a53199620b99e4bcb664c6f97485ec6e8933833adda59fc"),
}


class HybridViews(Dataset):
    def __init__(self, source, setup_views):
        self.source = source
        self.ids = source.ids
        self.labels = source.labels
        setup_views(source)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        _, seed = item
        np.random.seed(int(seed))
        random.seed(int(seed))
        return self.source[item]


def student_from_author(author, architecture, outputs):
    name, constructor, dimension, expected = MODEL_SOURCES[architecture]
    source = author / "lavid/models" / name
    if sha256(source) != expected:
        raise ValueError(f"The fixed author model has changed: {source}")
    spec = util.spec_from_file_location(f"lavid_author_{architecture}", source)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, constructor)(pretrained=False, num_classes=outputs), dimension


def single_view_epoch(model, projection, optimizer, data, target, beta, device, amp):
    model.train()
    projection.train()
    aggregate = torch.zeros(4, dtype=torch.float64, device=device)
    records = 0
    for step, (images, labels, _) in enumerate(data, 1):
        if step == 1:
            first = hashlib.sha256(images.numpy().tobytes()).hexdigest()
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            features, logits = model(images, is_feat=True)
            projected = projection(features[-1])
        ce = F.cross_entropy(logits.float(), labels)
        mse = (projected.float() - target[labels]).square().mean()
        objective = ce + beta * mse
        objective.backward()
        optimizer.step()
        aggregate += torch.stack((ce.detach() * len(labels), mse.detach() * len(labels),
                                  objective.detach() * len(labels), (logits.argmax(1) == labels).sum())).double()
        records += len(labels)
    ce, mse, objective, top1 = (aggregate / records).tolist()
    if records != len(data.dataset) or not np.isfinite((ce, mse, objective, top1)).all():
        raise FloatingPointError("Non-finite or incomplete single-view label-code epoch")
    return {"records": records, "updates": step, "ce": ce, "raw_auxiliary": mse, "weighted_auxiliary": beta * mse,
            "train_objective": objective, "train_top1": top1, "first_augmented_batch_sha256": first}


def frozen_teacher(checkpoint, args, training, test, meta, classes, device):
    folder = checkpoint.parent
    result = json.loads((folder / "results.json").read_bytes())
    config = json.loads((folder / "config.json").read_bytes())
    digest = sha256(checkpoint)
    if (result["epochs"] != EPOCHS or digest != result["artifact_sha256"]["author-best.pt"]
            or config["architecture"] != "resnet50" or config["metadata_sha256"] != meta
            or config["training_image_ids"] != training.ids.tolist() or config["evaluation_image_ids"] != test.ids.tolist()):
        raise ValueError("The selected visual teacher must be sealed on the exact same author cohort")
    model = resnet_from_author(args.author_source / "lavid/models/resnet.py", 50, classes)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(state["student"], strict=True)
    return model.to(device).requires_grad_(False).eval(), digest, state["epoch"]


def paired_qwen_history(directory, config, method):
    if method != "lavid-ls-ecoc":
        return None
    reference = directory.parent / "lavid-ls-qwen"
    result = json.loads((reference / "results.json").read_bytes())
    precedent = json.loads((reference / "config.json").read_bytes())
    history = json.loads((reference / "history.json").read_bytes())
    target_fields = {"target_kind", "target_source_sha256", "target_float64_sha256"}
    if (result["epochs"] != EPOCHS or len(history) != EPOCHS or
            {key: v for key, v in precedent.items() if key not in target_fields} !=
            {key: v for key, v in config.items() if key not in target_fields}):
        raise ValueError("The code hybrid requires its sealed, same-config Qwen arm")
    return history


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    amp = device.type == "cuda"
    hybrid = args.method != "ecoc-only"
    qwen = args.method == "lavid-ls-qwen"
    true_classes, outputs, _, _ = RELEASES[args.dataset]
    beta = LOSS_WEIGHTS[args.dataset][ARCHITECTURES.index(args.architecture)]
    training, test, metadata = populations(args.dataset, args.images, args.cohort, args.training_rgb_cache,
                                           args.validation_cache)
    names, qwen_names = load_names(args.dataset, args.author_source, args.cohort)
    rows = language_target(args.dataset, args.target, qwen_names) if qwen else soft_code(args.target, args.dataset, names)
    target = torch.as_tensor(rows.reshape(true_classes, NUM_OUTPUTS), device=device, dtype=torch.float32)
    student, dimension = student_from_author(args.author_source, args.architecture, outputs)
    student = student.to(device)
    teacher_sha = teacher_epoch = None
    teacher_logits = distiller = None
    if hybrid:
        setup, distiller_class = author_components(args.author_source)
        teacher, teacher_sha, teacher_epoch = frozen_teacher(args.teacher_checkpoint, args, training, test,
                                                             metadata, outputs, device)
        training = HybridViews(training, setup)
        teacher_logits = VisualTeacherLogits(teacher) if args.teacher_logit_cache else None
        distiller = distiller_class(student, teacher_logits if teacher_logits is not None else teacher,
                                    SimpleNamespace(T=1, ALPHA=.5, CE_WEIGHT=.5, LOGIT_STAND=True))
    projector = nn.Linear(dimension, NUM_OUTPUTS).to(device)
    initial = state_hash(student, projector)
    optimizer = torch.optim.SGD(chain(student.parameters(), projector.parameters()), lr=.01, momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.1)
    order = PairedOrder(training, args.seed)
    loader_options = {"batch_size": BATCH_SIZE, "num_workers": args.workers,
                      "persistent_workers": args.workers > 0, "pin_memory": amp}
    batches = DataLoader(training, sampler=order, generator=torch.Generator().manual_seed(args.seed), **loader_options)
    heldout = DataLoader(test, **loader_options)
    directory = args.out / args.dataset / args.architecture / f"seed-{args.seed}" / args.method
    directory.mkdir(parents=True, exist_ok=True)
    config = {"dataset": args.dataset, "architecture": args.architecture, "seed": args.seed,
              "author_revision": AUTHOR_REVISION, "student_source_sha256": MODEL_SOURCES[args.architecture][3],
              "true_classes": true_classes, "classifier_outputs": outputs, "metadata_sha256": metadata,
              "epochs": EPOCHS, "batch_size": BATCH_SIZE, "precision": "bfloat16-mixed" if amp else "float32",
              "gamma": 1., "auxiliary_weight": beta, "target_kind": "qwen" if qwen else "ecoc",
              "target_source_sha256": sha256(args.target), "target_float64_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
              "teacher_sha256": teacher_sha, "teacher_selected_epoch": teacher_epoch,
              "initial_state_sha256": initial, "training_view_recipe": "author MLKD weak/strong" if hybrid else "author single view",
              "optimizer": {"lr": .01, "momentum": .9, "weight_decay": .0005,
                            "milestones": list(MILESTONES), "gamma": .1},
              "author_mlkd_source_sha256": SOURCE_HASHES if hybrid else None}
    config_path, last, best_path = (directory / name for name in ("config.json", "last.pt", "author-best.pt"))
    if config_path.exists() and json.loads(config_path.read_bytes()) != config:
        raise ValueError(f"Cannot change the frozen comparison settings at {directory}")
    if (directory / "results.json").is_file():
        return
    reference = paired_qwen_history(directory, config, args.method)
    write_json(config_path, config)
    write_json(directory / "input-cache.json", {
        "validation_crop_sha256": test.metadata["sha256"] if args.validation_cache else None,
        "training_rgb": str(args.training_rgb_cache) if args.training_rgb_cache else None,
        "teacher_logits": str(args.teacher_logit_cache) if args.teacher_logit_cache else None,
        "checkpoint_every_n_epochs": args.checkpoint_every,
    })
    history, best = [], {"epoch": None, "validation_nll": float("inf")}
    if last.is_file():
        recovered = torch.load(last, weights_only=True, map_location=device)
        if recovered["initial_state_sha256"] != initial:
            raise ValueError("Student initialization changed at resume")
        student.load_state_dict(recovered["student"], strict=True)
        projector.load_state_dict(recovered["auxiliary"], strict=True)
        optimizer.load_state_dict(recovered["optimizer"])
        scheduler.load_state_dict(recovered["scheduler"])
        history, best = recovered["history"], recovered["author_best"]
        stored = directory / "history.json"
        if stored.is_file():
            previous = json.loads(stored.read_bytes())
            if len(previous) > len(history):
                if previous[:len(history)] != history:
                    raise ValueError("The stored history does not match the last complete state")
                write_json(directory / f"history-before-resume-from-{len(history):03d}.json", previous)
        del recovered

    for epoch in range(len(history) + 1, EPOCHS + 1):
        started = time.perf_counter()
        order.epoch = epoch
        torch.manual_seed(1000003 * args.seed + epoch)
        plan_sha = hashlib.sha256(order.plan().tobytes()).hexdigest()
        if teacher_logits:
            path = args.teacher_logit_cache / f"seed-{args.seed}" / f"epoch-{epoch:03d}.npz"
            binding = {"seed": args.seed, "epoch": epoch, "teacher_sha256": teacher_sha, "batch": BATCH_SIZE,
                       "data_metadata_sha256": metadata, "paired_order_sha256": plan_sha,
                       "author_augmentation_sha256": SOURCE_HASHES["lavid/data/mlkd_augment.py"],
                       "author_view_setup_sha256": SOURCE_HASHES["lavid/data/base_dataset.py"],
                       "precision": config["precision"], "device": torch.cuda.get_device_name(device),
                       "torch": torch.__version__}
            mode = "record" if qwen and args.record_teacher_logits else "replay"
            teacher_logits.begin_epoch(path, mode, binding, len(training), outputs, device)
        if hybrid:
            row = hybrid_epoch(student, projector, distiller, batches, optimizer, target, device, amp, epoch,
                               teacher_logits, beta)
        else:
            row = single_view_epoch(student, projector, optimizer, batches, target, beta, device, amp)
        row.update(epoch=epoch, lr=optimizer.param_groups[0]["lr"], training_seconds=time.perf_counter()-started,
                   ordered_index_and_augmentation_seed_sha256=plan_sha)
        if reference and any(row[key] != reference[epoch - 1][key] for key in PAIRED_EPOCH_FIELDS):
            raise ValueError(f"The strong or weak batches, actual updates or LR changed for epoch {epoch}")
        scheduler.step()
        state = {"epoch": epoch, "student": student.state_dict(), "auxiliary": projector.state_dict(),
                 "initial_state_sha256": initial}
        if epoch == 1 or epoch % 5 == 0:
            before = time.perf_counter()
            scores = evaluate(student, heldout, device, amp, outputs)
            if not all(np.isfinite(list(scores.values()))):
                raise FloatingPointError("Non-finite Table 1 validation score")
            row.update(author_eval_split=scores, validation_seconds=time.perf_counter()-before)
            if epoch % 5 == 0 and scores["nll"] < best["validation_nll"]:
                best = {"epoch": epoch, "validation_nll": scores["nll"]}
                write_checkpoint(best_path, state)
        history.append(row)
        if epoch % args.checkpoint_every == 0 or epoch == EPOCHS:
            write_checkpoint(last, state | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                            "history": history, "author_best": best})
        write_json(directory / "history.json", history)
        if epoch <= 3 or "author_eval_split" in row:
            print(f"{args.dataset}/{args.architecture}/{args.method} seed={args.seed} epoch={epoch}/{EPOCHS} "
                  f"n={row['records']} updates={row['updates']} wall={time.perf_counter()-started:.2f} "
                  f"teacher_cache={row.get('teacher_logit_cache')}", flush=True)

    write_checkpoint(directory / "final.pt", {"epoch": EPOCHS, "student": student.state_dict(),
                                             "auxiliary": projector.state_dict(), "initial_state_sha256": initial})
    final = evaluate(student, heldout, device, False, outputs, directory / "final_predictions.npz")
    selected = torch.load(best_path, weights_only=True, map_location=device)
    if selected["epoch"] != best["epoch"]:
        raise ValueError("The saved selected epoch does not match the author's recorded validation rule")
    student.load_state_dict(selected["student"], strict=True)
    best_scores = evaluate(student, heldout, device, False, outputs, directory / "author_best_predictions.npz")
    saved = ("final.pt", "author-best.pt", "final_predictions.npz", "author_best_predictions.npz")
    write_json(directory / "results.json", {"dataset": args.dataset, "architecture": args.architecture,
                                             "method": args.method, "target_kind": config["target_kind"],
                                             "epochs": EPOCHS, "optimizer_steps": sum(row["updates"] for row in history),
                                             "teacher_sha256": teacher_sha, "initial_state_sha256": initial,
                                             "fixed_final_fp32": final, "author_selected_fp32": best_scores,
                                             "author_selection": best, "artifact_sha256": {f: sha256(directory / f) for f in saved}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=RELEASES, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    for flag in ("images", "author-source", "target", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("cohort", "teacher-checkpoint", "teacher-logit-cache", "training-rgb-cache", "validation-cache"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--record-teacher-logits", action="store_true")
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, choices=(1, 5), default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if bool(args.teacher_checkpoint) != (args.method != "ecoc-only"):
        parser.error("Provide the sealed teacher for hybrid arms; the single-view arm takes no teacher")
    if args.record_teacher_logits and (args.method != "lavid-ls-qwen" or not args.teacher_logit_cache):
        parser.error("Only the designated Qwen arm records shared teacher logits")
    if args.teacher_logit_cache and (args.method == "ecoc-only" or torch.device(args.device).type != "cuda"):
        parser.error("Teacher logit caching is for hybrid CUDA training")
    if args.workers < 0:
        parser.error("Worker count must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
