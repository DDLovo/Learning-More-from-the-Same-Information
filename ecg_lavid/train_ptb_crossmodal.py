"""Matched patient-budget rhythm classification for the original CNN and a BiLSTM."""

import argparse
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter, time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import CLASSES, LEADS
from .ecoc import load_codebook
from .teacher import write_json
from .train import ECGStudent, cohort_splits, fit_normalization, load_data, select_train_subset, sha256, state_hash


SEEDS = (17, 29, 43, 71, 101)
FRACTIONS = (.1, .25, .5, 1.)
ARCHITECTURES = ("cnn", "bilstm")
ALL_ARCHITECTURES = (*ARCHITECTURES, "resnet1d")
ARMS = ("A", "H")
REFERENCE_ARMS = ("smooth", "squentropy", "qwen")
PURE_SQUARED_ARMS = ("raw-onehot", "prob-onehot")
GRADIENT_MATCHED_ARM = "ecoc-soft-ce-gradmatch"
JOINT_LOSS_ARMS = ("ce-same-onehot", "ce-independent-onehot", "ecoc-soft-ce", GRADIENT_MATCHED_ARM)


class BiLSTMStudent(nn.Module):
    """A shared 50-ms-stride input stem and 128-D bidirectional recurrent readout."""

    def __init__(self, question_shape):
        super().__init__()
        self.question_shape = tuple(question_shape)
        self.encoder = nn.Sequential(nn.Conv1d(len(LEADS), 32, 11, stride=5, padding=5, bias=False),
                                     nn.BatchNorm1d(32), nn.GELU())
        self.recurrent = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(128, len(CLASSES))
        self.question_head = nn.Linear(128, int(np.prod(question_shape)))

    def forward(self, ecg, with_auxiliary=False):
        sequence = self.encoder(ecg).transpose(1, 2).contiguous()
        # Use FP32 through the recurrent operator, with mixed precision in the stem and heads.
        with torch.autocast(ecg.device.type, enabled=False):
            sequence, _ = self.recurrent(sequence.float())
        pooled = sequence.mean(dim=1)
        output = self.classifier(pooled)
        return (output, self.question_head(pooled).reshape(-1, *self.question_shape)) if with_auxiliary else output


class BasicBlock1d(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm1d(cout),
                                    nn.ReLU(), nn.Conv1d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm1d(cout))
        self.shortcut = (nn.Identity() if cin == cout and stride == 1 else
                         nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)))

    def forward(self, x):
        return F.relu(self.branch(x) + self.shortcut(x))


class ResNet18Student1d(nn.Module):
    """Four two-block residual stages, with the same 128-D output as the primary students."""

    def __init__(self, question_shape):
        super().__init__()
        self.question_shape = tuple(question_shape)
        stages = []
        cin = 32
        for i, cout in enumerate((32, 64, 96, 128)):
            stages += [BasicBlock1d(cin, cout, 1 if i == 0 else 2), BasicBlock1d(cout, cout, 1)]
            cin = cout
        self.encoder = nn.Sequential(nn.Conv1d(len(LEADS), 32, 15, stride=2, padding=7, bias=False),
                                     nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(3, stride=2, padding=1), *stages)
        self.classifier = nn.Linear(128, len(CLASSES))
        self.question_head = nn.Linear(128, int(np.prod(question_shape)))

    def forward(self, ecg, with_auxiliary=False):
        pooled = self.encoder(ecg).mean(-1)
        output = self.classifier(pooled)
        return (output, self.question_head(pooled).reshape(-1, *self.question_shape)) if with_auxiliary else output


def build_network(architecture):
    models = {"cnn": ECGStudent, "bilstm": BiLSTMStudent, "resnet1d": ResNet18Student1d}
    return models[architecture]((12, 5))


def run_directory(root, architecture, fraction, seed, arm):
    return root / architecture / f"fraction-{fraction:g}" / f"seed-{seed}" / arm


def save_checkpoint(path, payload):
    part = path.with_name(f"{path.name}.part-{os.getpid()}")
    torch.save(payload, part)
    part.replace(path)


def freeze_json(path, payload):
    if path.is_file():
        if json.loads(path.read_bytes()) != payload:
            raise ValueError(f"The frozen run protocol has changed at {path}")
    else:
        write_json(path, payload)


def hash_values(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def metrics(truth, prediction):
    width = len(CLASSES)
    counts = np.bincount(width * truth + prediction, minlength=width * width).reshape(width, width)
    actual, assigned, true = counts.sum(1), counts.sum(0), counts.diagonal()
    recall = np.divide(true, actual, out=np.zeros(width), where=actual > 0)
    precision = np.divide(true, assigned, out=np.zeros(width), where=assigned > 0)
    f1 = np.divide(2 * true, actual + assigned, out=np.zeros(width), where=actual + assigned > 0)
    return {"accuracy": float(true.sum() / counts.sum()), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "confusion_matrix": counts.tolist(),
            "per_class": {code: {"support": int(actual[i]), "precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i])} for i, code in enumerate(CLASSES)}}


def orders(indices, seed, epochs, exposure):
    generator = np.random.default_rng(np.random.SeedSequence([11127, seed]))
    repeats = (exposure + len(indices) - 1) // len(indices)
    return np.stack([np.concatenate([generator.permutation(indices) for _ in range(repeats)])[:exposure]
                     for _ in range(epochs)])


def model_state(model):
    return {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}


def batch_options(args):
    return {"device_type": args.device.type, "dtype": torch.bfloat16, "enabled": args.device.type == "cuda"}


@torch.inference_mode()
def infer(model, x, order, normalization, batch_size, autocast):
    model.eval()
    center, scale = normalization
    outputs = []
    for rows in order.split(batch_size):
        with torch.autocast(**autocast):
            outputs.append(model((x[rows] - center) / scale).float().cpu())
    return torch.cat(outputs).numpy()


def onehot_square(logits, label):
    target = F.one_hot(label, num_classes=len(CLASSES)).to(logits.dtype)
    return F.mse_loss(logits, target)


def option_cross_entropy(logits, targets):
    return -(F.log_softmax(logits.float(), dim=-1) * targets).sum(dim=-1).mean()


def objectives(arm, model, signals, label, code, autocast):
    has_auxiliary = arm in ("H", "qwen", "ce-independent-onehot", "ecoc-soft-ce", GRADIENT_MATCHED_ARM)
    with torch.autocast(**autocast):
        out = model(signals, with_auxiliary=has_auxiliary)
    logits, auxiliary = out if has_auxiliary else (out, None)
    logits = logits.float()
    ce = F.cross_entropy(logits, label, label_smoothing=.1 if arm == "smooth" else 0.)
    extra = torch.zeros_like(ce)
    if arm in ("H", "qwen"):
        extra = F.mse_loss(auxiliary.float(), code[label])
    elif arm in ("raw-onehot", "ce-same-onehot"):
        extra = onehot_square(logits, label)
    elif arm == "prob-onehot":
        extra = onehot_square(logits.softmax(dim=-1), label)
    elif arm == "ce-independent-onehot":
        extra = onehot_square(auxiliary.float().squeeze(1), label)
    elif arm in ("ecoc-soft-ce", GRADIENT_MATCHED_ARM):
        extra = option_cross_entropy(auxiliary, code[label])
    elif arm == "squentropy":
        nontarget = ~F.one_hot(label, num_classes=logits.shape[-1]).bool()
        extra = logits[nontarget].square().mean()
    return logits, ce, extra


def match_initial_auxiliary_gradients(args, path, model, first_rows, x, y, stats, code, initial_state, common):
    calibration_path = path / "initial-gradient-match.json"
    binding = {"batch_size": args.batch_size, "codebook_sha256": common["codebook_sha256"], "precision": common["precision"],
               "lambda_reference_mse": args.weight, "shared_parameter": "encoder.0.weight",
               "training_order_sha256": common["training_order_sha256"], "initial_state_sha256": initial_state,
               "first_batch_rows_sha256": hash_values(first_rows.astype("<i8"))}
    if calibration_path.is_file():
        result = json.loads(calibration_path.read_bytes())
        if result["binding"] != binding:
            raise ValueError(f"Gradient match disagrees with the frozen initialization or batch: {calibration_path}")
        return result

    rows = torch.as_tensor(first_rows, device=args.device)
    center, scale = stats
    signal = (x[rows] - center) / scale
    targets = code[y[rows]]
    cuda_devices = [args.device] if args.device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        probe = deepcopy(model).train()
        shared_weight = probe.encoder[0].weight
        with torch.autocast(**batch_options(args)):
            _, auxiliary = probe(signal, with_auxiliary=True)
        mse = F.mse_loss(auxiliary.float(), targets)
        option_ce = option_cross_entropy(auxiliary, targets)
        grad_mse = torch.autograd.grad(mse, shared_weight, retain_graph=True)[0].float()
        grad_ce = torch.autograd.grad(option_ce, shared_weight)[0].float()
    mse_norm = float(torch.linalg.vector_norm(grad_mse))
    ce_norm = float(torch.linalg.vector_norm(grad_ce))
    if not np.isfinite([mse_norm, ce_norm]).all() or min(mse_norm, ce_norm) <= 0:
        raise ValueError("The shared layer did not produce positive, finite reference gradient norms")
    weight = args.weight * mse_norm / ce_norm
    result = {"binding": binding, "shared_parameter_shape": list(shared_weight.shape),
              "shared_parameter_initial_sha256": hash_values(shared_weight.detach().cpu().numpy()),
              "mse_grad_l2": mse_norm, "option_ce_grad_l2": ce_norm, "lambda_gradient_matched_option_ce": weight,
              "weighted_mse_grad_l2": args.weight * mse_norm, "weighted_option_ce_grad_l2": weight * ce_norm}
    freeze_json(calibration_path, result)
    return result


def fit(args, architecture, fraction, seed, arm, x, y, y_cpu, validation, order, stats, targets, common):
    path = run_directory(args.out, architecture, fraction, seed, arm)
    path.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(seed)
    model = build_network(architecture)
    initial_state = state_hash(model.state_dict())
    model = model.to(args.device)
    is_code = arm in ("H", "qwen", *JOINT_LOSS_ARMS)
    weight = args.weight if is_code else float(arm == "squentropy" or arm in PURE_SQUARED_ARMS)
    if arm == GRADIENT_MATCHED_ARM:
        calibration = match_initial_auxiliary_gradients(args, path, model, order[0, :args.batch_size],
                                                        x, y, stats, targets, initial_state, common)
        weight = calibration["lambda_gradient_matched_option_ce"]
    if arm == "ce-independent-onehot":
        model.question_head = deepcopy(model.classifier)
        model.question_shape = (1, len(CLASSES))
    config = common | {"architecture": architecture, "seed": seed, "fraction": fraction, "arm": arm,
                       "initial_state_sha256": initial_state, "auxiliary_or_penalty_weight": weight,
                       "model_parameters": sum(p.numel() for p in model.parameters()),
                       "classification_parameters": sum(p.numel() for name, p in model.named_parameters()
                                                        if not name.startswith("question_head."))}
    if arm in PURE_SQUARED_ARMS:
        config["classification_objective"] = arm
        config["cross_entropy_optimized"] = False
    if arm in JOINT_LOSS_ARMS:
        config["method_initial_state_sha256"] = state_hash(model_state(model))
    if arm == GRADIENT_MATCHED_ARM:
        config["initial_auxiliary_gradient_match"] = calibration
    freeze_json(path / "config.json", config)
    if (path / "seal.json").is_file():
        return True
    center, scale = stats
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule_length = config["planned_optimizer_steps"] if args.match_full_updates else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_length)
    state, best, history, elapsed, steps, peak = None, None, [], 0., 0, 0
    if (path / "last.pt").is_file():
        state = torch.load(path / "last.pt", weights_only=True, map_location=args.device)
        if state["config"] != config:
            raise ValueError(f"Saved state disagrees with {path}/config.json")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, history, elapsed, steps, peak = (state[k] for k in ("best", "history", "elapsed", "steps", "peak"))
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    autocast = batch_options(args)
    val_tensor = torch.as_tensor(validation, device=args.device)
    since_start = perf_counter()
    for epoch in range(len(history) + 1, args.epochs + 1):
        if time() >= args.deadline:
            print(f"Deadline reached at the committed epoch boundary: {path} epoch {epoch-1}", flush=True)
            return False
        model.train()
        totals = torch.zeros(4, device=args.device)
        batches = torch.as_tensor(order[epoch - 1], device=args.device).split(args.batch_size)
        learning_rate = optimizer.param_groups[0]["lr"]
        started = perf_counter()
        for ids in batches:
            signal, label = (x[ids] - center) / scale, y[ids]
            optimizer.zero_grad(set_to_none=True)
            logits, ce, extra = objectives(arm, model, signal, label, targets, autocast)
            loss = extra if arm in PURE_SQUARED_ARMS else ce + weight * extra
            loss.backward()
            optimizer.step()
            if args.match_full_updates:
                scheduler.step()
            totals += torch.stack((ce.detach() * len(ids), extra.detach() * len(ids),
                                   loss.detach() * len(ids), (logits.detach().argmax(1) == label).sum()))
            steps += 1
        ce, extra, loss, accuracy = (totals / len(order[epoch - 1])).cpu().tolist()
        if not np.isfinite([ce, extra, loss, accuracy]).all():
            raise FloatingPointError(f"Non-finite training values in {path} at epoch {epoch}")
        seconds_training = perf_counter() - started
        logits = infer(model, x, val_tensor, stats, args.batch_size, autocast)
        if not np.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite validation logits in {path} at epoch {epoch}")
        validation_score = metrics(y_cpu[validation], logits.argmax(1))
        row = {"epoch": epoch, "optimizer_steps": steps, "lr": learning_rate, "train_ce": ce, "train_aux_or_penalty": extra,
               "train_objective": loss, "train_accuracy": accuracy, "train_seconds": seconds_training,
               "validation": validation_score | {"nll": float(F.cross_entropy(torch.from_numpy(logits), torch.as_tensor(y_cpu[validation])))}}
        history.append(row)
        if best is None or validation_score["macro_f1"] > best["validation"]["macro_f1"]:
            best = row
            save_checkpoint(path / "best.pt", {"config": config, "epoch": epoch, "state": model_state(model),
                                              "mean": center.cpu(), "std": scale.cpu()})
        if not args.match_full_updates:
            scheduler.step()
        if args.device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated(args.device))
        save_checkpoint(path / "last.pt", {"config": config, "model": model_state(model), "optimizer": optimizer.state_dict(),
                                          "scheduler": scheduler.state_dict(), "best": best, "history": history,
                                          "elapsed": elapsed + perf_counter() - since_start, "steps": steps, "peak": peak})
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"{architecture} {fraction:g} {seed} {arm} epoch {epoch}/{args.epochs} steps {steps}: "
                  f"validation F1={validation_score['macro_f1']:.4f}, train seconds={seconds_training:.2f}", flush=True)
    save_checkpoint(path / "final.pt", {"config": config, "epoch": args.epochs, "state": model_state(model),
                                       "mean": center.cpu(), "std": scale.cpu()})
    write_json(path / "history.json", history)
    write_json(path / "seal.json", {"epochs": args.epochs, "steps": steps, "best_epoch": best["epoch"],
                                    "best_val_macro_f1": best["validation"]["macro_f1"],
                                    "elapsed_seconds": elapsed + perf_counter() - since_start,
                                    "train_epoch_seconds_total": sum(row["train_seconds"] for row in history),
                                    "max_cuda_allocated_bytes": peak, "best_sha256": sha256(path / "best.pt"),
                                    "final_sha256": sha256(path / "final.pt"), "history_sha256": sha256(path / "history.json")})
    return True


def scope(args):
    dataset, ids, patients, folds = load_data(args.data)
    x, y = dataset.tensors
    truth = y.numpy()
    split, overlaps = cohort_splits(folds, patients, truth)
    out = {"ids": ids, "patients": patients, "folds": folds, "truth": truth, "split": split}
    args.out.mkdir(parents=True, exist_ok=True)
    groups = {part: {"records": len(rows), "patients": int(np.unique(patients[rows]).size),
                     "class_counts": np.bincount(truth[rows], minlength=len(CLASSES)).tolist(),
                     "ordered_id_sha256": hash_values(ids[rows].astype("<i8"))} for part, rows in split.items()}
    freeze_json(args.out / "cohort.json", {"class_codes": list(CLASSES), "leads": list(LEADS), "sampling_hz": 100,
                                          "patient_overlap": overlaps, "groups": groups})
    return x, y, out


def training(args):
    x, y, info = scope(args)
    x_cpu = x.numpy()
    x, y = x.to(args.device), y.to(args.device)
    split, y_cpu = info["split"], info["truth"]
    code = torch.as_tensor(load_codebook(args.codebook), device=args.device)
    teacher = None
    if "qwen" in args.arms:
        from .train import load_teacher
        teacher = load_teacher(args.teacher).to(args.device)
        if teacher.shape != code.shape:
            raise ValueError("The Qwen reference must use the same auxiliary head shape")
    protocol = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
                "codebook_sha256": sha256(args.codebook), "lambda_h": args.weight,
                "precision": "BF16 stem/heads; FP32 BiLSTM" if args.device.type == "cuda" else "float32",
                "selection": "highest fold-9 macro F1, earlier epoch breaks ties", "class_codes": list(CLASSES),
                "optim": "AdamW; cosine each optimizer step" if args.match_full_updates else "AdamW; cosine each epoch",
                "regime": "official PTB-XL v1.0.3 four rhythms",
                "schedule": "matched full-budget updates and exposures" if args.match_full_updates else "natural-size epochs",
                "teacher_sha256": sha256(args.teacher) if teacher is not None else None}
    freeze_json(args.out / "protocol.json", protocol)
    already_selected = {seed: set() for seed in args.seeds}
    budget_path = args.out / "patients"
    budget_path.mkdir(exist_ok=True)
    for fraction in sorted(args.fractions):
        for seed in args.seeds:
            train = select_train_subset(split["train"], y_cpu, info["patients"], fraction, seed)
            p = np.unique(info["patients"][train])
            if not already_selected[seed].issubset(set(p)):
                raise ValueError(f"Patient budgets are not nested for seed {seed}, fraction {fraction}")
            already_selected[seed] = set(p)
            exposure = len(split["train"]) if args.match_full_updates else len(train)
            indices = orders(train, seed, args.epochs, exposure)
            stats = fit_normalization(x_cpu, train, args.device)
            meta = {"train_records": len(train), "train_patients": len(p),
                    "class_train_counts": np.bincount(y_cpu[train], minlength=len(CLASSES)).tolist(),
                    "record_exposures_per_epoch": exposure,
                    "planned_optimizer_steps": args.epochs * ((exposure + args.batch_size - 1) // args.batch_size),
                    "training_order_sha256": hash_values(info["ids"][indices].astype("<i8")),
                    "normalization": {k: v.cpu().reshape(-1).tolist() for k, v in zip(("mean", "std"), stats)}}
            freeze_json(budget_path / f"fraction-{fraction:g}-seed-{seed}.json", meta |
                        {"patient_ids": p.tolist(), "train_ecg_ids": info["ids"][train].tolist()})
            for architecture in args.architectures:
                for arm in args.arms:
                    if not fit(args, architecture, fraction, seed, arm, x, y, y_cpu, split["validation"], indices, stats,
                               teacher if arm == "qwen" else code, protocol | meta):
                        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "codebook", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--architectures", nargs="+", choices=ALL_ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--fractions", nargs="+", type=float, choices=FRACTIONS, default=list(FRACTIONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=ARMS + REFERENCE_ARMS + PURE_SQUARED_ARMS + JOINT_LOSS_ARMS,
                        default=list(ARMS))
    parser.add_argument("--match-full-updates", action="store_true", help="Match the full training fold's exposure and per-step cosine at every budget")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--weight", type=float, default=.4)
    parser.add_argument("--deadline", type=lambda stamp: datetime.fromisoformat(stamp).timestamp(), default=float("inf"))
    args = parser.parse_args()
    if "qwen" in args.arms and args.teacher is None:
        parser.error("Qwen reference requires the archived teacher input")
    return args


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = parse_args()
    raise SystemExit(0 if training(args) else 75)


if __name__ == "__main__":
    main()
