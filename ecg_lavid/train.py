"""Train and compare waveform-only CE and two LaViD signature assignments."""

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from . import CLASSES, LEADS, N_SAMPLES, SAMPLE_RATE
from .teacher import LETTERS, write_json


METHODS = ("ce", "lavid", "permuted")
CLASS_INDEX = np.arange(len(CLASSES))
PRIMARY_METRIC = "macro_f1"
METRICS = (PRIMARY_METRIC, "balanced_accuracy", "macro_auroc", "macro_auprc", "accuracy")


class ResidualBlock(nn.Sequential):
    def __init__(self, channels, dilation):
        super().__init__(
            nn.Conv1d(channels, channels, 5, dilation=dilation, padding=2 * dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return F.gelu(x + super().forward(x))


class ECGStudent(nn.Module):
    def __init__(self, question_shape):
        super().__init__()
        self.question_shape = tuple(question_shape)
        self.encoder = nn.Sequential(
            nn.Conv1d(len(LEADS), 32, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResidualBlock(32, dilation=1),
            nn.Conv1d(32, 48, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            ResidualBlock(48, dilation=1),
            nn.Conv1d(48, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualBlock(64, dilation=2),
            ResidualBlock(64, dilation=4),
        )
        self.classifier = nn.Linear(128, len(CLASSES))
        self.question_head = nn.Linear(128, math.prod(self.question_shape))

    def forward(self, ecg, with_auxiliary=False):
        features = self.encoder(ecg)
        pooled = torch.cat((features.mean(-1), features.amax(-1)), dim=1)
        logits = self.classifier(pooled)
        if with_auxiliary:
            return logits, self.question_head(pooled).reshape(-1, *self.question_shape)
        return logits


def load_data(data_path):
    with np.load(data_path, allow_pickle=False) as source:
        ecg, labels = source["waveforms"], source["labels"]
        ecg_ids, patient_ids, folds = source["ecg_id"], source["patient_id"], source["folds"]
        if tuple(source["class_codes"]) != tuple(CLASSES) or tuple(source["leads"]) != LEADS:
            raise ValueError("Unexpected class or lead order in the prepared dataset")
    if ecg.shape != (len(labels), len(LEADS), N_SAMPLES) or ecg.dtype != np.float32:
        raise ValueError("Expected ECG float32 array with shape records x 12 x 1000")
    if any(array.shape != (len(labels),) for array in (labels, ecg_ids, patient_ids, folds)):
        raise ValueError("Inconsistent length of the label and split arrays")
    if len(np.unique(ecg_ids)) != len(ecg_ids) or not np.isin(labels, CLASS_INDEX).all():
        raise ValueError("Duplicate ECG IDs or unknown labels in prepared dataset")
    if not np.isin(folds, np.arange(1, 11)).all():
        raise ValueError("Official folds must be in 1..10")

    return TensorDataset(torch.from_numpy(ecg), torch.from_numpy(labels).long()), ecg_ids, patient_ids, folds


def load_teacher(teacher_path):
    teacher = json.loads(teacher_path.read_text(encoding="utf-8-sig"))
    if tuple(teacher["class_codes"]) != tuple(CLASSES):
        raise ValueError("Teacher class order differs from the cohort")
    targets = torch.tensor(teacher["probabilities"], dtype=torch.float32)
    expected = (len(CLASSES), len(teacher["questions"]), len(teacher["answer_labels"]))
    if targets.shape != expected or expected[1] < 1 or expected[2] < 2:
        raise ValueError("Malformed class-by-question-by-option teacher probabilities")
    if not torch.isfinite(targets).all() or (targets < 0).any() or not torch.allclose(
        targets.sum(dim=-1), torch.ones(expected[:-1]), atol=1e-5,
    ):
        raise ValueError("Teacher probabilities must be finite and sum to one per question")
    return targets


def cohort_splits(folds, patient_ids, labels):
    splits = {
        "train": np.flatnonzero(folds <= 8),
        "validation": np.flatnonzero(folds == 9),
        "test": np.flatnonzero(folds == 10),
    }
    patients = {name: set(patient_ids[index]) for name, index in splits.items()}
    overlap = {f"{first}_{second}": len(patients[first] & patients[second])
               for first, second in itertools.combinations(patients, 2)}
    if any(overlap.values()):
        raise ValueError(f"Patient leakage across splits: {overlap}")
    if len(np.unique(np.stack((patient_ids, folds), axis=1), axis=0)) != len(np.unique(patient_ids)):
        raise ValueError("A patient was assigned to more than one official fold")
    for name, index in splits.items():
        if len(np.unique(labels[index])) != len(CLASSES):
            raise ValueError(f"Missing at least one class in the {name} split")
    return splits, overlap


def select_train_subset(train, labels, patients, fraction, seed):
    if fraction == 1.0:
        return train
    counts = np.bincount(labels[train], minlength=len(CLASSES))
    by_patient = {}
    for index in train:
        by_patient.setdefault(patients[index], set()).add(labels[index])
    strata = [[] for _ in CLASSES]
    for patient, label_set in sorted(by_patient.items()):
        rarest = min(label_set, key=lambda label: (counts[label], label))
        strata[rarest].append(patient)
    generator = np.random.default_rng(seed)
    selected = []
    for group in strata:
        if group:
            size = max(1, round(len(group) * fraction))
            selected.extend(generator.permutation(group)[:size])
    chosen = train[np.isin(patients[train], selected)]
    if len(np.unique(labels[chosen])) != len(CLASSES):
        raise ValueError("The selected label budget has no training examples for at least one class")
    return chosen


def class_derangement(seed):
    options = [order for order in itertools.permutations(CLASS_INDEX)
               if all(original != assigned for original, assigned in zip(CLASS_INDEX, order))]
    return tuple(int(index) for index in random.Random(seed).choice(options))


def fit_normalization(ecgs, indices, device):
    totals = np.zeros(len(LEADS), dtype=np.float64)
    square_totals = np.zeros_like(totals)
    for start in range(0, len(indices), 256):
        chunk = ecgs[indices[start:start + 256]]
        totals += chunk.sum(axis=(0, 2), dtype=np.float64)
        square_totals += np.square(chunk, dtype=np.float64).sum(axis=(0, 2))
    size = len(indices) * N_SAMPLES
    mean = totals / size
    std = np.sqrt(np.maximum(square_totals / size - mean ** 2, 1e-8))
    return tuple(torch.tensor(array, dtype=torch.float32, device=device)[None, :, None] for array in (mean, std))


def make_loader(dataset, indices, batch_size, device, seed=None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=seed is not None,
                      generator=generator, pin_memory=device.type == "cuda")


def score_predictions(labels, probabilities):
    predictions = probabilities.argmax(axis=-1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=CLASS_INDEX, zero_division=0,
    )
    one_vs_rest_auc = [roc_auc_score(labels == idx, probabilities[:, idx]) for idx in CLASS_INDEX]
    one_vs_rest_ap = [average_precision_score(labels == idx, probabilities[:, idx]) for idx in CLASS_INDEX]
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_auroc": float(np.mean(one_vs_rest_auc)),
        "macro_auprc": float(np.mean(one_vs_rest_ap)),
        "per_class": {code: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                             "support": int(support[i]), "auroc": one_vs_rest_auc[i], "auprc": one_vs_rest_ap[i]}
                      for i, code in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(labels, predictions, labels=CLASS_INDEX).tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, normalization, device, amp):
    model.eval()
    batches = []
    targets = []
    loss = torch.zeros((), device=device)
    mean, std = normalization
    for signals, labels in loader:
        signals = (signals.to(device, non_blocking=True) - mean) / std
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(signals).float()
        loss += F.cross_entropy(logits, labels.to(device, non_blocking=True), reduction="sum")
        batches.append(logits.softmax(-1).cpu())
        targets.append(labels)
    probabilities = torch.cat(batches).numpy()
    labels = torch.cat(targets).numpy()
    metrics = score_predictions(labels, probabilities)
    metrics["ce_loss"] = loss.item() / len(labels)
    return metrics, probabilities


def train_one(args, model, initial_state, signatures, train_loader, val_loader, test_loader, normalization, device, amp):
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mean, std = normalization
    best_metrics = {PRIMARY_METRIC: -math.inf}
    best_state = None
    history = []
    gradient = None
    classification_gradient = None
    first_batch_losses = None
    steps = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = torch.zeros(3, device=device)
        lr = optimizer.param_groups[0]["lr"]
        for signals, labels, *codes in train_loader:
            labels = labels.to(device, non_blocking=True)
            auxiliary_codes = codes[0].to(device, non_blocking=True) if codes else labels
            signals = (signals.to(device, non_blocking=True) - mean) / std
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                if signatures is None:
                    logits = model(signals)
                else:
                    logits, auxiliary = model(signals, with_auxiliary=True)
            ce_loss = F.cross_entropy(logits.float(), labels)
            if args.validation_only and classification_gradient is None:
                ce_gradient, = torch.autograd.grad(ce_loss, model.encoder[0].weight, retain_graph=True)
                classification_gradient = ce_gradient.norm().item()
                if not math.isfinite(classification_gradient):
                    raise RuntimeError("The classification loss produced a non-finite encoder gradient")
            if signatures is None:
                mse_loss = torch.zeros_like(ce_loss)
            else:
                mse_loss = F.mse_loss(auxiliary.float(), signatures[auxiliary_codes])
                if gradient is None:
                    encoder_gradient, = torch.autograd.grad(mse_loss, model.encoder[0].weight, retain_graph=True)
                    gradient = encoder_gradient.norm().item()
                    if not math.isfinite(gradient) or gradient <= 0:
                        raise RuntimeError("The auxiliary loss did not produce a finite nonzero encoder gradient")
            loss = ce_loss + args.lavid_weight * mse_loss
            if args.validation_only and first_batch_losses is None:
                first_batch_losses = describe_losses(ce_loss.item(), mse_loss.item(), args.lavid_weight)
            loss.backward()
            optimizer.step()
            totals += len(labels) * torch.stack((loss.detach(), ce_loss.detach(), mse_loss.detach()))
            steps += 1
        scheduler.step()
        val_metrics, _ = evaluate(model, val_loader, normalization, device, amp)
        train_loss, train_ce, train_mse = (totals / len(train_loader.dataset)).tolist()
        epoch_metrics = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_ce": train_ce,
                         "train_mse": train_mse, **{f"val_{name}": val_metrics[name] for name in METRICS}}
        if args.validation_only:
            loss_description = describe_losses(train_ce, train_mse, args.lavid_weight)
            epoch_metrics.update({key: loss_description[key] for key in ("weighted_mse_to_ce_ratio", "weighted_mse_loss_share")})
        history.append(epoch_metrics)
        if val_metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(f"  epoch {epoch:02}/{args.epochs}  train CE={train_ce:.4f}  MSE={train_mse:.4f}  "
              f"val macro-F1={val_metrics[PRIMARY_METRIC]:.4f}", flush=True)

    result = {"best_epoch": best_epoch, "optimizer_steps": steps, "aux_encoder_gradient_norm_first_batch": gradient,
              "validation": best_metrics, "history": history}
    probabilities = None
    if test_loader is not None:
        model.load_state_dict(best_state)
        result["test"], probabilities = evaluate(model, test_loader, normalization, device, amp)
    if args.validation_only:
        weighted_aux_gradient = args.lavid_weight * (gradient or 0.0)
        result["first_batch_losses"] = first_batch_losses
        result["ce_encoder_gradient_norm_first_batch"] = classification_gradient
        result["weighted_aux_to_ce_encoder_grad_norm_ratio_first_batch"] = (
            weighted_aux_gradient / classification_gradient if classification_gradient else None
        )
    result["train_and_eval_seconds"] = time.perf_counter() - start
    return result, best_state, probabilities


def describe_losses(ce, mse, weight):
    weighted_mse = weight * mse
    return {
        "ce": ce, "mse": mse, "weighted_mse": weighted_mse,
        "weighted_mse_to_ce_ratio": weighted_mse / ce if ce else None,
        "weighted_mse_loss_share": weighted_mse / (ce + weighted_mse) if ce + weighted_mse else None,
    }


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def counts_for(indices, labels, patients):
    counts = np.bincount(labels[indices], minlength=len(CLASSES))
    return {"records": len(indices), "patients": len(np.unique(patients[indices])),
            "per_class": {code: int(n) for code, n in zip(CLASSES, counts)}}


def write_subset_index(path, indices, ecg_ids, patient_ids, labels, folds):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["array_index", "ecg_id", "patient_id", "class_code", "strat_fold"])
        for index in indices:
            writer.writerow([index, ecg_ids[index], patient_ids[index], tuple(CLASSES)[labels[index]], folds[index]])


def aggregate_results(runs, report_split="test"):
    aggregates = {}
    for fraction in sorted({run["fraction"] for run in runs}):
        relevant = [run for run in runs if run["fraction"] == fraction]
        by_method = {method: {run["seed"]: run[report_split] for run in relevant if run["method"] == method}
                     for method in METHODS}
        averages = {method: {metric: statistics([metrics[metric] for metrics in per_seed.values()]) for metric in METRICS}
                    for method, per_seed in by_method.items() if per_seed}
        paired = {}
        for other in ("ce", "permuted"):
            seeds = sorted(by_method["lavid"].keys() & by_method[other].keys())
            if seeds:
                paired[f"lavid_minus_{other}"] = {
                    metric: statistics([by_method["lavid"][seed][metric] - by_method[other][seed][metric] for seed in seeds])
                    for metric in METRICS
                }
        aggregates[str(fraction)] = {report_split: averages, f"paired_{report_split}_deltas": paired}
    return aggregates


def statistics(values):
    return {"n": len(values), "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "values": values}


def make_row_shuffle_dataset(dataset, subset, seed, ecg_ids, setting):
    from .ecoc import shuffled_training_records

    receivers, donors = shuffled_training_records(subset, seed)
    labels = dataset.tensors[1]
    code_indices = labels.clone()
    code_indices[receivers] = labels[donors]
    with (setting / "code_row_assignments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["array_index", "ecg_id", "donor_ecg_id", "class_code", "assigned_code"])
        codes = tuple(CLASSES)
        for receiver, donor in zip(receivers, donors):
            writer.writerow([receiver, ecg_ids[receiver], ecg_ids[donor], codes[labels[receiver]], codes[labels[donor]]])
    return TensorDataset(*dataset.tensors, code_indices)


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    amp = args.precision == "bfloat16" or (
        args.precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    dataset, ecg_ids, patient_ids, folds = load_data(args.data)
    labels = dataset.tensors[1].numpy()
    splits, overlap = cohort_splits(folds, patient_ids, labels)
    report_split = "validation" if args.validation_only else "test"
    index_splits = ("validation",) if args.validation_only else ("validation", "test")
    permutation = class_derangement(args.permutation_seed)
    question_shape = (args.question_count, len(LETTERS))
    methods = {"ce": None}
    codebook = getattr(args, "codebook", None)
    shuffle_codebook_rows = getattr(args, "shuffle_codebook_rows", False)
    targets = None
    if args.teacher:
        targets = load_teacher(args.teacher)
    elif codebook:
        from .ecoc import load_codebook

        targets = torch.from_numpy(load_codebook(codebook))
    if targets is not None:
        targets = targets.to(device)
        question_shape = tuple(targets.shape[1:])
        methods.update(lavid=targets, permuted=targets[list(permutation)])
    config = {
        "data_sha256": sha256(args.data), "teacher_sha256": sha256(args.teacher) if args.teacher else None,
        "class_codes": list(CLASSES),
        "leads": list(LEADS), "question_shape": list(question_shape), "seeds": args.seeds, "fractions": args.fractions,
        "epochs": args.epochs, "batch_size": args.batch_size, "weight_decay": args.weight_decay, "init_lr": args.lr,
        "lavid_weight": args.lavid_weight, "precision": "bfloat16" if amp else "float32",
        "device": str(device), "cpu_threads": args.cpu_threads, "torch_version": str(torch.__version__),
        "permutation_seed": args.permutation_seed,
        "permuted_class_to_signature": dict(zip(CLASSES, [tuple(CLASSES)[index] for index in permutation])),
        "cross_split_patient_overlap": overlap,
        "full_cohort_counts": {name: counts_for(splits[name], labels, patient_ids) for name in ("train", *index_splits)},
        "training_inputs": {"ce": ["ECG", "rhythm label"], "lavid": ["ECG", "rhythm label", "matching teacher signature"],
                            "permuted": ["ECG", "rhythm label", "permuted teacher signature"]},
        "inference_inputs_all_groups": ["ECG"],
        "checkpoint_selection": "max validation macro F1, earliest epoch on ties",
    }
    if codebook:
        config["codebook_sha256"] = sha256(codebook)
        config["training_inputs"].update(lavid=["ECG", "rhythm label", "matching class codeword"],
                                         permuted=["ECG", "rhythm label", "permuted class codeword"])
    if shuffle_codebook_rows:
        from .ecoc import ROW_SHUFFLE_SEED

        config["codeword_row_shuffle_seed"] = ROW_SHUFFLE_SEED
        config["training_inputs"]["lavid"] = ["ECG", "rhythm label", "codeword from a fixed random training record"]
    if args.validation_only:
        config["evaluation_split"] = "validation_only"
    if device.type == "cuda":
        config["gpu"] = torch.cuda.get_device_name(device)
    existing = args.out / "config.json"
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if config["teacher_sha256"] is None:
            config["teacher_sha256"] = previous["teacher_sha256"]
        if previous["teacher_sha256"] is None:
            previous["teacher_sha256"] = config["teacher_sha256"]
        if previous != config:
            raise ValueError("Output directory already contains a different experiment configuration")
    write_json(existing, config)
    for split in index_splits:
        write_subset_index(args.out / f"{split}_indices.csv", splits[split], ecg_ids, patient_ids, labels, folds)

    val_loader = make_loader(dataset, splits["validation"], args.batch_size, device)
    test_loader = None
    if not args.validation_only:
        test_loader = make_loader(dataset, splits["test"], args.batch_size, device)
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = ECGStudent(question_shape)
        initial_state = copy.deepcopy(model.state_dict())
        init_hash = state_hash(initial_state)
        model.to(device)
        for fraction in args.fractions:
            subset = select_train_subset(splits["train"], labels, patient_ids, fraction, seed)
            setting = args.out / f"fraction-{fraction}_seed-{seed}"
            setting.mkdir(parents=True, exist_ok=True)
            subset_path = setting / "train_indices.csv"
            write_subset_index(subset_path, subset, ecg_ids, patient_ids, labels, folds)
            subset_hash = sha256(subset_path)
            normalization = fit_normalization(dataset.tensors[0].numpy(), subset, device)
            shuffled_data = make_row_shuffle_dataset(dataset, subset, seed, ecg_ids, setting) if shuffle_codebook_rows else None
            for name in METHODS:
                result_path = setting / f"{name}.json"
                checkpoint_path = setting / f"{name}.pt"
                required_paths = [result_path, checkpoint_path]
                if test_loader is not None:
                    predictions_path = setting / f"{name}_test_predictions.npz"
                    required_paths.append(predictions_path)
                completed = all(path.is_file() for path in required_paths)
                if not completed and name not in args.methods:
                    continue
                print(f"fraction={fraction}  seed={seed}  method={name}  training ECGs={len(subset)}", flush=True)
                if completed:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if result["initial_state_sha256"] != init_hash or result["train_indices_sha256"] != subset_hash:
                        raise ValueError(f"Existing result is inconsistent with current initialization or subset: {result_path}")
                    print("  loaded completed result", flush=True)
                else:
                    training_data = shuffled_data if name == "lavid" and shuffled_data is not None else dataset
                    loader = make_loader(training_data, subset, args.batch_size, device, seed)
                    result, state, probabilities = train_one(
                        args, model, initial_state, methods[name], loader, val_loader, test_loader, normalization, device, amp,
                    )
                    result.update({"seed": seed, "fraction": fraction, "method": name,
                                   "initial_state_sha256": init_hash, "train_indices_sha256": subset_hash,
                                   "train_count": counts_for(subset, labels, patient_ids)})
                    temporary = checkpoint_path.with_suffix(".pt.tmp")
                    torch.save({"model_state": state, "question_shape": question_shape, "class_codes": tuple(CLASSES),
                                "leads": LEADS, "sample_rate": SAMPLE_RATE,
                                "normalization": tuple(t.cpu() for t in normalization)}, temporary)
                    temporary.replace(checkpoint_path)
                    if test_loader is not None:
                        np.savez(predictions_path, ecg_id=ecg_ids[splits["test"]], patient_id=patient_ids[splits["test"]],
                                 array_index=splits["test"], class_codes=np.array(tuple(CLASSES)),
                                 y_true=labels[splits["test"]], probabilities=probabilities)
                    write_json(result_path, result)
                results.append(result)
                write_json(args.out / "results.json", {"runs": results, "summary_by_fraction": aggregate_results(results, report_split)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--teacher", type=Path, help="Class-by-question language-model probabilities")
    target.add_argument("--codebook", type=Path, help="Fixed class-index-only ECOC targets")
    parser.add_argument("--shuffle-codebook-rows", action="store_true",
                        help="Use a fixed train-subset row permutation for the lavid codeword target")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--lavid-weight", type=float, default=40.0)
    parser.add_argument("--permutation-seed", type=int, default=104729)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--question-count", type=int, default=12, help="Size of the unused auxiliary head in a CE-only run without a teacher")
    parser.add_argument("--validation-only", action="store_true", help="Evaluate on validation only, with no test predictions or scores")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1 or args.question_count < 1:
        parser.error("Epochs, batch size, CPU thread count and question count must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be distinct")
    if args.teacher is None and args.codebook is None and args.methods != ["ce"]:
        parser.error("--teacher or --codebook is required unless only --methods ce is selected")
    if args.shuffle_codebook_rows and (args.codebook is None or "lavid" not in args.methods):
        parser.error("--shuffle-codebook-rows requires --codebook and the lavid method")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    if len(set(args.fractions)) != len(args.fractions) or any(not 0 < fraction <= 1 for fraction in args.fractions):
        parser.error("Fractions must be distinct numbers in (0, 1]")
    if not all(math.isfinite(value) for value in (args.lr, args.weight_decay, args.lavid_weight)) or (
        args.lr <= 0 or args.weight_decay < 0 or args.lavid_weight <= 0
    ):
        parser.error("Learning rate and LaViD weight must be positive and weight decay must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
