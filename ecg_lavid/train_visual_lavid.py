"""Run matched visual students on the author Flowers-102 or CUB-200 recipe."""

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
from itertools import chain
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
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import v2

from .teacher import write_json
from .train import describe_losses, sha256
from .visual_reference import AUTHOR_REVISION, author_model


SHAPE = (50, 5)
NUM_OUTPUTS = math.prod(SHAPE)
BATCH_SIZE = 16
MILESTONES = (150, 180, 210)
DATASETS = {"flowers": (102, 2040, 6149, 50.), "cub200": (200, 5994, 5794, 40.)}
ARMS = ("ce", "qwen", "qwen-permuted", "muse", "muse-permuted", "ecoc",
        "ecoc-hard", "qwen-span", "qwen-complement", "record-shuffle",
        "ecoc-constant", "ecoc-repeat", "ecoc-simplex",
        "raw-onehot", "prob-onehot", "ce-same-onehot", "squentropy", "ce-independent-onehot",
        "ecoc-soft-ce", "ecoc-soft-ce-gradmatch")
CODE_CONTROLS = ("ecoc-constant", "ecoc-repeat", "ecoc-simplex")
SOFT_CE = ("ecoc-soft-ce", "ecoc-soft-ce-gradmatch")
PURE_SQUARE = ("raw-onehot", "prob-onehot")
CLASS_HEAD_ARMS = (*PURE_SQUARE, "ce-same-onehot", "squentropy", "ce-independent-onehot")
RAW_SOURCES = {"flowers": ("imagelabels.mat", "setid.mat"),
               "cub200": ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")}
MASKED_SCOPE_SHA = "6b291faf4e157160e3d7e60a2fe6d0dbba8634fcc71915ba1c55cd72a10cbf1d"


def transforms(training):
    spatial = [v2.RandomResizedCrop(224), v2.RandomHorizontalFlip()] if training else [v2.Resize(256), v2.CenterCrop(224)]
    return v2.Compose([v2.ToImage(), *spatial, v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def read_cub_table(path, cast):
    with path.open(encoding="utf-8") as stream:
        pairs = [(int(index), cast(value)) for index, value in (line.strip().split(maxsplit=1) for line in stream)]
    table = dict(pairs)
    if len(table) != len(pairs):
        raise ValueError(f"Duplicated CUB IDs in {path.name}")
    return table


class VisualDataset(Dataset):
    def __init__(self, root, name, training):
        class_count, train_n, test_n, _ = DATASETS[name]
        if name == "flowers":
            split = loadmat(root / "setid.mat")
            ids = np.concatenate((split["trnid"].ravel(), split["valid"].ravel())) if training else split["tstid"].ravel()
            labels = loadmat(root / "imagelabels.mat")["labels"].ravel()[ids - 1] - 1
            paths = [root / "jpg" / f"image_{index:05d}.jpg" for index in ids]
        else:
            filenames = read_cub_table(root / "images.txt", str)
            targets = read_cub_table(root / "image_class_labels.txt", int)
            split = read_cub_table(root / "train_test_split.txt", int)
            if set(filenames) != set(targets) or set(split) != set(filenames) or set(split.values()) != {0, 1}:
                raise ValueError("CUB files disagree on the official data split")
            ids = [key for key in filenames if split[key] == int(training)]
            labels = [targets[key] - 1 for key in ids]
            paths = [root / "images" / filenames[key] for key in ids]
        self.ids = np.asarray(ids, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.paths = paths
        self.transform = transforms(training)
        if (len(self.ids) != (train_n if training else test_n) or np.unique(self.ids).size != len(self.ids)
                or set(self.labels) != set(range(class_count))):
            raise ValueError("Unexpected record counts, duplicate IDs or class support")
        if name == "flowers" and training and not np.all(np.bincount(self.labels) == 20):
            raise ValueError("Expected 20 training images from each flower class")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        index, seed = item if isinstance(item, tuple) else (item, None)
        if seed is not None:
            torch.random.default_generator.manual_seed(int(seed))
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.labels[index]), index


class PairedOrder(Sampler):
    """Use the same permutation and per-image augmentation seed in each arm and restart."""

    def __init__(self, dataset, seed):
        self.image_ids = dataset.ids
        self.seed = seed
        self.epoch = None

    def __len__(self):
        return len(self.image_ids)

    def plan(self):
        random = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 260511]))
        order = random.permutation(len(self))
        seeds = random.integers(0, 2 ** 32, size=len(self), dtype=np.int64)
        return np.column_stack((order, seeds))

    def __iter__(self):
        return iter([tuple(pair) for pair in self.plan().tolist()])


def read_signatures(path, names):
    values = json.loads(path.read_bytes())
    if set(values) != set(names):
        raise ValueError("Signature class names differ from the pinned author index")
    rows = np.asarray([values[name] for name in names], dtype=np.float64)
    if rows.shape != (len(names), *SHAPE) or not np.isfinite(rows).all() or (rows < 0).any():
        raise ValueError("Expected a finite nonnegative 50×5 signature for each label")
    np.testing.assert_allclose(rows.sum(axis=-1), 1., atol=.005, rtol=0)
    return rows


def centered_energy(rows):
    return float(np.square(rows - rows.mean(axis=0)).mean())


def code_controls(symbols, scaled, scale):
    n = len(scaled)
    repeat_symbols = (symbols[:, :1] + np.arange(SHAPE[0])) % SHAPE[1]
    for q in range(SHAPE[0]):
        np.testing.assert_array_equal(np.bincount(repeat_symbols[:, q], minlength=5),
                                      np.bincount(symbols[:, q], minlength=5))
    repeated = .2 + scale * (np.eye(5)[repeat_symbols] - .2)
    mean = scaled.mean(axis=0)
    np.testing.assert_allclose(repeated.mean(axis=0), mean, atol=1e-14, rtol=0.)
    np.testing.assert_allclose(centered_energy(repeated), centered_energy(scaled), atol=1e-14, rtol=0.)
    centered = (scaled - mean).reshape(n, -1)
    u, s, v = np.linalg.svd(centered, full_matrices=False)
    rank = np.count_nonzero(s > s[0] * max(centered.shape) * np.finfo(float).eps)
    if rank != n - 1:
        raise ValueError("A full class-contrast space is required for the matched simplex")
    alpha = math.sqrt(np.square(s).sum() / rank)
    simplex = mean + (alpha * (u[:, :rank] @ v[:rank])).reshape(scaled.shape)
    np.testing.assert_allclose(simplex.mean(axis=0), mean, atol=1e-14, rtol=0.)
    np.testing.assert_allclose(centered_energy(simplex), centered_energy(scaled), atol=1e-14, rtol=0.)
    gram = ((simplex - mean).reshape(n, -1) @ (simplex - mean).reshape(n, -1).T)
    np.testing.assert_allclose(gram, alpha ** 2 * (np.eye(n) - 1 / n), atol=1e-12, rtol=0.)
    return {"ecoc-constant": mean[None, :], "ecoc-repeat": repeated, "ecoc-simplex": simplex}, {
        "repeat_symbols_zero_based": repeat_symbols.tolist(), "constant_option_means": mean.tolist(),
        "simplex_rank": int(rank), "simplex_equal_pair_squared_distance": float(2 * alpha ** 2),
        "simplex_min": float(simplex.min()), "simplex_max": float(simplex.max()),
        "simplex_targets": simplex.tolist(),
    }


def make_targets(qwen, muse, names, train, seed, enabled, destination):
    n = len(names)
    mapping_rng = np.random.default_rng(104729)
    cycle = mapping_rng.permutation(n)
    mapping = np.empty(n, dtype=int)
    mapping[cycle] = np.roll(cycle, -1)
    if len(set(mapping)) != n or np.any(mapping == np.arange(n)):
        raise ValueError("Incorrect class derangement")
    ecoc_rng = np.random.default_rng(271828)
    symbols = np.column_stack([ecoc_rng.permutation((np.arange(n) + q) % 5) for q in range(SHAPE[0])])
    hard = np.eye(5, dtype=np.float64)[symbols]
    scale = math.sqrt(centered_energy(qwen) / centered_energy(hard))
    scaled = .2 + scale * (hard - .2)
    if (scaled < 0).any() or (scaled > 1).any():
        raise ValueError("Energy matching would produce a target outside probability support")
    np.testing.assert_allclose(centered_energy(scaled), centered_energy(qwen), atol=1e-14, rtol=0)
    flattened = qwen.reshape(n, NUM_OUTPUTS)
    mean = flattened.mean(axis=0)
    _, singular, direction = np.linalg.svd(flattened - mean, full_matrices=False)
    rank = int(np.count_nonzero(singular > np.finfo(float).eps * max(flattened.shape) * singular.max()))
    basis = direction[:rank].T
    donor = np.random.default_rng(270371 + seed).permutation(len(train))
    values = {"ce": None, "qwen": qwen, "qwen-permuted": qwen[mapping], "muse": muse,
              "muse-permuted": muse[mapping] if muse is not None else None, "ecoc": scaled, "ecoc-hard": hard,
              "qwen-span": qwen, "qwen-complement": mean[None, :], "record-shuffle": qwen[train.labels[donor]]}
    values.update(dict.fromkeys(CLASS_HEAD_ARMS))
    values.update(dict.fromkeys(SOFT_CE, scaled))
    geometry = {}
    if set(enabled).intersection(CODE_CONTROLS):
        extra, geometry = code_controls(symbols, scaled, scale)
        values.update(extra)
    write_json(destination, {"class_names": names, "derangement_seed": 104729,
                             "donor_class_index_zero_based": mapping.tolist(), "ecoc_seed": 271828,
                             "ecoc_symbols_zero_based": symbols.tolist(), "ecoc_scale_about_uniform": scale,
                             "ecoc_hard_centered_energy": centered_energy(hard),
                             "ecoc_scaled_centered_energy": centered_energy(scaled),
                             "published_qwen_centered_energy": centered_energy(qwen),
                             "published_qwen_centered_rank": rank, "published_qwen_singular_values": singular.tolist(),
                             "record_shuffle_seed": 270371 + seed,
                             "record_shuffle_target_donor_image_ids": train.ids[donor].tolist()
                             if "record-shuffle" in enabled else None} | geometry)
    return values, torch.from_numpy(basis).float()


def masked_targets(args, qwen, names, train, heldout, qwen_path, destination):
    if sha256(args.mask_manifest) != MASKED_SCOPE_SHA:
        raise ValueError("Expected the frozen 53-class pause snapshot")
    scope = json.loads(args.mask_manifest.read_bytes())
    active = np.array(scope["eligible_index_zero_based"])
    if scope["author_class_order"] != names or len(active) != 53 or len(set(active)) != 53:
        raise ValueError("Unexpected eligible class labels in the frozen mask")
    if (sha256(args.muse) != scope["frozen_eligible_muse_sha256"] or
            sha256(qwen_path) != scope["published_qwen_sha256"]):
        raise ValueError("Teachers do not match the frozen partial signature input hashes")
    for split, dataset in (("train", train), ("heldout", heldout)):
        ids = dataset.ids[np.isin(dataset.labels, active)]
        expected = scope[split]
        fingerprint = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
        if (expected["all"] != len(dataset) or expected["eligible"] != len(ids) or
                expected["selected_ids_original_order_sha256_le_i64"] != fingerprint):
            raise ValueError(f"The same 53-class mask is required on the original {split} records")
    rng = np.random.default_rng(104729)
    cycle = [i for i in rng.permutation(len(names)).tolist() if i in active]
    induced = dict(zip(cycle, cycle[1:] + cycle[:1], strict=True))
    published = scope["donor_index_zero_based_by_original_class"]
    for label, entry in enumerate(published):
        if induced.get(label) != entry:
            raise ValueError("Expected the induced one-cycle derangement on eligible labels")
    order = [published[label] for label in active]
    label_to_row = np.full(len(names), -1, dtype=np.int64)
    label_to_row[active] = np.arange(len(active))
    native = read_signatures(args.muse, [names[label] for label in active])
    write_json(destination, {"class_names": names, "eligible_index_zero_based": active.tolist(),
                             "donor_class_index_zero_based": published,
                             "masked_scope_sha256": MASKED_SCOPE_SHA,
                             "masked_train_and_heldout": {s: scope[s] for s in ("train", "heldout")},
                             "mse_denominator": "all images in the minibatch times 250"})
    return ({"qwen": qwen[active], "qwen-permuted": qwen[order],
             "muse": native, "muse-permuted": native[label_to_row[order]]},
            torch.from_numpy(label_to_row).to(args.device),
            {"masked_scope_sha256": MASKED_SCOPE_SHA, "eligible_training_records": scope["train"]["eligible"],
             "mse_denominator": "all images in the minibatch times 250"})


def state_hash(*models):
    checksum = hashlib.sha256()
    for model in models:
        for name, tensor in model.state_dict().items():
            checksum.update(name.encode())
            check = tensor.detach().cpu().contiguous().numpy()
            checksum.update(check.tobytes())
    return checksum.hexdigest()


def write_checkpoint(path, state):
    temp = path.with_name(f"{path.name}.tmp")
    torch.save(state, temp)
    temp.replace(path)


def same_training_config(old, new):
    # A verified, pixel-identical cache may replace the original eval reader on resume.
    field = "validation_cache_sha256"
    return {k: v for k, v in old.items() if k != field} == {k: v for k, v in new.items() if k != field}


def metrics(log_probability, y, n):
    predicted = log_probability.argmax(axis=1)
    confusion = np.bincount(n * y + predicted, minlength=n * n).reshape(n, n)
    supported = confusion.sum(axis=1) > 0
    precision_denominator = confusion.sum(axis=0)[supported] + confusion.sum(axis=1)[supported]
    return {"top1": float((predicted == y).mean()),
            "macro_f1": float((2 * confusion.diagonal()[supported] / precision_denominator).mean()),
            "nll": float(-log_probability[np.arange(len(y)), y].astype(np.float64).mean())}


@torch.inference_mode()
def evaluate(model, loader, device, amp, n, path=None):
    model.eval()
    outputs = []
    for images, _, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images.to(device, non_blocking=True))
        outputs.append(logits.float().log_softmax(-1).cpu())
    logs = torch.cat(outputs).numpy()
    y = loader.dataset.labels
    if path is not None:
        np.savez_compressed(path, image_id=loader.dataset.ids, y_true=y, log_probability=logs)
    return metrics(logs, y, n)


def auxiliary_loss(arm, output, target, basis):
    residual = output - target
    if arm == "qwen-span":
        residual = (residual @ basis) @ basis.T
    if arm == "qwen-complement":
        residual = residual - (residual @ basis) @ basis.T
    return residual.square().mean()


def class_head_term(arm, logits, label, projected):
    onehot = F.one_hot(label, logits.size(1)).to(dtype=logits.dtype)
    if arm == "squentropy":
        return ((1 - onehot) * logits.square()).sum(dim=1).mean() / (logits.size(1) - 1)
    if arm == "prob-onehot":
        logits = logits.softmax(dim=1)
    if arm == "ce-independent-onehot":
        logits = projected[:, :logits.size(1)]
    return (logits - onehot).square().mean()


def train_epoch(model, auxiliary, optimizer, loader, device, amp, arm, targets, beta, basis, epoch, label_to_row):
    model.train()
    auxiliary.train()
    losses = torch.zeros(3, dtype=torch.float64, device=device)
    labeled_aux = torch.zeros((), dtype=torch.int64, device=device) if label_to_row is not None else None
    first, total, steps = {}, 0, 0
    for steps, (images, labels, records) in enumerate(loader, 1):
        if steps == 1:
            first["first_augmented_batch_sha256"] = hashlib.sha256(images.numpy().tobytes()).hexdigest()
        batch = images.to(device, non_blocking=True)
        label = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            features, logits = model(batch, is_feat=True)
            if arm == "ce-independent-onehot":
                k = logits.shape[1]
                predicted_aux = F.linear(features[-1], auxiliary.weight[:k], auxiliary.bias[:k])
            else:
                predicted_aux = auxiliary(features[-1]) if targets is not None else None
        ce = F.cross_entropy(logits.float(), label)
        mse = ce.new_zeros(())
        same_batch_mse = None
        if label_to_row is not None:
            rows = label_to_row[label]
            use = rows >= 0
            difference = predicted_aux.float()[use] - targets[rows[use]]
            mse = difference.square().sum() / (len(label) * NUM_OUTPUTS)
            labeled_aux += use.sum()
        elif arm in CLASS_HEAD_ARMS:
            mse = class_head_term(arm, logits.float(), label, predicted_aux.float() if predicted_aux is not None else None)
        elif targets is not None:
            if arm == "record-shuffle":
                target = targets[records.to(device)]
            elif arm in ("qwen-complement", "ecoc-constant"):
                target = targets
            else:
                target = targets[label]
            if arm in SOFT_CE:
                log_options = predicted_aux.float().reshape(-1, *SHAPE).log_softmax(dim=-1)
                mse = -(target.reshape(-1, *SHAPE) * log_options).sum(dim=-1).mean()
                if arm == "ecoc-soft-ce-gradmatch" and epoch == steps == 1:
                    same_batch_mse = (predicted_aux.float() - target).square().mean()
            else:
                mse = auxiliary_loss(arm, predicted_aux.float(), target, basis)
        if steps == 1 and epoch in (1, 120, 240):
            ce_grad, = torch.autograd.grad(ce, model.conv1.weight, retain_graph=True)
            first["ce"] = float(ce)
            first["ce_first_layer_grad_norm"] = float(ce_grad.norm())
            first["mse"] = float(mse)
            if targets is not None or arm in CLASS_HEAD_ARMS:
                aux_grad, = torch.autograd.grad(mse, model.conv1.weight, retain_graph=True)
                if same_batch_mse is not None:
                    reference, = torch.autograd.grad(same_batch_mse, model.conv1.weight, retain_graph=True)
                    reference_norm, candidate_norm = float(reference.norm()), float(aux_grad.norm())
                    if min(reference_norm, candidate_norm) <= 0.:
                        raise ValueError("Cannot calibrate a zero first-batch shared-layer gradient")
                    first["paired_same_output_mse"] = float(same_batch_mse)
                    first["paired_same_output_mse_gradient_norm"] = reference_norm
                    first["precalibration_target_weight"] = beta
                    beta *= reference_norm / candidate_norm
                first["aux_first_layer_grad_norm"] = float(aux_grad.norm())
                first["weighted_aux_to_ce_first_layer_grad_norm"] = beta * float(aux_grad.norm() / ce_grad.norm())
                first["gradient_cosine"] = float(F.cosine_similarity(ce_grad.flatten(), aux_grad.flatten(), dim=0))
        objective = mse if arm in PURE_SQUARE else ce + beta * mse
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            losses += torch.stack((ce * len(label), mse * len(label), (logits.argmax(dim=1) == label).sum()))
        total += len(label)
    ce_mean, mse_mean, accuracy = (losses / total).tolist()
    first["auxiliary_weight_effective"] = beta
    measurements = {"updates": steps, "records": total, "train_top1": accuracy, "first_batch": first,
                    **describe_losses(ce_mean, mse_mean, beta)}
    if arm in (*CLASS_HEAD_ARMS, *SOFT_CE):
        measurements.update(auxiliary_term="soft_ce" if arm in SOFT_CE else "squared_error",
                            ce_was_optimized=arm not in PURE_SQUARE, effective_auxiliary_weight=beta,
                            train_objective=mse_mean if arm in PURE_SQUARE else ce_mean + beta * mse_mean)
    if labeled_aux is not None:
        measurements.update(auxiliary_records=int(labeled_aux), auxiliary_denominator_coordinates=total * NUM_OUTPUTS)
    return measurements


def arm_training(args, arm, config, source, qwen_values, basis, train, heldout, n, beta, label_to_row):
    directory = args.out / args.dataset / f"seed-{args.seed}" / arm
    directory.mkdir(parents=True, exist_ok=True)
    config_path, result_path = directory / "config.json", directory / "results.json"
    target_values = qwen_values[arm]
    if target_values is None and arm not in ("ce", *CLASS_HEAD_ARMS):
        raise ValueError(f"Supply the complete Muse signature to train {arm}")
    digest = hashlib.sha256(target_values.tobytes()).hexdigest() if target_values is not None else None
    config = config | {"arm": arm, "target_float64_sha256": digest,
                       "muse_signature_sha256": sha256(args.muse) if arm.startswith("muse") else None}
    if arm in (*CLASS_HEAD_ARMS, *SOFT_CE):
        config["auxiliary_weight"] = 1. if arm in (*PURE_SQUARE, "squentropy") else beta
        config["uses_primary_ce"] = arm not in PURE_SQUARE
        config["gradient_calibrated"] = arm == "ecoc-soft-ce-gradmatch"
    if arm == "ce-independent-onehot":
        config["active_independent_head_initialization"] = "copy of untouched classification head, independent storage"
    if config_path.exists() and not same_training_config(json.loads(config_path.read_bytes()), config):
        raise ValueError(f"Run settings do not match prior results in {directory}")
    write_json(config_path, config)
    if result_path.exists():
        print(f"Reusing completed {args.dataset} seed={args.seed} {arm}", flush=True)
        return
    device = torch.device(args.device)
    amp = args.precision == "bfloat16"
    torch.manual_seed(args.seed)
    model = author_model(source, n).to(device)
    auxiliary = nn.Linear(512, NUM_OUTPUTS).to(device)
    initial = state_hash(model, auxiliary)
    method_initial = initial
    if arm == "ce-independent-onehot":
        with torch.no_grad():
            auxiliary.weight[:n].copy_(model.fc.weight)
            auxiliary.bias[:n].copy_(model.fc.bias)
        method_initial = state_hash(model, auxiliary)
    targets = None if target_values is None else torch.as_tensor(target_values.reshape(len(target_values), -1),
                                                                device=device, dtype=torch.float32)
    optimizer = torch.optim.SGD(chain(model.parameters(), auxiliary.parameters()), lr=.01, momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(MILESTONES), gamma=.1)
    sampler = PairedOrder(train, args.seed)
    options = {"batch_size": BATCH_SIZE, "num_workers": args.workers, "persistent_workers": args.workers > 0,
               "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train, sampler=sampler, **options)
    test_loader = DataLoader(heldout, **options)
    latest = directory / "last.pt"
    best_path = directory / "author-best.pt"
    progress = []
    best = {"epoch": None, "validation_nll": float("inf")}
    if latest.exists():
        recovered = torch.load(latest, weights_only=True, map_location=device)
        if recovered["initial_state_sha256"] != initial:
            raise ValueError("Initial model fingerprint changed during resume")
        model.load_state_dict(recovered["student"])
        auxiliary.load_state_dict(recovered["auxiliary"])
        optimizer.load_state_dict(recovered["optimizer"])
        scheduler.load_state_dict(recovered["scheduler"])
        progress, best = recovered["history"], recovered["author_best"]
    started = time.perf_counter()
    target_epoch = min(args.epochs, len(progress) + args.max_new_epochs) if args.max_new_epochs else args.epochs
    effective_beta = config["auxiliary_weight"]
    if arm == "ecoc-soft-ce-gradmatch" and progress:
        effective_beta = progress[0]["effective_auxiliary_weight"]
    for epoch in range(len(progress) + 1, target_epoch + 1):
        sampler.epoch = epoch
        learning_rate = optimizer.param_groups[0]["lr"]
        values = train_epoch(model, auxiliary, optimizer, train_loader, device, amp, arm, targets, effective_beta, basis, epoch,
                             label_to_row)
        effective_beta = values.get("effective_auxiliary_weight", effective_beta)
        scheduler.step()
        values.update(epoch=epoch, lr=learning_rate,
                      ordered_index_and_augmentation_seed_sha256=hashlib.sha256(sampler.plan().tobytes()).hexdigest())
        artifact = {"epoch": epoch, "student": model.state_dict(), "auxiliary": auxiliary.state_dict(),
                    "initial_state_sha256": initial, "method_initial_state_sha256": method_initial}
        if args.capture_epochs and epoch in args.capture_epochs:
            write_checkpoint(directory / f"epoch-{epoch:03d}.pt", artifact)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            validation = evaluate(model, test_loader, device, amp, n)
            values["author_eval_split"] = validation
            if args.validation_cache:
                values["validation_cache_sha256"] = config["validation_cache_sha256"]
            if epoch % 5 == 0 and validation["nll"] < best["validation_nll"]:
                best = {"epoch": epoch, "validation_nll": validation["nll"]}
                write_checkpoint(best_path, artifact)
        progress.append(values)
        write_checkpoint(latest, artifact | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                             "history": progress, "author_best": best})
        write_json(directory / "history.json", progress)
        if "author_eval_split" in values:
            print(f"{args.dataset} seed={args.seed} {arm} epoch={epoch}: "
                  f"top1={values['author_eval_split']['top1']:.5f} CE={values['ce']:.5f}", flush=True)
    if len(progress) != args.epochs:
        return
    end = {"epoch": args.epochs, "student": model.state_dict(), "auxiliary": auxiliary.state_dict(),
           "initial_state_sha256": initial, "method_initial_state_sha256": method_initial}
    final_path = directory / "final.pt"
    write_checkpoint(final_path, end)
    final_metrics = evaluate(model, test_loader, device, False, n, directory / "final_predictions.npz")
    selected_metrics = None
    if best_path.is_file():
        selected = torch.load(best_path, weights_only=True, map_location=device)
        if selected["epoch"] != best["epoch"]:
            raise ValueError("Best checkpoint differs from archived evaluation")
        model.load_state_dict(selected["student"])
        selected_metrics = evaluate(model, test_loader, device, False, n, directory / "author_best_predictions.npz")
    output_files = ["final.pt", "final_predictions.npz"]
    if selected_metrics is not None:
        output_files += ["author-best.pt", "author_best_predictions.npz"]
    write_json(result_path, {"arm": arm, "initial_state_sha256": initial,
                             "method_initial_state_sha256": method_initial, "epochs": len(progress),
                             "optimizer_steps": sum(row["updates"] for row in progress),
                             "author_selection": best if best["epoch"] is not None else None,
                             "fixed_final_fp32": final_metrics, "author_selected_fp32": selected_metrics,
                             "artifact_sha256": {name: sha256(directory / name) for name in output_files},
                             "seconds_this_invocation": time.perf_counter() - started})


def run(args):
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    n, _, _, beta = DATASETS[args.dataset]
    files = {"student": args.author_source / "lavid/models/resnet.py",
             "class_names": args.author_source / f"lavid/data/classes/{args.dataset}_classes.json",
             "qwen": args.author_source / f"questions/{args.dataset}/logits/qwen2.5-7b_chat_50-5.json"}
    names = json.loads(files["class_names"].read_bytes())
    if len(names) != n:
        raise ValueError("Unexpected author class list")
    qwen = read_signatures(files["qwen"], names)
    train = VisualDataset(args.images, args.dataset, True)
    heldout = VisualDataset(args.images, args.dataset, False)
    if set(train.ids).intersection(heldout.ids):
        raise ValueError("Overlap between author training and held-out split")
    parents = args.out / args.dataset / f"seed-{args.seed}"
    label_to_row = None
    masked = {}
    if args.mask_manifest:
        values, label_to_row, masked = masked_targets(args, qwen, names, train, heldout, files["qwen"],
                                                    parents / "targets.json")
        basis = None
    else:
        muse = read_signatures(args.muse, names) if args.muse else None
        values, basis = make_targets(qwen, muse, names, train, args.seed, args.arms, parents / "targets.json")
    cohort = {"training_ids": train.ids.tolist(), "training_labels": train.labels.tolist(),
              "heldout_ids": heldout.ids.tolist(), "heldout_labels": heldout.labels.tolist()}
    write_json(parents / "cohort.json", cohort)
    config = {"author_revision": AUTHOR_REVISION, "dataset": args.dataset, "seed": args.seed,
              "epochs": args.epochs, "batch_size": BATCH_SIZE, "precision": args.precision, "workers": args.workers,
              "optimizer": {"lr": .01, "momentum": .9, "weight_decay": .0005, "milestones": list(MILESTONES), "gamma": .1},
              "auxiliary_shape": list(SHAPE), "auxiliary_weight": beta, "input_sha256": {
                  **{name: sha256(path) for name, path in files.items()},
                  **{name: sha256(args.images / name) for name in RAW_SOURCES[args.dataset]}},
              "software": {name: version(name) for name in ("torch", "torchvision", "pillow", "scipy", "numpy")},
              "cohort_sha256": sha256(parents / "cohort.json")} | masked
    if args.validation_cache:
        from .visual_validation_cache import CachedVisualEval

        heldout = CachedVisualEval(args.validation_cache, heldout, dataset=args.dataset)
        if "scores_identical" not in heldout.metadata:
            raise ValueError("Only use the evaluation cache after full-checkpoint output equivalence has been verified")
        config["validation_cache_sha256"] = heldout.metadata["sha256"]
    if args.capture_epochs:
        config["capture_epochs"] = args.capture_epochs
    basis = basis.to(args.device) if basis is not None else None
    for arm in args.arms:
        arm_training(args, arm, config, files["student"], values, basis, train, heldout, n, beta, label_to_row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for parameter in ("author-source", "images", "out"):
        parser.add_argument(f"--{parameter}", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="flowers")
    parser.add_argument("--arms", choices=ARMS, nargs="+", required=True)
    parser.add_argument("--muse", type=Path)
    parser.add_argument("--mask-manifest", type=Path, help="Frozen eligible label subset for an independent paired experiment")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--max-new-epochs", type=int)
    parser.add_argument("--capture-epochs", type=int, nargs="+")
    parser.add_argument("--validation-cache", type=Path, help="Verified read-only author evaluation uint8 cache")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 31 or not 1 <= args.epochs < 2 ** 16 or args.workers < 0 or args.cpu_threads < 1:
        parser.error("Invalid seed, epoch or worker count")
    if args.max_new_epochs is not None and args.max_new_epochs < 1:
        parser.error("Resume limit must be positive")
    if args.capture_epochs and (any(not 1 <= epoch <= args.epochs for epoch in args.capture_epochs)
                                or len(set(args.capture_epochs)) != len(args.capture_epochs)):
        parser.error("Captured epochs must be unique and within the training schedule")
    if args.mask_manifest and (args.dataset != "flowers" or args.seed != 17 or not args.muse or
                               set(args.arms) - {"qwen", "qwen-permuted", "muse", "muse-permuted"}):
        parser.error("The paused-signature design uses Flowers seed 17 and the four paired teacher arms")
    run(args)


if __name__ == "__main__":
    main()
