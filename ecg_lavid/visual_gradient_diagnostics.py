"""Measure fixed-anchor analytic pooled gradients and auxiliary-head spectra."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
import torch

from .teacher import write_json
from .train import sha256
from .train_visual_lavid import DATASETS, read_signatures
from .visual_waterbirds import DATASET as WATERBIRDS, WaterbirdsDataset


METHODS = ("ce", "qwen", "ecoc", "ecoc-constant", "ecoc-repeat", "ecoc-simplex",
           "ecoc-soft-ce", "ecoc-soft-ce-gradmatch")
CLASSIFIERS = {"resnet18": ("fc", "encoder.fc"), "mobilenetv2": ("encoder.classifier.1",),
               "vit_small_patch16": ("encoder.head",)}


def expected_targets(method, codebook, qwen):
    if method == "qwen":
        return qwen
    code = np.asarray(codebook["ecoc_symbols_zero_based"])
    scale = codebook["ecoc_scale_about_uniform"]
    if method == "ecoc-simplex":
        return np.asarray(codebook["simplex_targets"], dtype=np.float64)
    if method == "ecoc-repeat":
        code = np.asarray(codebook["repeat_symbols_zero_based"])
    values = .2 + scale * (np.eye(5, dtype=np.float64)[code] - .2)
    return values.mean(axis=0, keepdims=True) if method == "ecoc-constant" else values


def stats(values):
    if not len(values):
        return None
    return dict(zip(("minimum", "median", "mean", "maximum"),
                    map(float, (np.min(values), np.median(values), np.mean(values), np.max(values)))))


def spectrum(weight):
    singular = np.linalg.svd(weight, compute_uv=False)
    energy = singular ** 2
    positive = energy[energy > 0] / energy.sum()
    return {"wtw_trace": float(energy.sum()),
            "wtw_energy_effective_rank": float(np.exp(-np.dot(positive, np.log(positive)))),
            "wtw_top3_energy_share": float(energy[:3].sum() / energy.sum()), "singular_values": singular.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, WATERBIRDS), required=True)
    parser.add_argument("--architecture", choices=CLASSIFIERS, default="resnet18")
    parser.add_argument("--method", choices=METHODS, required=True)
    for key in ("features", "checkpoint", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--target-book", type=Path)
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--waterbirds-images", type=Path, help="Official image root for Waterbirds species IDs")
    parser.add_argument("--weight", type=float, help="Training-fixed coefficient of an already calibrated soft CE run")
    args = parser.parse_args()
    if (args.method != "ce" and not args.target_book) or (args.method == "qwen" and not args.qwen):
        parser.error("Supply the actual source class codebook and Qwen file when required")
    if args.method == "ecoc-soft-ce-gradmatch" and not args.weight:
        parser.error("Supply the training-frozen effective coefficient for G*")
    if args.dataset == WATERBIRDS and args.method not in ("ce", "ecoc-constant") and not args.waterbirds_images:
        parser.error("Waterbirds class-conditioned auxiliary diagnostics need --waterbirds-images")
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    features_meta = json.loads(args.features.with_suffix(".json").read_bytes())
    if features_meta["archive_sha256"] != sha256(args.features) or features_meta["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Frozen embeddings refer to another model or changed archive bytes")
    with np.load(args.features, allow_pickle=False) as x:
        keys, labels, features = (x[key] for key in ("train_ids", "train_y", "train_features"))
    anchor = sorted(range(len(keys)), key=lambda i: hashlib.sha256(
        f"visual-pooled-gradient-anchor-v1|{args.dataset}|{int(keys[i])}".encode()).digest())[:128]
    image_ids, y, h = keys[anchor], labels[anchor], features[anchor].astype(np.float64)
    classifier, auxiliary = state["student"], state["auxiliary"]
    layer = next((prefix for prefix in CLASSIFIERS[args.architecture] if prefix + ".weight" in classifier), None)
    if layer is None:
        raise ValueError("The checkpoint does not contain the selected architecture's classification head")
    classifier_matrix = classifier[layer + ".weight"].numpy().astype(np.float64)
    classifier_bias = classifier[layer + ".bias"].numpy().astype(np.float64)
    delta_ce = softmax(h @ classifier_matrix.T + classifier_bias, axis=1)
    delta_ce[np.arange(len(y)), y] -= 1
    ce_gradient = delta_ce @ classifier_matrix
    w = auxiliary["weight"].numpy().astype(np.float64)
    b = auxiliary["bias"].numpy().astype(np.float64)
    result = {"dataset": args.dataset, "architecture": args.architecture, "method": args.method, "checkpoint_epoch": state["epoch"],
              "checkpoint_sha256": features_meta["checkpoint_sha256"], "input_anchor_ids": image_ids.tolist(),
              "anchor_labels": y.tolist(), "numerics": "analytic float64 gradients at frozen FP32 eval-mode representation",
              "anchored_feature_l2": stats(np.linalg.norm(h, axis=-1)),
              "anchored_ce_gradient_l2": stats(np.linalg.norm(ce_gradient, axis=-1)),
              "auxiliary_head": spectrum(w)}
    if args.method != "ce":
        book = json.loads(args.target_book.read_bytes())
        qwen = read_signatures(args.qwen, book["class_names"]) if args.qwen else None
        expected = expected_targets(args.method, book, qwen)
        if len(expected) != 1:
            code_labels = y
            if args.dataset == WATERBIRDS:
                train = WaterbirdsDataset(args.waterbirds_images, "train")
                by_id = dict(zip(train.ids, zip(train.labels, train.auxiliary_labels), strict=True))
                primary, code_labels = np.asarray([by_id[img] for img in image_ids]).T
                if not np.array_equal(primary, y):
                    raise ValueError("Frozen representations differ from the official Waterbirds training labels")
                result["anchor_species_ids"] = code_labels.tolist()
            expected = expected[code_labels]
        z = (h @ w.T + b).reshape(len(y), 50, 5)
        logit_gradient = ((softmax(z, axis=-1) - expected) / 50 if args.method.startswith("ecoc-soft-ce")
                          else 2 * (z - expected) / 250)
        heads = w.reshape(50, 5, -1)
        individual = np.einsum("bqo,qoh->bqh", logit_gradient, heads, optimize=True)
        auxiliary_gradient = individual.sum(axis=1)
        norm_blocks = np.linalg.norm(individual, axis=-1)
        keep = norm_blocks > 1e-15
        normalized = np.divide(individual, norm_blocks[..., None], out=np.zeros_like(individual),
                               where=keep[..., None])
        squared_cosines = np.square(normalized @ normalized.transpose(0, 2, 1))
        eligible = keep[:, :, None] & keep[:, None, :] & ~np.eye(50, dtype=bool)
        redundancy = np.divide((squared_cosines * eligible).sum((1, 2)), eligible.sum((1, 2)),
                               out=np.full(len(y), np.nan), where=eligible.sum((1, 2)) > 0)
        beta = args.weight if args.weight is not None else (40. if args.dataset == WATERBIRDS else DATASETS[args.dataset][3])
        norm_ce = np.linalg.norm(ce_gradient, axis=-1)
        norm_aux = np.linalg.norm(auxiliary_gradient, axis=-1)
        choice = (norm_ce > 1e-15) & (norm_aux > 1e-15)
        cosine = (auxiliary_gradient[choice] * ce_gradient[choice]).sum(axis=-1) / (norm_ce[choice] * norm_aux[choice])
        result["measured_multi_question_gradients"] = {
            "target_book_sha256": sha256(args.target_book), "term_weight": beta,
            "mean_each_of_50_block_gradient_norms": norm_blocks.mean(axis=0).tolist(),
            "per_image_squared_cosine_redundancy": [float(x) if np.isfinite(x) else None for x in redundancy],
            "squared_cosine_redundancy": stats(redundancy[np.isfinite(redundancy)]),
            "weighted_auxiliary_over_ce_norm_ratio": stats(beta * norm_aux[norm_ce > 1e-15] / norm_ce[norm_ce > 1e-15]),
            "auxiliary_to_ce_gradient_cosine": stats(cosine),
        }
    write_json(args.out, result)


if __name__ == "__main__":
    main()
