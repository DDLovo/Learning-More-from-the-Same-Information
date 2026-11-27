"""Build fixed, class-index-only auxiliary codebooks for four-way ECG training."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import CLASSES


SHAPE = (4, 12, 5)
FIVE_WAY_SEED = 271828
BINARY_SEED = 314159
ROW_SHUFFLE_SEED = 161803


def distances(words):
    return np.count_nonzero(words[:, None] != words[None, :], axis=-1)


def shuffled_training_records(indices, seed):
    """Draw one fixed random cycle over the selected training records."""
    if len(indices) < 2:
        raise ValueError("The ECOC row shuffle requires at least two training records")
    generator = np.random.default_rng(np.random.SeedSequence([ROW_SHUFFLE_SEED, seed]))
    cycle = generator.permutation(indices)
    return cycle, np.roll(cycle, -1)


def build_codebooks():
    generator = np.random.default_rng(FIVE_WAY_SEED)
    symbols = np.stack([generator.permutation(5)[:4] for _ in range(12)], axis=1)
    five_way = np.eye(5, dtype=np.uint8)[symbols]
    binary_ids = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    parity = np.column_stack((binary_ids, binary_ids[:, 0] ^ binary_ids[:, 1]))
    complemented = np.concatenate([np.column_stack((np.repeat(col[:, None], 10, axis=1),
                                                     np.repeat(1 - col[:, None], 10, axis=1)))
                                  for col in parity.T], axis=1)
    order = np.random.default_rng(BINARY_SEED).permutation(60)
    binary = complemented[:, order].reshape(SHAPE)
    specifications = {
        "five_way": (FIVE_WAY_SEED, five_way, symbols, {"symbol_words": symbols.tolist()}),
        "binary_simplex": (BINARY_SEED, binary, binary.reshape(4, 60), {"column_order": order.tolist()}),
    }
    for name, (seed, targets, words, details) in specifications.items():
        code_distances = distances(words)
        yield name, {"codebook_type": name, "generation_seed": seed,
                     "generator": "numpy.random.default_rng (PCG64)", "class_codes": list(CLASSES),
                     "targets": targets.tolist(), "distance_matrix": code_distances.tolist(),
                     "minimum_distance": int(code_distances[np.triu_indices(4, 1)].min()),
                     "row_weights": targets.sum(axis=(1, 2)).tolist(), **details}


def load_codebook(path):
    codebook = json.loads(path.read_text(encoding="utf-8-sig"))
    targets = np.asarray(codebook["targets"])
    if (tuple(codebook["class_codes"]) != tuple(CLASSES) or targets.ndim != 3 or targets.shape[0] != len(CLASSES)
            or not targets.shape[1] or targets.shape[2] != 5 or not np.isin(targets, (0, 1)).all()):
        raise ValueError("Codebook must contain a 4 × questions × 5 binary tensor in ECG class order")
    variant = codebook["codebook_type"]
    if variant == "five_way":
        if not np.all(targets.sum(-1) == 1):
            raise ValueError("Each five-way code symbol must be one-hot")
        words, minimum = targets.argmax(-1), targets.shape[1]
    elif variant == "binary_simplex":
        if targets.shape != SHAPE or not np.all(targets.sum(axis=(1, 2)) == 30) or not np.all(targets.sum(axis=0) == 2):
            raise ValueError("Binary simplex code must have 30 ones in each word and two ones in each column")
        words, minimum = targets.reshape(4, 60), 40
    else:
        raise ValueError(f"Unknown codebook type {variant!r}")
    actual_distances = distances(words)
    if not (np.all(actual_distances[np.triu_indices(4, 1)] == minimum)
            and np.array_equal(actual_distances, codebook["distance_matrix"])):
        raise ValueError(f"Unexpected {variant} codeword distances")
    return targets.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, codebook in build_codebooks():
        destination = args.out / f"{name}.json"
        destination.write_text(json.dumps(codebook, indent=2) + "\n", encoding="utf-8", newline="\n")
        load_codebook(destination)


if __name__ == "__main__":
    main()
