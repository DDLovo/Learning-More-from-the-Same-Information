"""Score selected ECOC checkpoints with the rhythm head and the fixed nearest-code decoder."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import CLASSES
from .ecoc import load_codebook
from .train import ECGStudent, cohort_splits, load_data, make_loader, score_predictions


SEEDS = (17, 29, 43, 71, 101, 137)
CODEBOOKS = ("five_way", "binary_simplex")


@torch.inference_mode()
def infer(model, loader, normalization, device, amp):
    model.eval()
    probabilities, auxiliary = [], []
    mean, std = (tensor.to(device) for tensor in normalization)
    for signals, _ in loader:
        signals = (signals.to(device, non_blocking=True) - mean) / std
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, outputs = model(signals, with_auxiliary=True)
        probabilities.append(logits.float().softmax(dim=-1).cpu().numpy())
        auxiliary.append(outputs.float().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(auxiliary)


def distances_to_codes(prediction, codebook, variant):
    if variant == "five_way":
        words, codes = prediction.argmax(axis=-1), codebook.argmax(axis=-1)
    else:
        words, codes = prediction.reshape(-1, 60) >= 0.5, codebook.reshape(4, 60).astype(bool)
    return np.count_nonzero(words[:, None] != codes[None, :], axis=-1)


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset, ids, patients, folds = load_data(args.data)
    labels = dataset.tensors[1].numpy()
    splits, _ = cohort_splits(folds, patients, labels)
    validation = splits["validation"]
    val_labels = labels[validation]
    loader = make_loader(dataset, validation, 256, device)
    model = ECGStudent((12, 5)).to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    output = {}
    for variant in CODEBOOKS:
        codebook = load_codebook(args.codebooks / f"{variant}.json")
        output[variant] = {}
        for seed in SEEDS:
            run_dir = args.results / variant / f"fraction-0.1_seed-{seed}"
            recorded = json.loads((run_dir / "lavid.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(run_dir / "lavid.pt", weights_only=True, map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            probabilities, auxiliary = infer(model, loader, checkpoint["normalization"], device, args.bfloat16)
            classifier = score_predictions(val_labels, probabilities)
            np.testing.assert_allclose(classifier["macro_f1"], recorded["validation"]["macro_f1"], atol=1e-12)
            np.testing.assert_array_equal(classifier["confusion_matrix"], recorded["validation"]["confusion_matrix"])
            code_distances = distances_to_codes(auxiliary, codebook, variant)
            decoded = code_distances.argmin(axis=-1)
            code_scores = score_predictions(val_labels, np.eye(len(CLASSES))[decoded])
            output[variant][str(seed)] = {
                "classifier_macro_f1": classifier["macro_f1"],
                "nearest_code_macro_f1": code_scores["macro_f1"],
                "nearest_code_accuracy": code_scores["accuracy"],
                "nearest_code_confusion_matrix": code_scores["confusion_matrix"],
                "distance_ties": int(np.sum(np.sum(code_distances == code_distances.min(axis=-1, keepdims=True), axis=-1) > 1)),
                "classifier_and_nearest_code_agreement": float(np.mean(decoded == probabilities.argmax(-1))),
            }
            np.savez_compressed(args.out / f"{variant}_seed-{seed}_fold9.npz",
                                ecg_id=ids[validation], y_true=val_labels, class_codes=np.asarray(tuple(CLASSES)),
                                probabilities=probabilities, code_distances=code_distances, nearest_code=decoded)
            print(variant, seed, output[variant][str(seed)]["nearest_code_macro_f1"], flush=True)
    (args.out / "nearest_code_validation.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--codebooks", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bfloat16", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
