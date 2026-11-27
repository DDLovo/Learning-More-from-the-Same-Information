"""Apply frozen PTB-XL RR auxiliary heads to the LUDB manual-reference cohort."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import torch
from torch.utils.data import TensorDataset

from . import LEADS, SAMPLE_RATE
from .evaluate_rr_aux import METHODS, PROJECTION, SHAPE, infer
from .rr_questions import FEATURES
from .teacher import write_json
from .train import ECGStudent, make_loader, sha256


SEEDS = (17, 29, 43)
REFERENCE_FIELDS = ("rr_mean_ms", "rr_sd_ms")
BOOTSTRAP_SEED = 20261126
BOOTSTRAP_DRAWS = 5000


def bootstrap_rank_correlation(target_rank, prediction):
    pred_rank = rankdata(prediction, axis=1)
    pred_rank -= pred_rank.mean(axis=1, keepdims=True)
    dot = np.einsum("bi,bi->b", target_rank, pred_rank)
    denominator = np.linalg.norm(target_rank, axis=1) * np.linalg.norm(pred_rank, axis=1)
    return dot / denominator


def paired_analysis(raw_scores, reference, scopes):
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {}
    for name, mask in scopes.items():
        n = int(mask.sum())
        draw = generator.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
        result = {"n_patients": n, "scores": {}}
        for concept, field in enumerate(FEATURES):
            expected = reference[mask, concept]
            rank_reference = rankdata(expected[draw], axis=1)
            rank_reference -= rank_reference.mean(axis=1, keepdims=True)
            for fraction, arms in raw_scores.items():
                observed = {method: [estimate[mask, concept] for estimate in estimates] for method, estimates in arms.items()}
                seed_scores = {method: [float(spearmanr(expected, estimate).statistic) for estimate in estimates]
                               for method, estimates in observed.items()}
                leave = {method: np.asarray([bootstrap_rank_correlation(rank_reference, estimate[draw]) for estimate in estimates])
                         for method, estimates in observed.items()}
                diffs = np.asarray(seed_scores["correct"]) - seed_scores["permuted"]
                bootstrap_paired = (leave["correct"] - leave["permuted"]).mean(axis=0)
                if not np.isfinite(bootstrap_paired).all() or not np.isfinite(diffs).all():
                    raise ValueError(f"Deficient bootstrap ranks in {name}/{field}")
                result["scores"].setdefault(fraction, {})[field] = {
                    "per_seed": seed_scores,
                    "mean": {method: float(np.mean(values)) for method, values in seed_scores.items()},
                    "paired_seed_deltas": diffs.tolist(),
                    "paired_mean_delta": float(diffs.mean()),
                    "paired_percentile_ci_95": np.quantile(bootstrap_paired, [0.025, 0.975]).tolist(),
                }
        summary[name] = result
    return summary


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with np.load(args.waveforms, allow_pickle=False) as data:
        ids = data["record_id"].astype(str)
        signals = data["waveforms"]
        if tuple(str(s).upper() for s in data["leads"]) != tuple(s.upper() for s in LEADS) or data["sample_rate"].item() != SAMPLE_RATE:
            raise ValueError("LUDB lead order or sampling rate differs from training")
    reference = pd.read_csv(args.manual, dtype={"id": str}).set_index("id", verify_integrity=True)
    assert len(ids) == 200 and len(set(ids)) == 200 and set(reference.index) == set(ids)
    assert signals.shape == (len(ids), len(LEADS), 1000) and signals.dtype == np.float32 and np.isfinite(signals).all()
    reference = reference.loc[ids]
    truth = reference[list(REFERENCE_FIELDS)].to_numpy(dtype=np.float64)
    valid = reference["valid"].astype(str).str.lower().isin(("true", "1")).to_numpy()
    sinus = reference["sinus_subgroup"].astype(str).str.lower().isin(("true", "1")).to_numpy()
    assert valid.sum() == 200 and (sinus & valid).sum() == 129 and np.isfinite(truth[valid]).all()
    data = TensorDataset(torch.from_numpy(signals), torch.zeros(len(ids)))
    loader = make_loader(data, np.arange(len(ids)), 256, device)
    model = ECGStudent(SHAPE).to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs, provenance = {}, {}
    for run_path in args.results:
        config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
        batch_size = config["batch_size"] if "batch_size" in config else (
            config["labeled_batch_size"] + config["unlabeled_batch_size"]
        )
        if config["seeds"] != args.seeds:
            raise ValueError(f"The requested seeds differ from the RR training run at {run_path}")
        amp = config["precision"] == "bfloat16"
        for value in config.get("fractions", [config.get("fraction")]):
            fraction = str(value)
            outputs[fraction] = {}
            for method, checkpoint_stem in zip(METHODS, args.checkpoint_methods):
                estimates = []
                for seed in args.seeds:
                    path = run_path / f"fraction-{fraction}_seed-{seed}"
                    run_meta = json.loads((path / f"{checkpoint_stem}.json").read_text(encoding="utf-8"))
                    checkpoint_path = path / f"{checkpoint_stem}.pt"
                    fingerprint = sha256(checkpoint_path)
                    if "checkpoint_sha256" in run_meta:
                        assert fingerprint == run_meta["checkpoint_sha256"]
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    assert tuple(checkpoint["leads"]) == tuple(LEADS) and checkpoint["sample_rate"] == SAMPLE_RATE
                    model.load_state_dict(checkpoint["model_state"])
                    aux, _ = infer(model, loader, checkpoint["normalization"], device, amp)
                    guess = aux @ PROJECTION
                    filename = f"fraction-{fraction}_seed-{seed}_{method}.npz"
                    np.savez_compressed(args.out / filename, record_id=ids, manual_rr_ms=truth, valid=valid,
                                        sinus=sinus, ordinal_score=guess, rr_raw_aux=aux, projection=PROJECTION,
                                        fields=np.asarray(REFERENCE_FIELDS))
                    provenance[filename] = {"checkpoint_sha256": fingerprint, "prediction_sha256": sha256(args.out / filename),
                                            "checkpoint_stem": checkpoint_stem}
                    estimates.append(guess)
                outputs[fraction][method] = estimates
    scopes = {"all_patients": valid, "strict_sinus": valid & sinus}
    write_json(args.out / "external_validation.json",
               {"waveforms_sha256": sha256(args.waveforms), "manual_reference_sha256": sha256(args.manual),
                "fixed_projection": PROJECTION.tolist(), "seed_order": args.seeds,
                "patient_bootstrap_draws": BOOTSTRAP_DRAWS, "patient_bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_unit": "LUDB record (one per patient); shared resample across all methods and seeds",
                "outputs": provenance, "results": paired_analysis(outputs, truth, scopes)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveforms", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs=3, default=list(SEEDS))
    parser.add_argument("--checkpoint-methods", nargs=2, default=METHODS, metavar=("TRUE", "SHUFFLED"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
