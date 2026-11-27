"""Summarize area-normalized foreground attribution with paired image bootstrap intervals."""
import argparse
import csv
from pathlib import Path

import numpy as np

COHORT_SIZE = {"flowers": 5978, "cub200": 5794}
VIEWS = ("clean", "foreground-only", "background-replaced", "background-only")
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261127


def assert_same(a, h, keys):
    for key in keys:
        np.testing.assert_array_equal(a[key], h[key], err_msg=key)


def bootstrap_bounds(delta):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_N, dtype=np.float64)
    n = len(delta)
    for start in range(0, BOOTSTRAP_N, 128):
        finish = min(start + 128, BOOTSTRAP_N)
        draw = rng.integers(n, size=(finish - start, n), dtype=np.int32)
        means[start:finish] = delta[draw].mean(axis=1)
    return np.quantile(means, [.025, .975])


def quantity_rows(dataset, architecture, a, h, expected_views):
    assert_same(a, h, ('image_id', 'donor_image_id', 'true_label', 'views', 'area_fraction', 'area_pixels'))
    n = len(a['image_id'])
    assert n == COHORT_SIZE[dataset] and len(np.unique(a['image_id'])) == n
    assert a['views'].tolist() == list(expected_views)
    area = a['area_fraction'].astype(np.float64)
    assert area.shape == (n,) and np.all((area > 0) & (area < 1))
    np.testing.assert_allclose(area, a['area_pixels'] / (224 * 224), atol=1e-7, rtol=0)
    for data in (a, h):
        q, zero, valid = (data[key] for key in ('q_fg', 'is_zero', 'is_valid'))
        assert q.shape == zero.shape == valid.shape == (n, len(expected_views))
        assert np.array_equal(np.isfinite(q), valid) and not np.any(valid & zero)
        assert np.all(q[valid] >= 0)
        fraction = data['foreground_mass_fraction']
        assert np.all((fraction[valid] >= 0) & (fraction[valid] <= 1))
        np.testing.assert_allclose(fraction[valid], data['foreground_mass'][valid] / data['cam_mass'][valid], atol=1e-7, rtol=0)
        np.testing.assert_allclose((q * area[:, None])[valid], fraction[valid], atol=1e-7, rtol=0)
        constants = data['is_constant_positive'] & valid
        np.testing.assert_allclose(q[constants], 1.0, atol=1e-7, rtol=0)
    output = []
    for column, view in enumerate(expected_views):
        z_a, z_h = a['is_zero'][:, column], h['is_zero'][:, column]
        keep = a['is_valid'][:, column] & h['is_valid'][:, column]
        assert keep.any()
        q_a = a['q_fg'][keep, column].astype(np.float64)
        q_h = h['q_fg'][keep, column].astype(np.float64)
        difference = q_h - q_a
        lower, upper = bootstrap_bounds(difference)
        output.append(dict(dataset=dataset, architecture=architecture, view=view, scored=n,
                           zero_a=int(z_a.sum()), zero_h=int(z_h.sum()), zero_either=int((z_a | z_h).sum()),
                           excluded_either=int((~keep).sum()),
                           nonfinite_a=int(a['is_nonfinite'][:, column].sum()), nonfinite_h=int(h['is_nonfinite'][:, column].sum()),
                           empty_mask_a=int(a['is_empty_mask'][:, column].sum()), empty_mask_h=int(h['is_empty_mask'][:, column].sum()),
                           constant_positive_a=int(a['is_constant_positive'][:, column].sum()),
                           constant_positive_h=int(h['is_constant_positive'][:, column].sum()),
                           valid_pairs=int(keep.sum()), foreground_area_percent=float(100 * area[keep].mean()),
                           qfg_a=float(q_a.mean()), qfg_h=float(q_h.mean()),
                           foreground_cam_share_a_percent=float(100 * a['foreground_mass_fraction'][keep, column].mean()),
                           foreground_cam_share_h_percent=float(100 * h['foreground_mass_fraction'][keep, column].mean()),
                           delta_h_minus_a=float(difference.mean()), ci95_low=float(lower), ci95_high=float(upper),
                           bootstrap_draws=BOOTSTRAP_N, bootstrap_seed=BOOTSTRAP_SEED))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(COHORT_SIZE), required=True)
    parser.add_argument("--architecture", choices=("resnet18", "mobilenetv2"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pair = args.inputs / args.dataset / args.architecture
    with np.load(pair / "A/per-image.npz") as a, np.load(pair / "H/per-image.npz") as h:
        rows = quantity_rows(args.dataset, args.architecture, dict(a), dict(h), VIEWS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
