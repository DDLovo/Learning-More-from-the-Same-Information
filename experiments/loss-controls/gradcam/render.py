"""Render the same image and true-class Grad-CAM from paired A and H checkpoints."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(a, h, out):
    for key in ("image_id", "true_label", "views", "rgb_float"):
        np.testing.assert_array_equal(a[key], h[key], err_msg=key)
    names = ("clean", "foreground-only", "background-replaced")
    views = [a["views"].tolist().index(name) for name in names]
    cmap = plt.get_cmap("magma")
    for row, image_id in enumerate(a["image_id"]):
        maps = [arm["cam"][row, views] * arm["cam_peak"][row, views, None, None] for arm in (a, h)]
        peak = max(float(cam.max()) for cam in maps)
        fig, axes = plt.subplots(len(views), 3, figsize=(8, 8), layout="constrained")
        fig.suptitle(f"Image {image_id} · true class {a['true_label'][row]}")
        for i, (name, view) in enumerate(zip(names, views)):
            rgb = a["rgb_float"][row, view]
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title(name)
            for column, (arm, cam, label) in enumerate(zip((a, h), maps, ("A", "H")), 1):
                scaled = cam[i] / peak if peak > 0 else np.zeros_like(cam[i])
                alpha = 0.57 * scaled[..., None]
                axes[i, column].imshow(np.clip((1 - alpha) * rgb + alpha * cmap(scaled)[..., :3], 0, 1))
                axes[i, column].set_title(f"{label} · predicted {arm['pred_label'][row, view]}")
        for axis in axes.flat:
            axis.set_axis_off()
        fig.savefig(out / f"{image_id}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True, help="Directory with A/ and H/ subdirectories")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.pair / "A/examples.npz") as a, np.load(args.pair / "H/examples.npz") as h:
        render(dict(a), dict(h), args.out)


if __name__ == "__main__":
    main()
