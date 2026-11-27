"""Train the supervised Flowers ResNet-50 teacher for the author hybrid benchmark."""

import argparse
import hashlib
from importlib import util
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .teacher import write_json
from .train import sha256
from .train_visual_lavid import BATCH_SIZE, MILESTONES, PairedOrder, VisualDataset, evaluate, state_hash, write_checkpoint
from .visual_reference import AUTHOR_REVISION


EPOCHS = 240
CLASSES = 102


class FlowerTrainingImages(VisualDataset):
    def __init__(self, root, rgb_cache=None):
        super().__init__(root, "flowers", True)
        self.rgb_cache = rgb_cache
        self.rgb_arrays = None
        if rgb_cache:
            proof = json.loads((rgb_cache / "rgb-audit.json").read_bytes())
            if not proof.get("decoded_rgb_byte_equality") or proof["images"] != 8189:
                raise ValueError("Use the complete, verified Flowers RGB cache")

    def seed_transform(self, seed):
        torch.random.default_generator.manual_seed(int(seed))

    def decoded_rgb(self, index):
        if self.rgb_cache is None:
            with Image.open(self.paths[index]) as image:
                return image.convert("RGB")
        if self.rgb_arrays is None:
            self.rgb_arrays = tuple(np.load(self.rgb_cache / f"rgb-{name}.npy", mmap_mode="r", allow_pickle=False)
                                    for name in ("shapes", "offsets", "bytes"))
            shapes, offsets, pixels = self.rgb_arrays
            if (shapes.shape != (8189, 2) or offsets.shape != (8190,) or pixels.dtype != np.uint8 or
                    pixels.ndim != 1 or int(offsets[-1]) != len(pixels)):
                raise ValueError("Invalid Flowers original-resolution RGB cache layout")
        shapes, offsets, pixels = self.rgb_arrays
        number = int(self.ids[index]) - 1
        height, width = map(int, shapes[number])
        value = pixels[int(offsets[number]):int(offsets[number + 1])]
        return Image.fromarray(value.reshape(height, width, 3))

    def __getitem__(self, item):
        index, seed = item if isinstance(item, tuple) else (item, None)
        if seed is not None:
            self.seed_transform(seed)
        return self.transform(self.decoded_rgb(index)), int(self.labels[index]), index


def flower_evaluation(root, cache=None):
    heldout = VisualDataset(root, "flowers", False)
    if cache:
        from .visual_validation_cache import CachedVisualEval

        heldout = CachedVisualEval(cache, heldout, dataset="flowers")
    return heldout


def resnet_from_author(path, depth, classes=CLASSES):
    spec = util.spec_from_file_location("lavid_author_resnet", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, f"resnet{depth}")(pretrained=False, num_classes=classes)


def teacher_epoch(model, loader, optimizer, device, amp):
    model.train()
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    seen = 0
    for step, (images, labels, _) in enumerate(loader, 1):
        if step == 1:
            first_batch = hashlib.sha256(images.numpy().tobytes()).hexdigest()
        x, y = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        loss.backward()
        optimizer.step()
        totals += torch.stack((loss.detach() * len(y), (logits.argmax(1) == y).sum())).double()
        seen += len(y)
    ce, accuracy = (totals / seen).tolist()
    if seen != len(loader.dataset) or not np.isfinite((ce, accuracy)).all():
        raise FloatingPointError("The supervised teacher epoch was incomplete or non-finite")
    return {"records": seen, "updates": step, "ce": ce, "train_objective": ce, "train_top1": accuracy,
            "first_augmented_batch_sha256": first_batch}


def run(args):
    from .visual_benchmark_data import RELEASES, populations

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    amp = device.type == "cuda"
    source = args.author_source / "lavid/models/resnet.py"
    train, heldout, metadata = populations(args.dataset, args.images, args.cohort, args.training_rgb_cache,
                                          args.validation_cache)
    classes = RELEASES[args.dataset][1]
    order = PairedOrder(train, args.seed)
    opts = {"batch_size": BATCH_SIZE, "num_workers": args.workers, "persistent_workers": args.workers > 0,
            "pin_memory": amp}
    loader = DataLoader(train, sampler=order, **opts)
    test = DataLoader(heldout, **opts)

    model = resnet_from_author(source, 50, classes).to(device)
    initial = state_hash(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.1)
    output = args.out / args.dataset / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = {"author_revision": AUTHOR_REVISION, "architecture": "resnet50", "dataset": args.dataset,
              "initialization": "author random initialization", "seed": args.seed, "epochs": EPOCHS,
              "batch_size": BATCH_SIZE, "precision": "bfloat16-mixed" if amp else "float32",
              "optimizer": {"lr": .01, "momentum": .9, "weight_decay": .0005,
                            "milestones": list(MILESTONES), "gamma": .1},
              "model_source_sha256": sha256(source), "initial_state_sha256": initial,
              "metadata_sha256": metadata,
              "training_image_ids": train.ids.tolist(), "evaluation_image_ids": heldout.ids.tolist()}
    cfg_path, last, best_path = (output / name for name in ("config.json", "last.pt", "author-best.pt"))
    if cfg_path.exists() and json.loads(cfg_path.read_bytes()) != config:
        raise ValueError(f"The teacher recipe or data changed at {output}")
    if (output / "results.json").is_file():
        print(f"Reuse the sealed teacher at {output}", flush=True)
        return
    write_json(cfg_path, config)
    write_json(output / "input-cache.json", {
        "train_rgb": str(args.training_rgb_cache) if args.training_rgb_cache else None,
        "validation_crop_sha256": heldout.metadata["sha256"] if args.validation_cache else None,
        "save_optimizer_every_n_epochs": args.checkpoint_every,
    })
    history, best = [], {"epoch": None, "validation_nll": float("inf")}
    if last.exists():
        state = torch.load(last, weights_only=True, map_location=device)
        if state["initial_state_sha256"] != initial:
            raise ValueError("Teacher seed produced a different initial model")
        model.load_state_dict(state["student"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best = state["history"], state["author_best"]
        history_path = output / "history.json"
        if history_path.is_file():
            previous = json.loads(history_path.read_bytes())
            if len(previous) > len(history):
                if previous[:len(history)] != history:
                    raise ValueError("Stored curve and teacher checkpoint have different histories")
                write_json(output / f"history-before-resume-from-{len(history):03d}.json", previous)
        del state

    for epoch in range(len(history) + 1, EPOCHS + 1):
        order.epoch = epoch
        started = time.perf_counter()
        row = teacher_epoch(model, loader, optimizer, device, amp)
        row.update(epoch=epoch, lr=optimizer.param_groups[0]["lr"],
                   training_seconds=time.perf_counter() - started,
                   ordered_index_and_augmentation_seed_sha256=hashlib.sha256(order.plan().tobytes()).hexdigest())
        scheduler.step()
        state = {"epoch": epoch, "student": model.state_dict(), "initial_state_sha256": initial}
        if epoch == 1 or epoch % 5 == 0:
            started_eval = time.perf_counter()
            scores = evaluate(model, test, device, amp, classes)
            if not np.isfinite(list(scores.values())).all():
                raise FloatingPointError(f"Non-finite teacher evaluation at epoch {epoch}")
            row.update(author_eval_split=scores, validation_seconds=time.perf_counter() - started_eval)
            if epoch % 5 == 0 and scores["nll"] < best["validation_nll"]:
                best = {"epoch": epoch, "validation_nll": scores["nll"]}
                write_checkpoint(best_path, state)
        history.append(row)
        if epoch % args.checkpoint_every == 0 or epoch == EPOCHS:
            write_checkpoint(last, state | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                            "history": history, "author_best": best})
        write_json(output / "history.json", history)
        if epoch <= 3 or epoch % 5 == 0:
            print(f"{args.dataset} teacher ResNet-50 seed={args.seed} epoch={epoch}/{EPOCHS} records={row['records']} "
                  f"steps={row['updates']} CE={row['ce']:.6f} wall={time.perf_counter() - started:.2f}", flush=True)

    final_path = output / "final.pt"
    write_checkpoint(final_path, {"epoch": EPOCHS, "student": model.state_dict(), "initial_state_sha256": initial})
    final = evaluate(model, test, device, False, classes, output / "final_predictions.npz")
    selected = torch.load(best_path, weights_only=True, map_location=device)
    model.load_state_dict(selected["student"], strict=True)
    selected_metrics = evaluate(model, test, device, False, classes, output / "author_best_predictions.npz")
    files = ("final.pt", "author-best.pt", "final_predictions.npz", "author_best_predictions.npz")
    write_json(output / "results.json", {"architecture": "resnet50", "epochs": EPOCHS,
                                         "optimizer_steps": sum(row["updates"] for row in history),
                                         "initial_state_sha256": initial, "author_selection": best,
                                         "fixed_final_fp32": final, "author_selected_fp32": selected_metrics,
                                         "artifact_sha256": {name: sha256(output / name) for name in files}})


def main():
    from .visual_benchmark_data import RELEASES

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "author-source", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--dataset", choices=RELEASES, default="flowers")
    for name in ("training-rgb-cache", "validation-cache", "cohort"):
        p.add_argument(f"--{name}", type=Path)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, choices=(1, 5), default=1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.seed < 0 or args.workers < 0:
        p.error("Use a nonnegative seed and worker count")
    run(args)


if __name__ == "__main__":
    main()
