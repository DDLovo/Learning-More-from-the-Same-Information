"""Build an exact memory map of all original Flowers JPEG pixels after RGB decoding."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image


def dimensions(path):
    with Image.open(path) as image:
        return image.height, image.width


def decode(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def batches(paths, threads, chunk=64):
    for begin in range(0, len(paths), chunk):
        batch = paths[begin:begin + chunk]
        yield from enumerate(threads.map(decode, batch), start=begin)


def build(images, out, workers):
    paths = [images / "jpg" / f"image_{index:05d}.jpg" for index in range(1, 8190)]
    out.mkdir(parents=True, exist_ok=True)
    audit = out / "rgb-audit.json"
    if audit.is_file():
        print(audit.read_text(), flush=True)
        return
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as threads:
        shapes = np.asarray(list(threads.map(dimensions, paths)), dtype=np.int64)
        offsets = np.r_[0, (np.prod(shapes, axis=1) * 3).cumsum()]
        total = int(offsets[-1])
        if shutil.disk_usage(out).free < total + 2**30:
            raise ValueError(f"Decoded cache requires {total} bytes")
        draft = out / "rgb-bytes.partial.npy"
        data = np.lib.format.open_memmap(draft, dtype=np.uint8, mode="w+", shape=(total,))
        print(f"Building all {len(paths)} original RGBs into {total} bytes", flush=True)
        for index, pixels in batches(paths, threads):
            if pixels.shape != (*shapes[index], 3):
                raise ValueError(f"Original geometry differed at {paths[index]}")
            start_i, end_i = offsets[index:index + 2]
            data[start_i:end_i] = pixels.reshape(-1)
        data.flush()
        del data
        check = np.load(draft, mmap_mode="r", allow_pickle=False)
        for index, pixels in batches(paths, threads):
            start_i, end_i = offsets[index:index + 2]
            if not np.array_equal(check[start_i:end_i], pixels.reshape(-1)):
                raise ValueError(f"Cached RGB differs from an independent original decode at {paths[index]}")
    del check
    draft.replace(out / "rgb-bytes.npy")
    np.save(out / "rgb-shapes.npy", shapes, allow_pickle=False)
    np.save(out / "rgb-offsets.npy", offsets, allow_pickle=False)
    names = ("rgb-bytes.npy", "rgb-shapes.npy", "rgb-offsets.npy")
    meta = {"images": len(paths), "decoded_rgb_byte_equality": True, "bytes_rgb": total,
            "indices": "Official image IDs 1 through 8189; backing row ID−1; RGB HWC uint8",
            "source": str(images), "files": {p: digest(out / p) for p in names},
            "seconds": time.monotonic() - start}
    tmp = audit.with_suffix(".partial")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.replace(audit)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    build(a.images, a.out, a.workers)
