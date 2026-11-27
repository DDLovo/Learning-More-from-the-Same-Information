"""Record and replay the two native BF16 teacher logits used by the author MLKD."""

import json
import zipfile

import numpy as np
import torch
from torch import nn


class VisualTeacherLogits(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher
        self.cache = None

    def begin_epoch(self, path, mode, binding, count, classes, device):
        self.path = path
        self.mode = mode
        self.binding = binding
        self.shape = (2, count, classes)
        self.rows = np.empty(count, dtype=np.int64)
        self.cache = None
        self.online = self.replayed = 0
        self.reason = None
        self.first = None
        if mode == "record":
            self.cache = torch.empty(self.shape, dtype=torch.bfloat16, device=device)
        if mode == "replay":
            try:
                with np.load(path, allow_pickle=False) as saved:
                    manifest = json.loads(saved["metadata"].item())
                    visual_bits = saved["logit_bits"]
                    row_indices = saved["row_indices"]
                    if (manifest["binding"] != binding or visual_bits.dtype != np.uint16 or visual_bits.shape != self.shape
                            or row_indices.dtype != np.int64 or row_indices.shape != (count,)):
                        raise ValueError("cache metadata or tensor layout did not match")
                self.cache = torch.from_numpy(visual_bits).view(torch.bfloat16).to(device)
                self.cached_rows = row_indices
                self.first = manifest["first_augmented_batch_sha256"]
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                self.reason = type(error).__name__

    def open_batch(self, start, row_indices, first_hashes=None):
        self.position = slice(start, start + len(row_indices))
        self.calls = 0
        self.rows[self.position] = row_indices
        if self.mode == "record" and start == 0:
            self.first = first_hashes
        if self.mode == "replay" and self.cache is not None:
            mismatched = not np.array_equal(self.cached_rows[self.position], row_indices)
            if mismatched or (start == 0 and first_hashes != self.first):
                self.reason = "row_ids_or_augmented_first_batch_changed"
                self.cache = None

    def forward(self, images, is_feat=False):
        if not is_feat or self.calls >= 2:
            raise ValueError("The frozen author MLKD must request two teacher feature forwards per training batch")
        view = self.calls
        self.calls += 1
        if self.mode == "replay" and self.cache is not None:
            self.replayed += len(images)
            return None, self.cache[view, self.position]
        features, logits = self.teacher(images, is_feat=True)
        self.online += len(images)
        if self.mode == "record":
            if logits.dtype != torch.bfloat16:
                raise ValueError("Record the actual native BF16 teacher output without converting its precision")
            self.cache[view, self.position].copy_(logits.detach())
        return features, logits

    def close_batch(self):
        if self.calls != 2:
            raise ValueError("The frozen author MLKD did not request both teacher views")

    def end_epoch(self):
        if self.mode == "record":
            if self.online != 2 * len(self.rows):
                raise ValueError("Refusing to save incomplete visual teacher logits")
            metadata = {"binding": self.binding, "first_augmented_batch_sha256": self.first}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".partial.npz")
            bits = self.cache.view(torch.uint16).cpu().numpy()
            np.savez(temporary, metadata=json.dumps(metadata, sort_keys=True), row_indices=self.rows, logit_bits=bits)
            temporary.replace(self.path)
        self.cache = None
        return {"requested_mode": self.mode, "online_teacher_image_views": self.online,
                "replayed_teacher_image_views": self.replayed, "fallback": self.reason}
