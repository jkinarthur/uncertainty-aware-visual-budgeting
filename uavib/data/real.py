"""Real dataset loaders (used on AWS with downloaded data).

Each loader reads a simple on-disk layout under ``root`` and yields ``Sample``
objects whose ``image`` is a PIL image. The expected layouts are documented per
function. These are intentionally thin: point them at the official downloads.

Directory conventions (override with --data-root):
    root/
      rsvqa/ , rsivqa/ , xlrs_bench/ , acdc/ , isic/ , mvtec_ad/

Each dataset dir is expected to contain a ``manifest.jsonl`` with one JSON per
line: {"image": "relative/path.png", "question": "...",
        "candidates": ["yes","no"], "answer": "yes",
        "roi": [[x0,y0,x1,y1], ...]}  (roi optional).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from ..types import Sample

_DOMAIN = {
    "rsvqa-lr": "remote-sensing", "rsvqa-hr": "remote-sensing",
    "rsivqa": "remote-sensing", "xlrs-bench": "remote-sensing",
    "acdc": "medical", "isic": "medical", "mvtec-ad": "industrial",
}

_SUBDIR = {
    "rsvqa-lr": "rsvqa", "rsvqa-hr": "rsvqa", "rsivqa": "rsivqa",
    "xlrs-bench": "xlrs_bench", "acdc": "acdc", "isic": "isic",
    "mvtec-ad": "mvtec_ad",
}


def _roi_to_grid_mask(roi_boxes, W, H, grid_h, grid_w) -> Optional[np.ndarray]:
    if not roi_boxes:
        return None
    mask = np.zeros(grid_h * grid_w, dtype=np.float64)
    for (x0, y0, x1, y1) in roi_boxes:
        for r in range(grid_h):
            for c in range(grid_w):
                cx0, cy0 = c * W / grid_w, r * H / grid_h
                cx1, cy1 = (c + 1) * W / grid_w, (r + 1) * H / grid_h
                inter = max(0, min(x1, cx1) - max(x0, cx0)) * max(0, min(y1, cy1) - max(y0, cy0))
                if inter > 0:
                    mask[r * grid_w + c] = 1.0
    return mask


def load_real(name: str, root: str, split: str = "test", limit: Optional[int] = None,
              grid_h: int = 8, grid_w: int = 8) -> List[Sample]:
    from PIL import Image  # lazy

    if name not in _SUBDIR:
        raise ValueError(f"Unknown dataset {name!r}")
    ds_dir = os.path.join(root, _SUBDIR[name])
    manifest = os.path.join(ds_dir, f"{split}.jsonl")
    if not os.path.exists(manifest):
        manifest = os.path.join(ds_dir, "manifest.jsonl")
    if not os.path.exists(manifest):
        raise FileNotFoundError(
            f"No manifest for {name} at {manifest}. Prepare data first "
            f"(see README: Datasets)."
        )

    samples: List[Sample] = []
    with open(manifest, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("dataset", name) != name:
                continue
            img_path = os.path.join(ds_dir, rec["image"])
            image = Image.open(img_path).convert("RGB")
            roi = _roi_to_grid_mask(rec.get("roi"), *image.size, grid_h, grid_w)
            samples.append(Sample(
                image=image, question=rec["question"],
                candidates=rec["candidates"], answer=rec["answer"],
                domain=_DOMAIN[name], dataset=name, roi_mask=roi,
                meta=rec.get("meta", {}),
            ))
            if limit and len(samples) >= limit:
                break
    return samples
