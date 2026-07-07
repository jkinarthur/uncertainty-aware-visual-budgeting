"""Synthetic data generator.

Produces domain-realistic ``Sample`` objects whose ``image`` field is a
``DummyImage`` (latent scene) so the whole pipeline, calibrator, baselines, and
metrics run without any real weights or datasets. Difficulty, ROI size, and
answer cardinality are tuned per dataset so the simulated accuracy ordering
matches the structured-imagery regimes described in the paper.

On AWS, replace these with the real loaders in ``uavib.data.real``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..backends.dummy import DummyImage
from ..types import Sample

# (domain, candidates, difficulty_mean, roi_radius, roi_count)
_SPEC: Dict[str, dict] = {
    "rsvqa-lr":   dict(domain="remote-sensing", cands=["yes", "no"], diff=0.30, radius=0.18, rois=1),
    "rsvqa-hr":   dict(domain="remote-sensing", cands=["yes", "no"], diff=0.38, radius=0.14, rois=1),
    "rsivqa":     dict(domain="remote-sensing", cands=["0", "1", "2", "many"], diff=0.45, radius=0.12, rois=2),
    "xlrs-bench": dict(domain="remote-sensing", cands=["A", "B", "C", "D"], diff=0.62, radius=0.06, rois=1),
    "acdc":       dict(domain="medical", cands=["normal", "abnormal"], diff=0.48, radius=0.10, rois=1),
    "isic":       dict(domain="medical", cands=["benign", "malignant"], diff=0.44, radius=0.11, rois=1),
    "mvtec-ad":   dict(domain="industrial", cands=["good", "defect"], diff=0.40, radius=0.08, rois=1),
}

DATASETS = list(_SPEC.keys())
DOMAINS = {"remote-sensing", "medical", "industrial"}


def dataset_domain(name: str) -> str:
    return _SPEC[name]["domain"]


def generate(name: str, n: int, seed: int = 0, full_tokens: int = 1024) -> List[Sample]:
    if name not in _SPEC:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")
    spec = _SPEC[name]
    rng = np.random.default_rng(seed + abs(hash(name)) % 10_000)
    samples: List[Sample] = []
    for i in range(n):
        cands = spec["cands"]
        ans = cands[int(rng.integers(len(cands)))]
        rois = [(float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)))
                for _ in range(spec["rois"])]
        difficulty = float(np.clip(rng.normal(spec["diff"], 0.12), 0.05, 0.95))
        img = DummyImage(
            correct=ans, roi_centers=rois, roi_radius=spec["radius"],
            difficulty=difficulty, full_tokens=full_tokens,
            seed=int(rng.integers(2**31)),
        )
        samples.append(Sample(
            image=img, question=f"[{name}] structured-image query #{i}",
            candidates=cands, answer=ans, domain=spec["domain"], dataset=name,
        ))
    return samples


def generate_calibration_pool(n_per_dataset: int = 200, seed: int = 0) -> List[Sample]:
    """Domain-mixed calibration split disjoint from test (different seed offset)."""
    pool: List[Sample] = []
    for j, name in enumerate(DATASETS):
        pool.extend(generate(name, n_per_dataset, seed=10_000 + seed + j))
    return pool
