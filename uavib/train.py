"""Calibrator fitting (paper Algorithm: Domain-agnostic calibrator training).

Runs the coarse pass over a domain-mixed pool, records the logit-derived
features z and the binary correctness label, then fits the calibration head.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .calibration import CalibrationHead
from .config import UAViBConfig
from .backends.base import MLLMBackend
from .types import Sample
from .uncertainty import compute_features


def collect_features(
    backend: MLLMBackend, pool: Sequence[Sample], cfg: UAViBConfig,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    Z: List[np.ndarray] = []
    y: List[float] = []
    it = tqdm(pool, desc="calib features", disable=not progress)
    for s in it:
        coarse = backend.coarse_answer(s.image, s.question, list(s.candidates), cfg.coarse_tokens)
        sampled = backend.sample_answers(s.image, s.question, list(s.candidates),
                                         cfg.k_passes, cfg.coarse_tokens)
        feat = compute_features(coarse, sampled)
        Z.append(feat.as_vector())
        y.append(1.0 if coarse.pred == s.answer else 0.0)
    return np.array(Z), np.array(y)


def fit_calibrator(
    backend: MLLMBackend, pool: Sequence[Sample], cfg: UAViBConfig,
    verbose: bool = True,
) -> CalibrationHead:
    Z, y = collect_features(backend, pool, cfg, progress=verbose)
    head = CalibrationHead(in_dim=Z.shape[1], hidden=cfg.calibrator_hidden, seed=cfg.seed)
    head.fit(
        Z, y, lr=cfg.calibrator_lr, epochs=cfg.calibrator_epochs,
        batch=cfg.calibrator_batch, lam=cfg.calibrator_lambda,
        n_bins=cfg.ece_bins, seed=cfg.seed, verbose=verbose,
    )
    return head
