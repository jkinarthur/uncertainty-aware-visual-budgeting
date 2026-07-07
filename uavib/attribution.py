"""Region-level uncertainty attribution (paper Section: Region-Level
Uncertainty Attribution).

We fuse two signals over the ``R = R_h x R_w`` region grid:

  * answer-to-vision attention  a_i  (correlational)
  * perturbation sensitivity    s_i  (causal), via batched region occlusion

The fused score is  u_i = alpha * ~a_i + (1 - alpha) * ~s_i  with min-max
normalisation and sum_i u_i = 1  (Eq. region).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .backends.base import MLLMBackend
from .config import UAViBConfig
from .types import AnswerOutput
from .uncertainty import predictive_entropy

_EPS = 1e-12


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < _EPS:
        return np.full_like(x, 1.0 / len(x))
    return (x - lo) / (hi - lo)


def perturbation_sensitivity(
    backend: MLLMBackend,
    image,
    question: str,
    candidates,
    coarse: AnswerOutput,
    cfg: UAViBConfig,
) -> np.ndarray:
    """s_i = [ H(p(.|V \\ i)) - H(p(.|V)) ]_+   (Eq. sensitivity).

    The R occluded variants share all non-occluded tokens, so the backend runs
    them in a single batched pass (region tokens replaced by the mean-pooled
    token — a soft mask that keeps sequence length constant).
    """
    base_h = predictive_entropy(coarse.probs)
    occluded = backend.answer_with_occlusions(
        image, question, list(candidates), cfg.grid_h, cfg.grid_w, cfg.coarse_tokens
    )
    s = np.array(
        [max(predictive_entropy(o.probs) - base_h, 0.0) for o in occluded],
        dtype=np.float64,
    )
    return s


def attribute(
    backend: MLLMBackend,
    image,
    question: str,
    candidates,
    coarse: AnswerOutput,
    cfg: UAViBConfig,
) -> np.ndarray:
    """Return the fused, normalised region-uncertainty scores u_i (sum = 1)."""
    r = cfg.num_regions

    # Attention mass a_i (fall back to uniform if the backend cannot expose it).
    if coarse.region_attention is not None and len(coarse.region_attention) == r:
        a = np.asarray(coarse.region_attention, dtype=np.float64)
    else:
        a = np.full(r, 1.0 / r, dtype=np.float64)

    s = perturbation_sensitivity(backend, image, question, candidates, coarse, cfg)

    u = cfg.alpha * _minmax(a) + (1.0 - cfg.alpha) * _minmax(s)
    total = u.sum()
    if total < _EPS:
        return np.full(r, 1.0 / r, dtype=np.float64)
    return u / total
