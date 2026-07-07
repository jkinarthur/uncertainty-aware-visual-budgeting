"""Uncertainty-driven token budgeting and water-filling redistribution
(paper Section: Uncertainty-Driven Token Budget + Algorithm: Water-filling).
"""

from __future__ import annotations

import numpy as np

from .config import UAViBConfig


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def global_budget(calibrated_uncertainty: float, cfg: UAViBConfig) -> int:
    """B = B_min + (B_max - B_min) * sigma(gamma (U_hat - tau))  (Eq. budget)."""
    frac = _sigmoid(cfg.gamma * (float(calibrated_uncertainty) - cfg.tau))
    b = cfg.b_min + (cfg.b_max - cfg.b_min) * frac
    return int(round(b))


def allocate(
    total_budget: int,
    region_scores: np.ndarray,
    region_caps: np.ndarray,
    cfg: UAViBConfig,
) -> np.ndarray:
    """Distribute ``total_budget`` across regions proportionally to uncertainty,
    clamp to [region_floor, cap], then water-fill the residual (Eq. allocate +
    Algorithm: Water-filling budget redistribution).
    """
    u = np.asarray(region_scores, dtype=np.float64)
    caps = np.asarray(region_caps, dtype=np.float64)
    floor = float(cfg.region_floor)

    b = np.clip(np.round(total_budget * u), floor, caps)
    residual = total_budget - b.sum()

    step = float(cfg.step_tokens)
    # Water-filling: move resolution steps to/from the most/least uncertain
    # unsaturated regions until the residual is exhausted.
    guard = 0
    max_iter = 10 * len(u) + 100
    while abs(residual) >= step and guard < max_iter:
        guard += 1
        if residual > 0:
            mask = b < caps
            if not mask.any():
                break
            j = np.where(mask, u, -np.inf).argmax()
            b[j] = min(b[j] + step, caps[j])
        else:
            mask = b > floor
            if not mask.any():
                break
            j = np.where(mask, u, np.inf).argmin()
            b[j] = max(b[j] - step, floor)
        residual = total_budget - b.sum()

    return b.astype(int)
