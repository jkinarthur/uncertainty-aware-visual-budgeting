"""Quality, calibration, and reliability metrics (paper Section: Metrics)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .calibration import expected_calibration_error

_EPS = 1e-12


def accuracy(correct: Sequence[bool]) -> float:
    return float(np.mean([1.0 if c else 0.0 for c in correct])) if len(correct) else 0.0


def nll(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Negative log-likelihood of the binary correctness under the confidence."""
    p = np.clip(np.asarray(confidences, dtype=np.float64), _EPS, 1 - _EPS)
    y = np.asarray(correct, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(confidences: np.ndarray, correct: np.ndarray) -> float:
    p = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def auroc_error(uncertainties: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of uncertainty as a predictor of *errors* (positive = incorrect)."""
    u = np.asarray(uncertainties, dtype=np.float64)
    err = 1.0 - np.asarray(correct, dtype=np.float64)
    n_pos, n_neg = err.sum(), (1 - err).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(u)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(u) + 1)
    auc = (ranks[err == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def risk_coverage_auc(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better).

    Order by decreasing confidence; risk = cumulative error rate among covered.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    err = 1.0 - np.asarray(correct, dtype=np.float64)
    order = np.argsort(-conf)
    err_sorted = err[order]
    cum_err = np.cumsum(err_sorted) / (np.arange(len(err_sorted)) + 1)
    coverage = (np.arange(len(err_sorted)) + 1) / len(err_sorted)
    return float(np.trapz(cum_err, coverage))


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    return expected_calibration_error(confidences, correct, n_bins)


def summarize(results, n_bins: int = 15) -> Dict[str, float]:
    """Aggregate a list of QueryResult into the paper's metric bundle."""
    correct = np.array([1.0 if r.correct else 0.0 for r in results], dtype=np.float64)
    conf = np.array([r.confidence for r in results], dtype=np.float64)
    unc = np.array([r.uncertainty for r in results], dtype=np.float64)
    tokens = np.array([r.tokens_used for r in results], dtype=np.float64)
    lat = np.array([r.latency_ms for r in results], dtype=np.float64)
    return {
        "accuracy": accuracy([bool(c) for c in correct]) * 100.0,
        "avg_tokens": float(tokens.mean()),
        "latency_ms": float(np.median(lat)),
        "ece": ece(conf, correct, n_bins),
        "nll": nll(conf, correct),
        "brier": brier_score(conf, correct),
        "rc_auc": risk_coverage_auc(conf, correct),
        "auroc": auroc_error(unc, correct),
        "n": len(results),
    }
