"""Predictive-uncertainty statistics computed from a frozen MLLM's answer
distribution (paper Section: Coarse Global Pass and Predictive Uncertainty).

All quantities are logit-derived and require no extra trained parameters.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .types import AnswerOutput, UncertaintyFeatures

_EPS = 1e-12


def predictive_entropy(probs: np.ndarray) -> float:
    """Shannon entropy H(p) = -sum p log p (Eq. entropy)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(probs: np.ndarray) -> float:
    """Entropy divided by log|C| so it lies in [0, 1] regardless of |C|."""
    n = len(probs)
    if n <= 1:
        return 0.0
    return predictive_entropy(probs) / float(np.log(n))


def confidence_margin(probs: np.ndarray) -> float:
    """m = p_(1) - p_(2): gap between the top two probabilities."""
    if len(probs) < 2:
        return 1.0
    top2 = np.sort(np.asarray(probs, dtype=np.float64))[-2:]
    return float(top2[1] - top2[0])


def cross_pass_agreement(modal_answer: str, sampled_answers: Sequence[str]) -> float:
    """Disagreement a = 1 - (1/K) sum 1[y_k == y_hat] (paper Eq. agreement).

    Returns a value in [0, 1]; higher means more epistemic uncertainty.
    """
    if not sampled_answers:
        return 0.0
    agree = np.mean([1.0 if a == modal_answer else 0.0 for a in sampled_answers])
    return float(1.0 - agree)


def semantic_entropy(
    candidates: Sequence[str],
    probs: np.ndarray,
    equivalence,
) -> float:
    """Meaning-level entropy: cluster candidates by semantic equivalence, sum
    probabilities within a cluster, then take entropy over clusters.

    Parameters
    ----------
    equivalence:
        Callable ``(a, b) -> bool`` deciding whether two answers mean the same
        thing (bidirectional entailment in the paper; exact-match fallback here).
    """
    probs = np.asarray(probs, dtype=np.float64)
    clusters: List[List[int]] = []
    for i, cand in enumerate(candidates):
        placed = False
        for cl in clusters:
            if equivalence(cand, candidates[cl[0]]):
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    cluster_probs = np.array([probs[cl].sum() for cl in clusters], dtype=np.float64)
    cluster_probs = cluster_probs / max(cluster_probs.sum(), _EPS)
    return predictive_entropy(cluster_probs)


def _default_equivalence(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def compute_features(
    coarse: AnswerOutput,
    sampled_answers: Sequence[str],
    equivalence=None,
) -> UncertaintyFeatures:
    """Stack the four statistics into the calibration-head feature vector z."""
    equivalence = equivalence or _default_equivalence
    probs = np.asarray(coarse.probs, dtype=np.float64)
    return UncertaintyFeatures(
        entropy=normalized_entropy(probs),
        margin=confidence_margin(probs),
        agreement=cross_pass_agreement(coarse.pred, sampled_answers),
        semantic_entropy=semantic_entropy(coarse.candidates, probs, equivalence),
        top_prob=float(np.max(probs)),
    )
