"""Baseline allocation policies (paper Section: Backbones and Baselines).

Every baseline is a *genuine* algorithm expressed through the same backend
interface as UAViB, so comparisons are apples-to-apples. They differ from UAViB
in three controlled ways: (a) whether the global budget is adaptive, (b) how the
budget is distributed spatially, and (c) whether the emitted confidence is
calibrated. UAViB is the only method that is adaptive + fused-attribution +
calibrated + progressively refined.

The external methods (ToMe/FastV/DynamicViT/PrATo/GRASP/DualComp) are represented
by their characteristic (budget, spatial-strategy) operating point; swap in their
official implementations on the real backends if desired.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..attribution import attribute, perturbation_sensitivity
from ..budget import allocate, global_budget
from ..config import UAViBConfig
from ..backends.base import MLLMBackend
from ..types import QueryResult
from ..uncertainty import compute_features, normalized_entropy


def _uncalibrated_confidence(probs) -> float:
    return float(1.0 - normalized_entropy(np.asarray(probs, dtype=np.float64)))


def _result(pred, conf, tokens, region_budgets, region_u, features, latency, answer):
    return QueryResult(
        answer=pred, confidence=conf, uncertainty=1.0 - conf,
        tokens_used=int(tokens), num_refine_steps=0,
        region_budgets=region_budgets, region_uncertainty=region_u,
        features=features, latency_ms=latency,
        correct=(None if answer is None else pred == answer),
    )


@dataclass
class SingleShotBaseline:
    """Fixed or adaptive global budget, one shot, uncalibrated confidence."""

    name: str
    budget: Optional[int] = None          # None -> adaptive entropy budget
    strategy: str = "uniform"             # uniform | attention | sensitivity | random
    cfg: UAViBConfig = None

    def run(self, backend: MLLMBackend, image, question, candidates, answer=None):
        cfg = self.cfg or UAViBConfig()
        t0 = time.perf_counter()
        candidates = list(candidates)
        coarse = backend.coarse_answer(image, question, candidates, cfg.coarse_tokens)
        sampled = backend.sample_answers(image, question, candidates, cfg.k_passes, cfg.coarse_tokens)
        features = compute_features(coarse, sampled)
        caps = backend.region_caps(image, cfg.grid_h, cfg.grid_w)
        r = cfg.num_regions

        if self.budget is None:  # adaptive global budget from raw entropy
            b = global_budget(features.entropy, cfg)
        else:
            b = int(self.budget)

        if self.strategy == "uniform":
            region_u = np.full(r, 1.0 / r)
        elif self.strategy == "random":
            rng = np.random.default_rng(cfg.seed + hash(question) % 9973)
            region_u = rng.dirichlet(np.ones(r))
        elif self.strategy == "attention":
            region_u = attribute(backend, image, question, candidates, coarse, cfg)
        elif self.strategy == "sensitivity":
            s = perturbation_sensitivity(backend, image, question, candidates, coarse, cfg)
            region_u = s / s.sum() if s.sum() > 0 else np.full(r, 1.0 / r)
        else:
            region_u = np.full(r, 1.0 / r)

        region_budgets = allocate(b, region_u, caps, cfg)
        out = backend.answer_with_budget(image, question, candidates, region_budgets,
                                         cfg.grid_h, cfg.grid_w)
        conf = _uncalibrated_confidence(out.probs)
        latency = (time.perf_counter() - t0) * 1000.0
        return _result(out.pred, conf, out.tokens_used, region_budgets, region_u,
                       features, latency, answer)


@dataclass
class CoarseOnly:
    name: str = "coarse-only"
    cfg: UAViBConfig = None

    def run(self, backend, image, question, candidates, answer=None):
        cfg = self.cfg or UAViBConfig()
        t0 = time.perf_counter()
        candidates = list(candidates)
        coarse = backend.coarse_answer(image, question, candidates, cfg.coarse_tokens)
        sampled = backend.sample_answers(image, question, candidates, cfg.k_passes, cfg.coarse_tokens)
        features = compute_features(coarse, sampled)
        conf = _uncalibrated_confidence(coarse.probs)
        latency = (time.perf_counter() - t0) * 1000.0
        return _result(coarse.pred, conf, coarse.tokens_used,
                       np.zeros(cfg.num_regions), np.full(cfg.num_regions, 1.0 / cfg.num_regions),
                       features, latency, answer)


@dataclass
class FullResolution:
    name: str = "full-resolution"
    cfg: UAViBConfig = None

    def run(self, backend, image, question, candidates, answer=None):
        cfg = self.cfg or UAViBConfig()
        t0 = time.perf_counter()
        candidates = list(candidates)
        caps = backend.region_caps(image, cfg.grid_h, cfg.grid_w)
        region_budgets = caps.astype(int)
        coarse = backend.coarse_answer(image, question, candidates, cfg.coarse_tokens)
        sampled = backend.sample_answers(image, question, candidates, cfg.k_passes, cfg.coarse_tokens)
        features = compute_features(coarse, sampled)
        out = backend.answer_with_budget(image, question, candidates, region_budgets,
                                         cfg.grid_h, cfg.grid_w)
        conf = _uncalibrated_confidence(out.probs)
        latency = (time.perf_counter() - t0) * 1000.0
        return _result(out.pred, conf, out.tokens_used, region_budgets,
                       np.full(cfg.num_regions, 1.0 / cfg.num_regions), features, latency, answer)


@dataclass
class OracleBudget:
    """Upper bound: smallest fixed budget that recovers the full-res answer."""

    name: str = "oracle"
    cfg: UAViBConfig = None
    grid_of_budgets: Sequence[int] = (144, 256, 384, 512, 768, 1024)

    def run(self, backend, image, question, candidates, answer=None):
        cfg = self.cfg or UAViBConfig()
        t0 = time.perf_counter()
        candidates = list(candidates)
        caps = backend.region_caps(image, cfg.grid_h, cfg.grid_w)
        region_u = attribute(backend, image, question, candidates,
                             backend.coarse_answer(image, question, candidates, cfg.coarse_tokens), cfg)
        full = backend.answer_with_budget(
            image, question, candidates, caps.astype(int), cfg.grid_h, cfg.grid_w)
        target = full.pred
        chosen = None
        for b in self.grid_of_budgets:
            rb = allocate(int(b), region_u, caps, cfg)
            out = backend.answer_with_budget(image, question, candidates, rb, cfg.grid_h, cfg.grid_w)
            if out.pred == target:
                chosen = (out, rb)
                break
        if chosen is None:
            chosen = (full, caps.astype(int))
        out, rb = chosen
        coarse = backend.coarse_answer(image, question, candidates, cfg.coarse_tokens)
        sampled = backend.sample_answers(image, question, candidates, cfg.k_passes, cfg.coarse_tokens)
        features = compute_features(coarse, sampled)
        conf = _uncalibrated_confidence(out.probs)
        latency = (time.perf_counter() - t0) * 1000.0
        return _result(out.pred, conf, out.tokens_used, rb, region_u, features, latency, answer)


def build_baselines(cfg: UAViBConfig) -> dict:
    """Return the full baseline suite keyed by display name."""
    b = {
        "full-resolution": FullResolution(cfg=cfg),
        "coarse-only": CoarseOnly(cfg=cfg),
        "fixed-512": SingleShotBaseline("fixed-512", budget=512, strategy="uniform", cfg=cfg),
        "tiling": SingleShotBaseline("tiling", budget=896, strategy="uniform", cfg=cfg),
        "tome": SingleShotBaseline("tome", budget=332, strategy="uniform", cfg=cfg),
        "fastv": SingleShotBaseline("fastv", budget=316, strategy="attention", cfg=cfg),
        "dynamicvit": SingleShotBaseline("dynamicvit", budget=340, strategy="sensitivity", cfg=cfg),
        "prato": SingleShotBaseline("prato", budget=358, strategy="attention", cfg=cfg),
        "grasp": SingleShotBaseline("grasp", budget=512, strategy="attention", cfg=cfg),
        "dualcomp": SingleShotBaseline("dualcomp", budget=300, strategy="attention", cfg=cfg),
        "random-regional": SingleShotBaseline("random-regional", budget=288, strategy="random", cfg=cfg),
        "entropy-only": SingleShotBaseline("entropy-only", budget=None, strategy="uniform", cfg=cfg),
        "attention-only": SingleShotBaseline("attention-only", budget=None, strategy="attention", cfg=cfg),
        "oracle": OracleBudget(cfg=cfg),
    }
    return b
