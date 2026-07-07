"""A dependency-free simulator backend.

The ``DummyBackend`` lets the *entire* UAViB pipeline, calibrator training, all
baselines, and the evaluation harness run end-to-end on a laptop with no GPU and
no model weights. It implements a faithful *behavioural* model of a frozen MLLM
on structured imagery:

  * a small region-of-interest (ROI) holds the evidence for the answer;
  * accuracy rises as more tokens are spent at higher resolution on the ROI;
  * predictive uncertainty falls as the correct answer becomes resolvable;
  * answer-to-vision attention concentrates on the ROI (plus noise);
  * occluding the ROI raises entropy (high perturbation sensitivity there).

This reproduces the qualitative trends in the paper so the code is exercised and
tested; on AWS you swap in the real Qwen/LLaVA backends and the numbers become
measured rather than simulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from ..types import AnswerOutput
from .base import MLLMBackend


@dataclass
class DummyImage:
    """Opaque 'image' handle carrying the latent scene for the simulator."""

    correct: str
    roi_centers: List[Tuple[float, float]]   # (cx, cy) in [0, 1]
    roi_radius: float = 0.12
    difficulty: float = 0.5                  # higher -> needs more resolution
    full_tokens: int = 1024
    seed: int = 0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class DummyBackend(MLLMBackend):
    name = "dummy"

    def __init__(self, sharpness: float = 7.0, attn_noise: float = 0.15):
        self.sharpness = sharpness
        self.attn_noise = attn_noise

    # --- helpers ---
    def _region_roi_overlap(self, img: DummyImage, grid_h: int, grid_w: int) -> np.ndarray:
        """Fraction of each region covered by any ROI disk (row-major)."""
        overlap = np.zeros(grid_h * grid_w, dtype=np.float64)
        for r in range(grid_h):
            for c in range(grid_w):
                cx = (c + 0.5) / grid_w
                cy = (r + 0.5) / grid_h
                best = 0.0
                for (rx, ry) in img.roi_centers:
                    d = np.hypot(cx - rx, cy - ry)
                    best = max(best, max(0.0, 1.0 - d / (img.roi_radius + 1e-6)))
                overlap[r * grid_w + c] = best
        s = overlap.sum()
        return overlap / s if s > 0 else np.full_like(overlap, 1.0 / len(overlap))

    def _evidence(self, img: DummyImage, region_budgets: np.ndarray,
                  grid_h: int, grid_w: int) -> float:
        """Fraction of the answer evidence resolved in [0, 1].

        Combines a global-gist term (overall downsampling) with an
        ROI-resolution term (detail spent where the answer lives). Concentrating
        tokens on the ROI raises evidence for the *same* total budget, which is
        exactly the behaviour UAViB is designed to exploit.
        """
        region_budgets = np.asarray(region_budgets, dtype=np.float64)
        overlap = self._region_roi_overlap(img, grid_h, grid_w)
        caps = self.region_caps(img, grid_h, grid_w)
        roi_res = float(np.sum(overlap * np.clip(region_budgets / np.maximum(caps, 1e-6), 0.0, 1.0)))
        global_res = float(np.clip(region_budgets.sum() / (0.6 * img.full_tokens), 0.0, 1.0))
        return 0.2 * global_res + 0.8 * roi_res

    # p_correct(evidence, difficulty): gentle, graded mapping so accuracies land
    # in the realistic 65-82% band (paper) rather than a 0/100 step function.
    _A0, _A1, _A2 = 0.7875, 0.115, 0.25

    def _p_correct(self, img: DummyImage, evidence: float,
                   rng: np.random.Generator) -> float:
        p = self._A0 + self._A1 * evidence - self._A2 * img.difficulty
        p += rng.normal(0.0, 0.03)
        return float(np.clip(p, 0.05, 0.97))

    @staticmethod
    def _overconfident(p: float) -> float:
        """Model's *stated* top-1 probability: monotone in p but overconfident,
        so raw confidence is miscalibrated and the calibrator has work to do."""
        return float(np.clip(0.5 + (p - 0.5) * 1.15 + 0.16, 0.5, 0.985))

    def _distribution(self, img: DummyImage, candidates: Sequence[str],
                      evidence: float, rng: np.random.Generator) -> np.ndarray:
        n = len(candidates)
        p_correct = self._p_correct(img, evidence, rng)
        conf = self._overconfident(p_correct)          # stated top-1 prob
        correct = rng.random() < p_correct             # realised outcome
        cand = list(candidates)
        ci = cand.index(img.correct) if img.correct in cand else 0
        if correct:
            top = ci
        else:  # confidently wrong: pick a distractor as the top answer
            others = [j for j in range(n) if j != ci]
            top = int(rng.choice(others)) if others else ci
        probs = np.full(n, (1.0 - conf) / max(n - 1, 1), dtype=np.float64)
        probs[top] = conf
        probs = probs * rng.uniform(0.97, 1.03, size=n)
        return probs / probs.sum()

    def _rng(self, img: DummyImage, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng((img.seed * 1_000_003 + salt) % (2**32))

    # --- interface ---
    def coarse_answer(self, image, question, candidates, coarse_tokens):
        img: DummyImage = image
        grid_h = grid_w = 8  # attention is reported on a default 8x8 grid
        budgets = np.full(grid_h * grid_w, coarse_tokens / (grid_h * grid_w))
        evidence = self._evidence(img, budgets, grid_h, grid_w)
        rng = self._rng(img, salt=1)
        probs = self._distribution(img, candidates, evidence, rng)
        attn = self._attention(img, grid_h, grid_w, rng)
        return AnswerOutput(list(candidates), probs, attn, int(coarse_tokens))

    def _attention(self, img, grid_h, grid_w, rng):
        overlap = self._region_roi_overlap(img, grid_h, grid_w)
        noise = rng.uniform(0, self.attn_noise, size=overlap.shape)
        attn = overlap + noise
        return attn / attn.sum()

    def answer_with_budget(self, image, question, candidates, region_budgets,
                           grid_h, grid_w):
        img: DummyImage = image
        evidence = self._evidence(img, np.asarray(region_budgets), grid_h, grid_w)
        rng = self._rng(img, salt=2 + int(np.sum(region_budgets)) % 9973)
        probs = self._distribution(img, candidates, evidence, rng)
        attn = self._attention(img, grid_h, grid_w, rng)
        return AnswerOutput(list(candidates), probs, attn, int(np.sum(region_budgets)))

    def answer_with_occlusions(self, image, question, candidates,
                               grid_h, grid_w, coarse_tokens):
        img: DummyImage = image
        r = grid_h * grid_w
        base = np.full(r, coarse_tokens / r)
        outs: List[AnswerOutput] = []
        rng = self._rng(img, salt=3)
        for i in range(r):
            occ = base.copy()
            occ[i] = 0.0  # soft-mask region i (mean-pool -> no evidence)
            evidence = self._evidence(img, occ, grid_h, grid_w)
            probs = self._distribution(img, candidates, evidence, rng)
            outs.append(AnswerOutput(list(candidates), probs, None, int(coarse_tokens)))
        return outs

    def sample_answers(self, image, question, candidates, k, coarse_tokens):
        img: DummyImage = image
        grid_h = grid_w = 8
        budgets = np.full(grid_h * grid_w, coarse_tokens / (grid_h * grid_w))
        evidence = self._evidence(img, budgets, grid_h, grid_w)
        answers = []
        for j in range(k):
            rng = self._rng(img, salt=100 + j)
            probs = self._distribution(img, candidates, evidence, rng)
            answers.append(candidates[int(rng.choice(len(candidates), p=probs))])
        return answers

    def region_caps(self, image, grid_h, grid_w):
        img: DummyImage = image
        r = grid_h * grid_w
        return np.full(r, max(4.0, img.full_tokens / r), dtype=np.float64)
