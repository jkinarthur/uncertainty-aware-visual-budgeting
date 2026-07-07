"""Abstract MLLM backend interface.

A backend wraps a *frozen* MLLM and exposes exactly the operations UAViB needs:
a coarse pass, region-budgeted encoding, batched region occlusion, stochastic
sampling, and per-region token caps. Concrete backends (Dummy / Qwen2.5-VL /
LLaVA-NeXT) implement these; the pipeline never touches model internals.
"""

from __future__ import annotations

import abc
from typing import List, Sequence

import numpy as np

from ..types import AnswerOutput


class MLLMBackend(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def coarse_answer(
        self, image, question: str, candidates: Sequence[str], coarse_tokens: int
    ) -> AnswerOutput:
        """Encode ``image`` at a low token budget and answer ``question``.

        Must populate ``region_attention`` (row-major over the caller's grid) when
        available, and ``tokens_used``.
        """

    @abc.abstractmethod
    def answer_with_budget(
        self,
        image,
        question: str,
        candidates: Sequence[str],
        region_budgets: np.ndarray,
        grid_h: int,
        grid_w: int,
    ) -> AnswerOutput:
        """Answer using a per-region token allocation (used during refinement)."""

    @abc.abstractmethod
    def answer_with_occlusions(
        self,
        image,
        question: str,
        candidates: Sequence[str],
        grid_h: int,
        grid_w: int,
        coarse_tokens: int,
    ) -> List[AnswerOutput]:
        """Return R = grid_h*grid_w answers, each with one region soft-masked.

        Implementations should batch the R variants into a single forward pass.
        """

    @abc.abstractmethod
    def sample_answers(
        self,
        image,
        question: str,
        candidates: Sequence[str],
        k: int,
        coarse_tokens: int,
    ) -> List[str]:
        """K cheap stochastic passes (MC-dropout / light TTA) for agreement."""

    @abc.abstractmethod
    def region_caps(self, image, grid_h: int, grid_w: int) -> np.ndarray:
        """Per-region token count at native sensor resolution (b_i^max)."""

    # Optional: real KV-cache reuse. Default no-op; Dummy/HF backends may override.
    def reset_cache(self) -> None:  # pragma: no cover - trivial
        pass
