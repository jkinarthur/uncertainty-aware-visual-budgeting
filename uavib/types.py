"""Shared dataclasses and light-weight structures used across UAViB."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AnswerOutput:
    """Result of a single forward pass of a (frozen) MLLM backend.

    Attributes
    ----------
    candidates:
        Ordered list of candidate answer strings the probabilities refer to.
    probs:
        Probability over ``candidates`` (sums to 1). For free-form queries these
        are probabilities over sampled/clustered answer candidates.
    region_attention:
        Answer-to-vision attention mass per region, flattened row-major over the
        ``R_h x R_w`` grid and normalised to sum to 1. ``None`` if unavailable.
    tokens_used:
        Number of vision tokens consumed by this pass.
    pred_index:
        Index into ``candidates`` of the arg-max answer.
    """

    candidates: List[str]
    probs: np.ndarray
    region_attention: Optional[np.ndarray] = None
    tokens_used: int = 0

    @property
    def pred_index(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def pred(self) -> str:
        return self.candidates[self.pred_index]


@dataclass
class RegionGrid:
    """An ``R_h x R_w`` partition of the image aligned to the patch stride."""

    height: int
    width: int

    @property
    def num_regions(self) -> int:
        return self.height * self.width

    def index(self, row: int, col: int) -> int:
        return row * self.width + col

    def coords(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.width)


@dataclass
class UncertaintyFeatures:
    """Logit-derived statistics consumed by the calibration head."""

    entropy: float
    margin: float
    agreement: float
    semantic_entropy: float
    top_prob: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.entropy,
                self.margin,
                self.agreement,
                self.semantic_entropy,
                self.top_prob,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def feature_names() -> List[str]:
        return ["entropy", "margin", "agreement", "semantic_entropy", "top_prob"]


@dataclass
class QueryResult:
    """Everything UAViB returns for one (image, question) query."""

    answer: str
    confidence: float
    uncertainty: float
    tokens_used: int
    num_refine_steps: int
    region_budgets: np.ndarray
    region_uncertainty: np.ndarray
    features: UncertaintyFeatures
    latency_ms: float = 0.0
    correct: Optional[bool] = None
    history: List[Dict] = field(default_factory=list)


@dataclass
class Sample:
    """A single evaluation example (backend-agnostic)."""

    image: object  # PIL.Image, ndarray, or an opaque handle for DummyBackend
    question: str
    candidates: Sequence[str]
    answer: str
    domain: str
    dataset: str
    roi_mask: Optional[np.ndarray] = None  # optional ground-truth ROI (R_h x R_w)
    meta: Dict = field(default_factory=dict)
