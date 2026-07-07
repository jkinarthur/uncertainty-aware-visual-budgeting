"""Baseline suite for UAViB comparisons."""

from .methods import (
    CoarseOnly,
    FullResolution,
    OracleBudget,
    SingleShotBaseline,
    build_baselines,
)

__all__ = [
    "SingleShotBaseline",
    "CoarseOnly",
    "FullResolution",
    "OracleBudget",
    "build_baselines",
]
