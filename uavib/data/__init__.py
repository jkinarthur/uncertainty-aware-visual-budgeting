"""Dataset access for UAViB."""

from typing import List, Optional

from ..types import Sample
from . import synthetic
from .synthetic import DATASETS, DOMAINS, dataset_domain

__all__ = ["DATASETS", "DOMAINS", "dataset_domain", "load_dataset",
           "calibration_pool"]


def load_dataset(
    name: str,
    split: str = "test",
    n: int = 500,
    seed: int = 0,
    source: str = "synthetic",
    data_root: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Sample]:
    """Load a dataset either synthetically (dummy backend) or from disk (real)."""
    if source == "synthetic":
        return synthetic.generate(name, n=n, seed=seed)
    if source == "real":
        from .real import load_real
        if data_root is None:
            raise ValueError("data_root is required for source='real'")
        return load_real(name, root=data_root, split=split, limit=limit)
    raise ValueError(f"Unknown source {source!r}")


def calibration_pool(source: str = "synthetic", n_per_dataset: int = 200,
                     seed: int = 0, data_root: Optional[str] = None,
                     exclude_domains=None, datasets=None) -> List[Sample]:
    """Domain-mixed calibration pool; optionally exclude domains (for LODO).

    ``datasets`` restricts the real-data pool to the given dataset names; any
    dataset without a ``calib`` manifest on disk is skipped so the pool can be
    built from whatever real data has been prepared.
    """
    if source == "synthetic":
        pool = synthetic.generate_calibration_pool(n_per_dataset, seed)
    else:
        from .real import load_real
        pool = []
        for name in (datasets or DATASETS):
            try:
                pool.extend(load_real(name, root=data_root, split="calib",
                                      limit=n_per_dataset))
            except FileNotFoundError:
                continue
    if exclude_domains:
        exclude = set(exclude_domains)
        pool = [s for s in pool if s.domain not in exclude]
    return pool
