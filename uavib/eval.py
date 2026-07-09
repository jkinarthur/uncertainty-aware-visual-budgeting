"""Evaluation harness: run UAViB and baselines over datasets and aggregate the
paper's metric bundle (accuracy, tokens, latency, ECE, NLL, Brier, RC-AUC,
AUROC), with a paired bootstrap for significance and LODO transfer support.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .config import UAViBConfig
from .backends.base import MLLMBackend
from .baselines import build_baselines
from .metrics import summarize
from .pipeline import UAViB
from .types import QueryResult, Sample


def run_method(method, backend, samples: Sequence[Sample], progress=False) -> List[QueryResult]:
    results = []
    it = tqdm(samples, desc=getattr(method, "name", "uavib"), disable=not progress)
    for s in it:
        try:
            if isinstance(method, UAViB):
                r = method.run(s.image, s.question, list(s.candidates), answer=s.answer)
            else:
                r = method.run(backend, s.image, s.question, list(s.candidates), answer=s.answer)
        except Exception as exc:  # pragma: no cover - GPU OOM safety net
            # A single pathological sample (typically a CUDA OOM on a very wide
            # budget) must not discard an entire method's results. Clear the GPU
            # cache and fall back to the cheap coarse pass for this one sample so
            # the run completes and is still saved.
            _clear_cuda_cache()
            r = _coarse_fallback(backend, method, s)
            if r is None:
                raise exc
        results.append(r)
    return results


def _clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _coarse_fallback(backend, method, s: Sample):
    """Cheap coarse-pass result used when a sample OOMs, so the run survives."""
    try:
        cfg = getattr(method, "cfg", None) or getattr(method, "config", None)
        coarse_tokens = getattr(cfg, "coarse_tokens", 64) if cfg is not None else 64
        num_regions = getattr(cfg, "num_regions", 1) if cfg is not None else 1
        out = backend.coarse_answer(s.image, s.question, list(s.candidates), coarse_tokens)
        conf = float(np.max(out.probs))
        return QueryResult(
            answer=out.pred, confidence=conf, uncertainty=1.0 - conf,
            tokens_used=int(out.tokens_used), num_refine_steps=0,
            region_budgets=np.zeros(num_regions),
            region_uncertainty=np.full(num_regions, 1.0 / max(num_regions, 1)),
            features=None, latency_ms=0.0,
            correct=(None if s.answer is None else out.pred == s.answer),
        )
    except Exception:  # pragma: no cover
        return None


def paired_bootstrap_pvalue(a: Sequence[bool], b: Sequence[bool], n_boot=10000, seed=0) -> float:
    """Two-sided paired bootstrap p-value on accuracy difference (a - b)."""
    a = np.array([1.0 if x else 0.0 for x in a])
    b = np.array([1.0 if x else 0.0 for x in b])
    diff = a - b
    obs = diff.mean()
    if obs == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    # p-value under H0: mean difference = 0 (center the bootstrap distribution)
    centered = boots - boots.mean()
    p = np.mean(np.abs(centered) >= abs(obs))
    return float(p)


def evaluate_all(
    backend: MLLMBackend,
    datasets: Dict[str, List[Sample]],
    cfg: UAViBConfig,
    calibrator=None,
    include_baselines: bool = True,
    progress: bool = True,
) -> Dict[str, Dict]:
    """Return {method: {dataset: metrics, ..., '_all': metrics}}."""
    methods = {}
    uavib = UAViB(backend, cfg, calibrator=calibrator)
    methods["uavib"] = uavib
    if include_baselines:
        methods.update(build_baselines(cfg))

    out: Dict[str, Dict] = {}
    all_samples = [s for ds in datasets.values() for s in ds]
    for mname, method in methods.items():
        out[mname] = {}
        pooled: List[QueryResult] = []
        for dname, samples in datasets.items():
            res = run_method(method, backend, samples, progress=progress)
            out[mname][dname] = summarize(res)
            pooled.extend(res)
        out[mname]["_all"] = summarize(pooled)
        out[mname]["_correct"] = [bool(r.correct) for r in pooled]
    return out


def significance_table(results: Dict[str, Dict], reference: str = "uavib") -> Dict[str, float]:
    """Paired-bootstrap p-value of ``reference`` vs each other method (pooled)."""
    ref = results[reference]["_correct"]
    pvals = {}
    for m, r in results.items():
        if m == reference or "_correct" not in r:
            continue
        pvals[m] = paired_bootstrap_pvalue(ref, r["_correct"])
    return pvals


def evaluate_lodo(
    backend: MLLMBackend,
    test_datasets: Dict[str, List[Sample]],
    calibration_pool_fn,
    cfg: UAViBConfig,
) -> Dict[str, Dict]:
    """Leave-one-domain-out: fit calibrator on the two non-target domains, test
    on the held-out target. ``calibration_pool_fn(exclude_domains)`` returns the
    training pool. Returns {target_domain: {'uavib': metrics}}.
    """
    from .train import fit_calibrator

    by_domain: Dict[str, List[Sample]] = {}
    for ds in test_datasets.values():
        for s in ds:
            by_domain.setdefault(s.domain, []).append(s)

    out: Dict[str, Dict] = {}
    for target in by_domain:
        pool = calibration_pool_fn(exclude_domains=[target])
        head = fit_calibrator(backend, pool, cfg, verbose=False)
        uavib = UAViB(backend, cfg, calibrator=head)
        res = run_method(uavib, backend, by_domain[target])
        out[target] = {"uavib": summarize(res)}
    return out
