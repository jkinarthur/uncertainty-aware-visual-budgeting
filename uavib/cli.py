"""Command-line entry points (installed as ``uavib-eval`` /
``uavib-train-calibrator``). Also runnable as ``python -m uavib.cli ...``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from .backends import get_backend
from .calibration import CalibrationHead
from .config import UAViBConfig
from .data import DATASETS, calibration_pool, load_dataset
from .eval import evaluate_all, significance_table
from .train import fit_calibrator
from .types import Sample


def _load_config(args) -> UAViBConfig:
    cfg = UAViBConfig.from_yaml(args.config) if args.config else UAViBConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device:
        cfg.device = args.device
    return cfg


def _datasets(args) -> Dict[str, List[Sample]]:
    names = args.datasets or DATASETS
    ds = {}
    for name in names:
        ds[name] = load_dataset(
            name, split=args.split, n=args.n, seed=args.seed or 0,
            source=args.source, data_root=args.data_root, limit=args.limit,
        )
    return ds


def main_train_calibrator(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the UAViB calibration head.")
    ap.add_argument("--backend", default="dummy")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n-per-dataset", type=int, default=200)
    ap.add_argument("--out", default="artifacts/calibrator.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args(argv)

    cfg = _load_config(args)
    kwargs = {"model_id": args.model_id} if args.model_id else {}
    if args.backend != "dummy" and args.device:
        kwargs["device"] = args.device
    backend = get_backend(args.backend, **kwargs)

    pool = calibration_pool(source=args.source, n_per_dataset=args.n_per_dataset,
                            seed=args.seed, data_root=args.data_root)
    print(f"Fitting calibrator on {len(pool)} domain-mixed samples...")
    head = fit_calibrator(backend, pool, cfg, verbose=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    head.save(args.out)
    print(f"Saved calibrator -> {args.out}")


def main_eval(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate UAViB and baselines.")
    ap.add_argument("--backend", default="dummy")
    ap.add_argument("--config", default=None)
    ap.add_argument("--calibrator", default="artifacts/calibrator.json")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=400, help="synthetic samples/dataset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--out", default="outputs/results.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args(argv)

    cfg = _load_config(args)
    kwargs = {"model_id": args.model_id} if args.model_id else {}
    if args.backend != "dummy" and args.device:
        kwargs["device"] = args.device
    backend = get_backend(args.backend, **kwargs)

    calibrator = None
    if args.calibrator and os.path.exists(args.calibrator):
        calibrator = CalibrationHead.load(args.calibrator)
        print(f"Loaded calibrator from {args.calibrator}")
    else:
        print("No calibrator found; using identity fallback (train one first).")

    datasets = _datasets(args)
    results = evaluate_all(
        backend, datasets, cfg, calibrator=calibrator,
        include_baselines=not args.no_baselines, progress=True,
    )
    pvals = significance_table(results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    serializable = {m: {k: v for k, v in r.items() if k != "_correct"}
                    for m, r in results.items()}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": serializable, "pvalues": pvals, "config": vars(args)},
                  fh, indent=2)

    _print_table(results, pvals)
    print(f"\nWrote {args.out}")


def _print_table(results, pvals) -> None:
    print("\n=== Pooled results (all datasets) ===")
    header = f"{'method':<18}{'acc':>7}{'tok':>8}{'lat_ms':>8}{'ECE':>7}{'NLL':>7}{'RC-AUC':>8}{'AUROC':>7}{'p':>9}"
    print(header)
    print("-" * len(header))
    order = sorted(results, key=lambda m: -results[m]["_all"]["accuracy"])
    for m in order:
        a = results[m]["_all"]
        p = pvals.get(m, float("nan"))
        pstr = "-" if m == "uavib" else (f"{p:.1e}" if p == p else "-")
        print(f"{m:<18}{a['accuracy']:>7.1f}{a['avg_tokens']:>8.0f}"
              f"{a['latency_ms']:>8.1f}{a['ece']:>7.3f}{a['nll']:>7.2f}"
              f"{a['rc_auc']:>8.3f}{a['auroc']:>7.3f}{pstr:>9}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        main_train_calibrator(sys.argv[2:])
    else:
        main_eval(sys.argv[1:] if sys.argv[1:] and sys.argv[1] != "eval" else sys.argv[2:])
