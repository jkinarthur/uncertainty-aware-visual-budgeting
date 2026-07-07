"""End-to-end smoke test: train a calibrator and evaluate UAViB + baselines on
the synthetic benchmark with the dummy backend. Runs in seconds on CPU and
verifies the whole pipeline is wired correctly.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavib.backends import DummyBackend            # noqa: E402
from uavib.config import UAViBConfig                # noqa: E402
from uavib.data import calibration_pool, load_dataset, DATASETS  # noqa: E402
from uavib.eval import evaluate_all, significance_table          # noqa: E402
from uavib.train import fit_calibrator              # noqa: E402


def main() -> int:
    cfg = UAViBConfig(seed=0)
    backend = DummyBackend()

    print("1) Fitting calibrator on domain-mixed pool...")
    pool = calibration_pool(n_per_dataset=120, seed=0)
    head = fit_calibrator(backend, pool, cfg, verbose=True)

    print("\n2) Building test sets (150 samples/dataset)...")
    datasets = {name: load_dataset(name, n=150, seed=1) for name in DATASETS}

    print("3) Evaluating UAViB + baselines...")
    results = evaluate_all(backend, datasets, cfg, calibrator=head, progress=False)
    pvals = significance_table(results)

    from uavib.cli import _print_table
    _print_table(results, pvals)

    uavib = results["uavib"]["_all"]
    full = results["full-resolution"]["_all"]
    fixed = results["fixed-512"]["_all"]
    print("\n=== sanity checks ===")
    print(f"UAViB tokens ({uavib['avg_tokens']:.0f}) < full-res tokens "
          f"({full['avg_tokens']:.0f}): {uavib['avg_tokens'] < full['avg_tokens']}")
    print(f"UAViB ECE ({uavib['ece']:.3f}) < fixed-512 ECE "
          f"({fixed['ece']:.3f}): {uavib['ece'] < fixed['ece']}")
    print(f"UAViB acc ({uavib['accuracy']:.1f}) >= fixed-512 acc "
          f"({fixed['accuracy']:.1f}): {uavib['accuracy'] >= fixed['accuracy']}")
    ok = (uavib["avg_tokens"] < full["avg_tokens"]) and (uavib["ece"] < fixed["ece"])
    print("\nSMOKE TEST:", "PASS" if ok else "CHECK OUTPUT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
