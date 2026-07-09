"""Run multiple UAViB config ablations with a single backend load.

Env:
  DATASET=xlrs-bench
  CONFIGS=configs/a.yaml,configs/b.yaml
  CALIB_N=60
  TEST_N=120
  OUT=outputs/ablation_xlrs_uavib.json
"""
import json
import os
import time

from uavib.backends import get_backend
from uavib.config import UAViBConfig
from uavib.data import calibration_pool, load_dataset
from uavib.eval import evaluate_all
from uavib.train import fit_calibrator

DATASET = os.environ.get("DATASET", "xlrs-bench")
CONFIGS = [c.strip() for c in os.environ.get("CONFIGS", "").split(",") if c.strip()]
CALIB_N = int(os.environ.get("CALIB_N", "60"))
TEST_N = int(os.environ.get("TEST_N", "120"))
OUT = os.environ.get("OUT", "outputs/ablation_xlrs_uavib.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if not CONFIGS:
    raise ValueError("CONFIGS is empty")

os.makedirs("outputs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

log("Loading backend once...")
t0 = time.time()
backend = get_backend("qwen", device="cuda")
backend._ensure_loaded()
log(f"Backend ready in {time.time() - t0:.0f}s")

test = load_dataset(
    DATASET,
    split="test",
    n=TEST_N,
    seed=0,
    source="real",
    data_root="data/raw",
    limit=TEST_N,
)
log(f"Loaded test set: {len(test)}")

summary = {}
for cfg_path in CONFIGS:
    cfg = UAViBConfig.from_yaml(cfg_path)
    cfg.device = "cuda"
    tag = os.path.splitext(os.path.basename(cfg_path))[0]

    pool = calibration_pool(
        source="real",
        n_per_dataset=CALIB_N,
        seed=0,
        data_root="data/raw",
        datasets=[DATASET],
    )
    log(f"[{tag}] fit calibrator on {len(pool)}")
    head = fit_calibrator(backend, pool, cfg, verbose=False)

    log(f"[{tag}] eval on {len(test)}")
    t1 = time.time()
    res = evaluate_all(
        backend,
        {DATASET: test},
        cfg,
        calibrator=head,
        include_baselines=False,
        progress=True,
    )
    m = res["uavib"]["_all"]
    summary[tag] = {
        "config": cfg_path,
        "accuracy": m["accuracy"],
        "avg_tokens": m["avg_tokens"],
        "ece": m["ece"],
        "nll": m["nll"],
        "brier": m["brier"],
        "rc_auc": m["rc_auc"],
        "auroc": m["auroc"],
        "n": m["n"],
        "eval_seconds": time.time() - t1,
    }
    log(
        f"[{tag}] acc={m['accuracy']:.2f} tok={m['avg_tokens']:.1f} "
        f"AUROC={m['auroc']:.3f} RC-AUC={m['rc_auc']:.3f}"
    )

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

print("=== ABLATION SUMMARY ===", flush=True)
for tag, m in summary.items():
    print(
        f"{tag}: acc={m['accuracy']:.2f} tok={m['avg_tokens']:.1f} "
        f"ece={m['ece']:.3f} auroc={m['auroc']:.3f} rc_auc={m['rc_auc']:.3f}",
        flush=True,
    )
print(f"WROTE {OUT}", flush=True)
print("ABLATION_DONE", flush=True)
