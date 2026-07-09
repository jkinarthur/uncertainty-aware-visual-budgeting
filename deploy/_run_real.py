"""One-shot real run: load Qwen once, fit calibrator, evaluate, save results.
Amortizes the ~210s model load across both phases. Sample counts via env vars:
  CALIB_N (default 40), TEST_N (default 24), BASELINES (1/0, default 1).
"""
import os, json, time
from uavib.backends import get_backend
from uavib.config import UAViBConfig
from uavib.data import calibration_pool, load_dataset
from uavib.train import fit_calibrator
from uavib.eval import evaluate_all, significance_table

CALIB_N = int(os.environ.get("CALIB_N", "40"))
TEST_N = int(os.environ.get("TEST_N", "24"))
BASELINES = os.environ.get("BASELINES", "1") == "1"
DATASET = os.environ.get("DATASET", "rsvqa-lr")
CONFIG = os.environ.get("CONFIG", "configs/qwen_rsvqa.yaml")
OUT = os.environ.get("OUT", "outputs/results_qwen_real.json")
CALIB_OUT = os.environ.get("CALIB_OUT", "artifacts/calibrator_qwen.json")

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

os.makedirs("artifacts", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

cfg = UAViBConfig.from_yaml(CONFIG)
cfg.device = "cuda"
log(f"Config: coarse={cfg.coarse_tokens} b=[{cfg.b_min},{cfg.b_max}] grid={cfg.grid_h}x{cfg.grid_w}")

t0 = time.time()
log("Loading Qwen backend (cold load ~200s)...")
backend = get_backend("qwen", device="cuda")
backend._ensure_loaded()
log(f"Model ready in {time.time()-t0:.0f}s")

# --- Calibrator ---
pool = calibration_pool(source="real", n_per_dataset=CALIB_N, seed=0,
                        data_root="data/raw", datasets=[DATASET])
log(f"Fitting calibrator on {len(pool)} calib samples...")
tc = time.time()
head = fit_calibrator(backend, pool, cfg, verbose=True)
head.save(CALIB_OUT)
log(f"Calibrator saved -> {CALIB_OUT} ({time.time()-tc:.0f}s)")

# --- Eval ---
test = load_dataset(DATASET, split="test", n=TEST_N, seed=0,
                    source="real", data_root="data/raw", limit=TEST_N)
log(f"Evaluating on {len(test)} test samples (baselines={BASELINES})...")
te = time.time()
results = evaluate_all(backend, {DATASET: test}, cfg, calibrator=head,
                       include_baselines=BASELINES, progress=True)
pvals = significance_table(results)
log(f"Eval done ({time.time()-te:.0f}s)")

serializable = {m: {k: v for k, v in r.items() if k != "_correct"}
                for m, r in results.items()}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump({"results": serializable, "pvalues": pvals}, fh, indent=2)

# summary table
print("\n=== Pooled results ===", flush=True)
hdr = f"{'method':<18}{'acc':>7}{'tok':>8}{'ECE':>7}{'AUROC':>7}{'RC-AUC':>8}"
print(hdr); print("-"*len(hdr))
for m in sorted(results, key=lambda x: -results[x]["_all"]["accuracy"]):
    a = results[m]["_all"]
    print(f"{m:<18}{a['accuracy']:>7.1f}{a['avg_tokens']:>8.0f}{a['ece']:>7.3f}"
          f"{a['auroc']:>7.3f}{a['rc_auc']:>8.3f}", flush=True)
log(f"Wrote {OUT}. TOTAL {time.time()-t0:.0f}s")
print("RUN_REAL_DONE", flush=True)
