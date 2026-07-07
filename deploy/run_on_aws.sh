#!/usr/bin/env bash
# UAViB — run training + evaluation on AWS with a real MLLM backend.
#   bash deploy/run_on_aws.sh qwen    # or: llava
#
# Assumes deploy/aws_setup.sh has been run and the venv is active, and that the
# real datasets are prepared under data/raw (see README: Datasets). To do a GPU
# smoke run without datasets, pass a 3rd arg: bash deploy/run_on_aws.sh qwen sim
set -euo pipefail

BACKEND="${1:-qwen}"
MODE="${2:-real}"          # real | sim
CONFIG="configs/${BACKEND}.yaml"
CALIB="artifacts/calibrator_${BACKEND}.json"
OUT="outputs/results_${BACKEND}.json"

mkdir -p artifacts outputs

if [[ "$MODE" == "sim" ]]; then
  echo "==> GPU dry-run with synthetic data (no datasets needed)"
  python scripts/train_calibrator.py --backend dummy --config "$CONFIG" --out "$CALIB"
  python scripts/run_eval.py --backend dummy --config "$CONFIG" --calibrator "$CALIB" --out "$OUT"
  exit 0
fi

echo "==> [1/3] Train calibrator on domain-mixed real data ($BACKEND)"
python scripts/train_calibrator.py \
  --backend "$BACKEND" --config "$CONFIG" --device cuda \
  --source real --data-root data/raw --out "$CALIB"

echo "==> [2/3] Evaluate UAViB + baselines on all datasets ($BACKEND)"
python scripts/run_eval.py \
  --backend "$BACKEND" --config "$CONFIG" --device cuda \
  --source real --data-root data/raw --calibrator "$CALIB" --out "$OUT"

echo "==> [3/3] Emit LaTeX table rows for the paper"
python scripts/results_to_latex.py --results "$OUT" --out "outputs/tables_${BACKEND}.tex"

echo "==> Done. Results in $OUT ; tables in outputs/tables_${BACKEND}.tex"
