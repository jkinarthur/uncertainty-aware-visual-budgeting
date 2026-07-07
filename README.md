# UAViB — Uncertainty-Calibrated Adaptive Vision-Token Budgeting

Reference implementation of **UAViB**, a plug-and-play, inference-time framework
that lets a **frozen** multimodal LLM decide, from its own **calibrated
predictive uncertainty**, *how many* vision tokens to spend and *where* to spend
them on high-resolution structured imagery (medical, remote-sensing, industrial).

> This repository accompanies the paper *"When to Look Closer: Uncertainty-
> Calibrated Adaptive Vision-Token Budgeting for Efficient and Reliable
> Multimodal LLMs on High-Resolution Structured Imagery."*

## Why it runs anywhere

The full pipeline — coarse pass, uncertainty estimation, region attribution,
budgeting, progressive refinement, the calibration head, all baselines, and the
metric suite — runs on a **CPU laptop with no model weights** via a faithful
`DummyBackend` simulator. Swap in the real **Qwen2.5-VL-7B** or **LLaVA-NeXT**
backends on a GPU/AWS box and the numbers become measured instead of simulated.

## Install

```bash
# CPU / dummy backend (everything runs)
pip install -e .

# GPU / real MLLM backends (on AWS)
pip install -e ".[gpu,viz]"
```

## Quickstart (no GPU)

```bash
# End-to-end smoke test in seconds
python scripts/smoke_test.py

# Train the calibrator, then evaluate UAViB + all baselines
python scripts/train_calibrator.py --backend dummy
python scripts/run_eval.py --backend dummy
```

`run_eval` prints a pooled results table (accuracy, tokens, latency, ECE, NLL,
RC-AUC, AUROC, paired-bootstrap p-value) and writes `outputs/results.json`.

## Run with a real MLLM (GPU / AWS)

```bash
bash deploy/aws_setup.sh                 # torch + deps + HF cache on a big disk
bash deploy/run_on_aws.sh qwen sim       # GPU dry-run (synthetic data, no datasets)
bash deploy/run_on_aws.sh qwen           # full run on prepared real datasets
bash deploy/run_on_aws.sh llava          # second backbone
```

## Method map (paper → code)

| Paper component | Module |
|---|---|
| Coarse pass + uncertainty (entropy, margin, agreement, semantic entropy) | `uavib/uncertainty.py` |
| Region attribution (attention + batched occlusion sensitivity) | `uavib/attribution.py` |
| Uncertainty-driven budget + water-filling redistribution | `uavib/budget.py` |
| Domain-agnostic calibration head (NLL + soft-ECE, temperature) | `uavib/calibration.py` |
| Progressive refinement loop | `uavib/pipeline.py` |
| Backends (frozen MLLM wrappers) | `uavib/backends/{dummy,qwen,llava}.py` |
| Baselines (fixed, ToMe, FastV, DynamicViT, PrATo, GRASP, DualComp, oracle, controls) | `uavib/baselines/methods.py` |
| Metrics (ECE, NLL, Brier, RC-AUC, AUROC) | `uavib/metrics.py` |
| Evaluation, LODO transfer, significance | `uavib/eval.py` |

## Datasets

Synthetic data is generated automatically for the dummy backend. For real runs,
place each dataset under `data/raw/<subdir>/` with a `test.jsonl` (and
`calib.jsonl`) manifest, one JSON per line:

```json
{"image": "img/0001.png", "question": "Is there a lesion?",
 "candidates": ["yes", "no"], "answer": "yes", "roi": [[120, 88, 210, 175]]}
```

Supported: `rsvqa-lr`, `rsvqa-hr`, `rsivqa`, `xlrs-bench`, `acdc`, `isic`,
`mvtec-ad` (see `uavib/data/real.py`). Do **not** commit weights or datasets —
`.gitignore` excludes them.

## Reproducing the paper tables

After a real run, convert results to LaTeX rows and paste them into
`paper/main.tex`:

```bash
python scripts/results_to_latex.py --results outputs/results_qwen.json \
                                   --out outputs/tables_qwen.tex
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
