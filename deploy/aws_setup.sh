#!/usr/bin/env bash
# UAViB — AWS GPU instance setup (Ubuntu 22.04 DLAMI / plain Ubuntu + NVIDIA).
# Run on the instance after cloning the repo:
#   bash deploy/aws_setup.sh
#
# Recommended instance: g5.2xlarge or larger (A10G 24GB) for a 7B MLLM;
# an A100 (p4d/p5) matches the paper's profiling. Use a >=100GB EBS root
# volume — 7B weights + HF cache do NOT fit on the default 39GB disk.
set -euo pipefail

echo "==> Checking disk (need >=100GB free for weights + cache)"
df -h /

echo "==> System packages"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git

echo "==> Python virtual environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "==> Install PyTorch (CUDA 12.1 build). Adjust the index-url to your CUDA."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> Install UAViB (gpu extra) + project deps"
pip install -e ".[gpu,viz]"

echo "==> Point HF cache at a large volume (avoids ENOSPC on small root disks)"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
mkdir -p "$HF_HOME"
echo "export HF_HOME=$HF_HOME" >> "$HOME/.bashrc"

echo "==> Verify GPU is visible to torch"
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "==> Done. Next:"
echo "    source .venv/bin/activate"
echo "    bash deploy/run_on_aws.sh qwen"
