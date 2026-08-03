#!/bin/bash
# SAM2.1 Hiera-Tiny fine-tuned at 1024x1024 on ARCADE coronary angiography.
# Uses Meta's facebookresearch/sam2 (NOT MedSAM2) — native 1024 config.
# Protocol: v2 (partial unfreeze, discriminative LRs, clDice, 5x aug).
# Goal: isolate whether 1024x1024 resolution explains MobileSAM v4's 0.806 Dice advantage.
set -euo pipefail
LOG=/tmp/sam2_1024.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/sam2_1024_run.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== SAM2.1 Hiera-Tiny 1024x1024 fine-tune (Meta, v2 protocol) $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "CPU: $(nproc) cores | RAM: $(free -h | awk '/^Mem:/{print $2}')"

# ── System deps ───────────────────────────────────────────────────────────────
apt-get install -y -q git python3-pip
pip3 install -q -U pip setuptools wheel
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub timm

# ── Meta SAM2 (facebookresearch/sam2, image_size: 1024) ───────────────────────
git clone https://github.com/facebookresearch/sam2.git /opt/sam2_meta
pip3 install -e /opt/sam2_meta --no-build-isolation

# Our code
git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2

mkdir -p /home/jupyter

# ── SAM2.1 Hiera-Tiny base weights (native 1024x1024) ────────────────────────
python3 -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(repo_id='facebook/sam2.1-hiera-tiny',
                filename='sam2.1_hiera_tiny.pt',
                local_dir='/home/jupyter')
size_mb = os.path.getsize('/home/jupyter/sam2.1_hiera_tiny.pt') // 1_000_000
print('SAM2.1 base weights ready:', size_mb, 'MB')
"

# ── ARCADE data ───────────────────────────────────────────────────────────────
mkdir -p /home/jupyter/arcade_train/images /home/jupyter/arcade_train/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/train/images/* /home/jupyter/arcade_train/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/train/masks/*  /home/jupyter/arcade_train/masks/

mkdir -p /home/jupyter/arcade_test/images /home/jupyter/arcade_test/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/images/* /home/jupyter/arcade_test/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/masks/*  /home/jupyter/arcade_test/masks/

echo "Data sizes: train=$(ls /home/jupyter/arcade_train/images | wc -l) test=$(ls /home/jupyter/arcade_test/images | wc -l)"

# ── Training ──────────────────────────────────────────────────────────────────
export PYTHONPATH=/opt/sam2_meta
export MAX_EPOCHS=50
export ES_PATIENCE=10
export BATCH_SIZE=4
export MODEL_VERSION=sam2_fluoroscopy_1024
export HF_REPO=Elakiya17/fluoroscopy-sam2
# Token injected at VM creation via --metadata hf-token=<value> (never commit the raw token)
export HUGGING_FACE_HUB_TOKEN=$(curl -sf \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token" \
  -H "Metadata-Flavor: Google" || echo "")

cd /opt/SAM2
python3 train_sam2_1024.py

echo "=== Done $(date) ==="
