#!/bin/bash
# SAM2.1 Hiera-Tiny fine-tuned at 1024x1024 on ARCADE coronary angiography
# Starts from SAM2.1 BASE pretrained weights (not 512-tuned CA-SAM2)
# Answers: does resolution explain MobileSAM v4's 0.806 advantage over teacher?
set -euo pipefail
LOG=/tmp/sam2_1024.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/sam2_1024_run.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== SAM2.1 Hiera-Tiny 1024x1024 fine-tune $(date) ==="

apt-get install -y -q git python3-pip
pip3 install -q -U pip setuptools wheel
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub timm

git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
pip3 install -q -e /opt/MedSAM2 --no-build-isolation
git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2

mkdir -p /home/jupyter
ln -sfn /opt/MedSAM2 /home/jupyter/MedSAM2

# SAM2.1 base pretrained weights (1024x1024 native)
python3 -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(repo_id='facebook/sam2.1-hiera-tiny',
                filename='sam2.1_hiera_tiny.pt',
                local_dir='/home/jupyter')
print('SAM2.1 base weights ready:', os.path.getsize('/home/jupyter/sam2.1_hiera_tiny.pt')//1e6, 'MB')
"

mkdir -p /home/jupyter/arcade_train/images /home/jupyter/arcade_train/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/train/images/* /home/jupyter/arcade_train/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/train/masks/*  /home/jupyter/arcade_train/masks/

mkdir -p /home/jupyter/arcade_test/images /home/jupyter/arcade_test/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/images/* /home/jupyter/arcade_test/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/masks/*  /home/jupyter/arcade_test/masks/

export PYTHONPATH=/opt/MedSAM2
export MAX_EPOCHS=50
export ES_PATIENCE=10
export BATCH_SIZE=2
export MODEL_VERSION=sam2_fluoroscopy_1024
export HF_REPO=Elakiya17/fluoroscopy-sam2

cd /opt/SAM2
python3 train_sam2_1024.py

echo "Done $(date)"
