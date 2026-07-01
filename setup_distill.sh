#!/usr/bin/env bash
# Headless distillation pipeline: Stage 1 (soft labels) + Stage 2 (both students).
# Passed via --metadata=startup-script-url=... on a GCP Compute Engine VM.
# Results → gs://coronary-angio-v2/results/distillation/

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

LOG=/tmp/distill.log
exec > >(tee -a "$LOG") 2>&1

GCS_BUCKET="gs://coronary-angio-v2"

# Always shut down on exit (success or failure) so VM never idles
trap 'gsutil cp "$LOG" "${GCS_BUCKET}/results/distillation/distill_run.log" 2>/dev/null || true; shutdown -h now' EXIT

echo "============================================================"
echo " CA-SAM2 Distillation Pipeline — $(date)"
echo "============================================================"

# ── System packages ───────────────────────────────────────────────────────────
apt-get update -q
apt-get install -y -q git ffmpeg libgl1 libglib2.0-0

# ── Python dependencies ───────────────────────────────────────────────────────
pip3 install -q --upgrade pip
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub \
               Pillow numpy mobile-sam timm

# ── MedSAM2 ──────────────────────────────────────────────────────────────────
if [ ! -d /opt/MedSAM2 ]; then
    git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
fi
pip3 install -q -e /opt/MedSAM2

# ── This repo ─────────────────────────────────────────────────────────────────
if [ ! -d /opt/SAM2 ]; then
    git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2
fi

# ── Mimic /home/jupyter layout expected by distill scripts ───────────────────
mkdir -p /home/jupyter
ln -sfn /opt/MedSAM2 /home/jupyter/MedSAM2

# ── Download teacher from HuggingFace ────────────────────────────────────────
echo "Downloading teacher checkpoint from HuggingFace..."
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Elakiya17/CA-SAM2', filename='medsam2_arcade_v2.pt',
                local_dir='/home/jupyter')
print('Teacher downloaded.')
"

# ── Download ARCADE train + val data ─────────────────────────────────────────
echo "Downloading ARCADE data..."
mkdir -p /home/jupyter/arcade_train/images /home/jupyter/arcade_train/masks
mkdir -p /home/jupyter/arcade_val/images   /home/jupyter/arcade_val/masks

gsutil -m cp -r "${GCS_BUCKET}/datasets/arcade/train/images/*" /home/jupyter/arcade_train/images/
gsutil -m cp -r "${GCS_BUCKET}/datasets/arcade/train/masks/*"  /home/jupyter/arcade_train/masks/
gsutil -m cp -r "${GCS_BUCKET}/datasets/arcade/val/images/*"   /home/jupyter/arcade_val/images/
gsutil -m cp -r "${GCS_BUCKET}/datasets/arcade/val/masks/*"    /home/jupyter/arcade_val/masks/

# ── Stage 1: Generate soft labels from teacher ────────────────────────────────
echo "============================================================"
echo " Stage 1: Soft label generation — $(date)"
echo "============================================================"
export PYTHONPATH="/opt/MedSAM2:${PYTHONPATH:-}"
cd /opt/SAM2/distill
python3 stage1_softlabels.py

# ── Stage 2: Train MobileSAM student ─────────────────────────────────────────
echo "============================================================"
echo " Stage 2a: MobileSAM distillation — $(date)"
echo "============================================================"
python3 distill_student.py --student mobilesam --ablation 4

# ── Stage 3: Train RepViT-SAM student ────────────────────────────────────────
echo "============================================================"
echo " Stage 2b: RepViT-SAM distillation — $(date)"
echo "============================================================"
python3 distill_student.py --student repvitsam --ablation 4

# ── Upload log ────────────────────────────────────────────────────────────────
gsutil cp "$LOG" "${GCS_BUCKET}/results/distillation/distill_run.log"

echo "============================================================"
echo " All done. Shutting down. $(date)"
echo "============================================================"

shutdown -h now
