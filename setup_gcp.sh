#!/usr/bin/env bash
# Headless startup script for GCP Compute Engine (Deep Learning VM + L4).
# Passed via --metadata=startup-script-url=... — runs automatically on boot.
# Logs to /tmp/ca-sam2.log and uploads to GCS when done. VM shuts down itself.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

LOG=/tmp/ca-sam2.log
exec > >(tee -a "$LOG") 2>&1

GCS_BUCKET="gs://coronary-angio-v2"

echo "============================================================"
echo " CA-SAM2 Endpoint Evaluation — $(date)"
echo "============================================================"

# ── System packages ───────────────────────────────────────────────────────────
apt-get update -q
apt-get install -y -q git ffmpeg libgl1 libglib2.0-0

# ── Python dependencies ───────────────────────────────────────────────────────
# DL VM already has torch; install only what's missing
pip install -q --upgrade pip
pip install -q scikit-image scipy opencv-python-headless huggingface_hub Pillow numpy

# ── MedSAM2 ──────────────────────────────────────────────────────────────────
if [ ! -d /opt/MedSAM2 ]; then
    git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
fi
pip install -q -e /opt/MedSAM2

# ── This repo ─────────────────────────────────────────────────────────────────
if [ ! -d /opt/SAM2 ]; then
    git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2
else
    git -C /opt/SAM2 pull
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
cd /opt/SAM2

# Override MedSAM2 path so eval_endpoints.py can find it
export PYTHONPATH="/opt/MedSAM2:${PYTHONPATH:-}"

python3 eval_endpoints.py \
    --data_dir   /tmp/arcade_val \
    --output_dir /tmp/endpoint_results \
    --ckpt       /tmp/medsam2_arcade_v2.pt \
    --gcs_out    "${GCS_BUCKET}/endpoint_results"

# ── Upload log ────────────────────────────────────────────────────────────────
gsutil cp "$LOG" "${GCS_BUCKET}/endpoint_results/run.log"

echo "============================================================"
echo " All done. Shutting down. $(date)"
echo "============================================================"

# Auto-shutdown to avoid idle charges
shutdown -h now
