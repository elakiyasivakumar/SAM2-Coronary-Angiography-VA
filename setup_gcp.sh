#!/usr/bin/env bash
# Headless startup script for GCP Compute Engine (Deep Learning VM).
# Passed via --metadata=startup-script-url=... — runs automatically on boot as root.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

LOG=/tmp/ca-sam2.log
exec > >(tee -a "$LOG") 2>&1

GCS_BUCKET="gs://coronary-angio-v2"

# Always shut down on exit (success or failure) so VM never idles
trap 'gsutil cp "$LOG" "${GCS_BUCKET}/endpoint_results/run.log" 2>/dev/null || true; shutdown -h now' EXIT

echo "============================================================"
echo " CA-SAM2 Endpoint Evaluation — $(date)"
echo "============================================================"

# ── System packages ───────────────────────────────────────────────────────────
apt-get update -q
apt-get install -y -q git ffmpeg libgl1 libglib2.0-0

# ── Python dependencies ───────────────────────────────────────────────────────
pip3 install -q --upgrade pip
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub Pillow numpy

# ── MedSAM2 ──────────────────────────────────────────────────────────────────
if [ ! -d /opt/MedSAM2 ]; then
    git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
fi
pip3 install -q -e /opt/MedSAM2

# ── This repo ─────────────────────────────────────────────────────────────────
if [ ! -d /opt/SAM2 ]; then
    git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2
else
    git config --global --add safe.directory /opt/SAM2
    git -C /opt/SAM2 pull
fi

# ── Download checkpoint from GCS ─────────────────────────────────────────────
gsutil cp "${GCS_BUCKET}/checkpoints/medsam2_arcade_v2.pt" /tmp/medsam2_arcade_v2.pt

# ── Run evaluation ────────────────────────────────────────────────────────────
export PYTHONPATH="/opt/MedSAM2:${PYTHONPATH:-}"
cd /opt/SAM2

python3 eval_endpoints.py \
    --data_dir   /tmp/arcade_val \
    --output_dir /tmp/endpoint_results \
    --ckpt       /tmp/medsam2_arcade_v2.pt \
    --gcs_out    "${GCS_BUCKET}/endpoint_results"

# ── Upload log and shut down ──────────────────────────────────────────────────
gsutil cp "$LOG" "${GCS_BUCKET}/endpoint_results/run.log"

echo "============================================================"
echo " All done. Shutting down. $(date)"
echo "============================================================"

shutdown -h now
