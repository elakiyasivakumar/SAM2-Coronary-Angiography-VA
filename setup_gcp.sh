#!/usr/bin/env bash
# Run this on a fresh GCP Vertex AI Workbench instance (L4 GPU recommended).
# Clones the repo, installs all dependencies, then runs the endpoint evaluation.
#
# One-liner to kick off from the instance terminal:
#   curl -sSL https://raw.githubusercontent.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA/main/setup_gcp.sh | bash

set -euo pipefail

echo "============================================================"
echo " CA-SAM2 Endpoint Evaluation — GCP Setup"
echo "============================================================"

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -q
apt-get install -y -q git ffmpeg libgl1 libglib2.0-0

# ── Python dependencies ───────────────────────────────────────────────────────
pip install -q --upgrade pip
pip install -q \
    torch torchvision \
    scikit-image scipy \
    opencv-python-headless \
    huggingface_hub \
    Pillow numpy

# ── MedSAM2 ──────────────────────────────────────────────────────────────────
if [ ! -d "$HOME/MedSAM2" ]; then
    echo "Cloning MedSAM2..."
    git clone https://github.com/bowang-lab/MedSAM2.git "$HOME/MedSAM2"
fi
pip install -q -e "$HOME/MedSAM2"

# ── This repo ─────────────────────────────────────────────────────────────────
if [ ! -d "$HOME/SAM2" ]; then
    echo "Cloning SAM2-Coronary-Angiography-VA..."
    git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git "$HOME/SAM2"
else
    echo "Updating SAM2-Coronary-Angiography-VA..."
    git -C "$HOME/SAM2" pull
fi

cd "$HOME/SAM2"

# ── Run evaluation ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Running clinical endpoint evaluation on ARCADE val set"
echo "============================================================"
python eval_endpoints.py \
    --data_dir   /home/jupyter/arcade_val \
    --output_dir /home/jupyter/endpoint_results \
    --ckpt       /home/jupyter/medsam2_arcade_v2.pt \
    --gcs_out    gs://coronary-angio-v2/endpoint_results

echo ""
echo "============================================================"
echo " All done. Results at:"
echo "   gs://coronary-angio-v2/endpoint_results/endpoint_results.json"
echo "   gs://coronary-angio-v2/endpoint_results/overlays/"
echo "============================================================"
