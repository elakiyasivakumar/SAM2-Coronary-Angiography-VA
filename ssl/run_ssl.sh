#!/bin/bash
# Stage 2 SSL runner for GCP (A100 recommended, V100 fallback).
# Self-terminates instance on completion.
set -e

GCS_BUCKET="coronary-angio-v2"
DATA_DIR="/tmp/coronary_data"
CKPT_DIR="/tmp/ssl_checkpoints"
MEDSAM2_DIR="/tmp/MedSAM2"
TEACHER_CKPT="/tmp/medsam2_arcade_v2.pt"
SSL_DIR="$(dirname "$0")"

echo "=== SSL Phase 2 — $(date) ==="

# Dependencies
pip install -q torch torchvision timm scikit-learn google-cloud-storage huggingface_hub opencv-python-headless

# MedSAM2 install
if [ ! -d "$MEDSAM2_DIR" ]; then
  git clone https://github.com/bowang-lab/MedSAM2.git "$MEDSAM2_DIR"
  pip install -q -e "$MEDSAM2_DIR"
fi

# Teacher checkpoint
python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id='Elakiya17/CA-SAM2',
    filename='medsam2_arcade_v2.pt',
    local_dir='/tmp',
)
print(f'Teacher checkpoint: {path}')
PYEOF

# Sync CoronaryDominance data from GCS
mkdir -p "$DATA_DIR"
echo "Syncing data from GCS (this may take a while)..."
gsutil -m rsync -r "gs://$GCS_BUCKET/coronary_dominance/" "$DATA_DIR/"
echo "Data sync complete."

mkdir -p "$CKPT_DIR"

# ── Stage 2a: SSL Pre-training ────────────────────────────────────────────────
echo ""
echo "=== Stage 2a: V-JEPA SSL Pre-training ==="
python3 "$SSL_DIR/pretrain.py" \
  --data_dir     "$DATA_DIR" \
  --teacher_ckpt "$TEACHER_CKPT" \
  --output_dir   "$CKPT_DIR" \
  --epochs       50 \
  --batch_size   16 \
  --lr           1.5e-4 \
  --warmup_epochs 10 \
  --gcs_bucket   "$GCS_BUCKET"

echo "Stage 2a complete. Best encoder: $CKPT_DIR/best_encoder.pt"

# ── Stage 2b: Downstream Fine-tuning ─────────────────────────────────────────
echo ""
echo "=== Stage 2b: Downstream Fine-tuning ==="
mkdir -p "$CKPT_DIR/downstream"

for TASK in occlusion acs collaterals; do
  echo ""
  echo "--- Task: $TASK ---"
  python3 "$SSL_DIR/finetune.py" \
    --data_dir         "$DATA_DIR" \
    --ssl_encoder_ckpt "$CKPT_DIR/best_encoder.pt" \
    --task             "$TASK" \
    --output_dir       "$CKPT_DIR/downstream" \
    --epochs           30 \
    --batch_size       32

  python3 "$SSL_DIR/eval.py" \
    --data_dir         "$DATA_DIR" \
    --ssl_encoder_ckpt "$CKPT_DIR/best_encoder.pt" \
    --head_ckpt        "$CKPT_DIR/downstream/best_${TASK}.pt" \
    --task             "$TASK"

  gsutil cp "$CKPT_DIR/downstream/best_${TASK}.pt" \
    "gs://$GCS_BUCKET/ssl/downstream/best_${TASK}.pt"
done

echo ""
echo "=== All stages complete — $(date) ==="

# Self-terminate
INSTANCE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" \
             -H "Metadata-Flavor: Google")
ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" \
         -H "Metadata-Flavor: Google" | awk -F'/' '{print $NF}')
echo "Terminating $INSTANCE in $ZONE"
gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
