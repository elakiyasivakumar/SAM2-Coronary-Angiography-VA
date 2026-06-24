#!/bin/bash
# Master runner for knowledge distillation pipeline.
# Runs on a fresh GCP n1-standard-4 + T4 instance.
# All results go to gs://coronary-angio-v2/

set -e
BUCKET="gs://coronary-angio-v2"
WORK="/home/jupyter"
LOG="$WORK/distill_run.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── 0. Environment setup ──────────────────────────────────────────────────────
log "=== Setting up environment ==="

# MedSAM2
if [ ! -d "$WORK/MedSAM2" ]; then
    log "Cloning MedSAM2..."
    cd "$WORK"
    git clone https://github.com/bowang-lab/MedSAM2.git
    cd MedSAM2
    pip install -e . -q
    bash download.sh
fi

# Dependencies
log "Installing packages..."
pip install -q mobile-sam timm

# Download distillation scripts from GCS
log "Downloading scripts..."
mkdir -p "$WORK/distill"
gsutil -m cp "$BUCKET/distill/*.py" "$WORK/distill/"

# Download ARCADE train data
log "Downloading ARCADE train data..."
mkdir -p "$WORK/arcade_train/images" "$WORK/arcade_train/masks"
gsutil -m cp -r "$BUCKET/datasets/arcade/train/images/*" "$WORK/arcade_train/images/"
gsutil -m cp -r "$BUCKET/datasets/arcade/train/masks/*"  "$WORK/arcade_train/masks/"

# Download ARCADE val data
log "Downloading ARCADE val data..."
mkdir -p "$WORK/arcade_val/images" "$WORK/arcade_val/masks"
gsutil -m cp -r "$BUCKET/datasets/arcade/val/images/*" "$WORK/arcade_val/images/"
gsutil -m cp -r "$BUCKET/datasets/arcade/val/masks/*"  "$WORK/arcade_val/masks/"

# Download fine-tuned teacher checkpoint from Hugging Face
log "Downloading teacher checkpoint from HuggingFace..."
pip install -q huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='Elakiya17/CA-SAM2', filename='medsam2_arcade_v2.pt',
                       local_dir='/home/jupyter')
print(f'Downloaded to {path}')
"

# ── 1. Stage 1: Soft labels ───────────────────────────────────────────────────
log "=== Stage 1: Generating soft labels ==="
cd "$WORK/distill"
python stage1_softlabels.py 2>&1 | tee -a "$LOG"

# ── 2. Stage 2: MobileSAM ablations ──────────────────────────────────────────
log "=== Stage 2: MobileSAM ablations ==="
for abl in 1 2 3 4; do
    log "  MobileSAM ablation $abl..."
    python distill_student.py --student mobilesam --ablation $abl 2>&1 | tee -a "$LOG"
done

# ── 3. Stage 3: RepViT-SAM ablations ─────────────────────────────────────────
log "=== Stage 3: RepViT-SAM ablations ==="
for abl in 1 2 3 4; do
    log "  RepViT-SAM ablation $abl..."
    python distill_student.py --student repvitsam --ablation $abl 2>&1 | tee -a "$LOG"
done

# ── 4. Upload log ─────────────────────────────────────────────────────────────
log "=== All done. Uploading log ==="
gsutil cp "$LOG" "$BUCKET/results/distillation/distill_run.log"

log "Complete. Results at $BUCKET/results/distillation/"

# ── 5. Self-terminate instance ────────────────────────────────────────────────
INSTANCE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" \
           -H "Metadata-Flavor: Google" || echo "")
ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" \
       -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}' || echo "")

if [ -n "$INSTANCE" ] && [ -n "$ZONE" ]; then
    log "Shutting down instance $INSTANCE in zone $ZONE..."
    gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
fi
