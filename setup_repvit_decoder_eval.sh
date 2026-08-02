#!/bin/bash
# RepViT-SAM + CA-SAM2 decoder transplant, NO ARCADE fine-tuning, eval on 200 test images
# Tests: does the decoder transplant alone (without any training) give RepViT useful performance?
set -euo pipefail
LOG=/tmp/repvit_decoder_eval.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/repvit_decoder_eval.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== RepViT + CA-SAM2 decoder eval (no training) $(date) ==="

apt-get install -y -q git python3-pip
pip3 install -q -U pip setuptools wheel
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub timm
git clone https://github.com/ChaoningZhang/MobileSAM.git /opt/MobileSAM
pip3 install -q -e /opt/MobileSAM --no-build-isolation

git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
pip3 install -q -e /opt/MedSAM2
git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2

mkdir -p /home/jupyter
ln -sfn /opt/MedSAM2 /home/jupyter/MedSAM2

# Teacher checkpoint for decoder transplant
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Elakiya17/CA-SAM2', filename='medsam2_arcade_v2.pt', local_dir='/home/jupyter')
print('Teacher checkpoint ready.')
"

# MobileSAM weights (needed to init SAM decoder inside RepViTSAM)
wget -q -O /tmp/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# Test images only — no training data needed
mkdir -p /home/jupyter/arcade_test/images /home/jupyter/arcade_test/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/images/* /home/jupyter/arcade_test/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/masks/*  /home/jupyter/arcade_test/masks/

export PYTHONPATH=/opt/MedSAM2

# Run eval directly in Python — build RepViT+CA-SAM2-decoder, no fine-tuned checkpoint
python3 - <<'PYEOF'
import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/opt/SAM2/distill")
os.environ["MODEL_VERSION"] = "decoder_only"

# Import the pieces we need from distill_student
import importlib.util, types
spec = importlib.util.spec_from_file_location("ds", "/opt/SAM2/distill/distill_student.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
teacher_ckpt = "/home/jupyter/medsam2_arcade_v2.pt"

print("Building RepViT-SAM with CA-SAM2 decoder transplant (no ARCADE training)...")
model = ds.build_repvitsam(teacher_ckpt)
model.eval()
print(f"  Device: {DEVICE}")
print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

img_paths  = sorted(glob.glob("/home/jupyter/arcade_test/images/*.png"))
mask_paths = sorted(glob.glob("/home/jupyter/arcade_test/masks/*.png"))
print(f"  Evaluating on {len(img_paths)} held-out test images...")

from torchvision import transforms
img_norm = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

dice_scores = []
with torch.no_grad():
    for ip, mp in zip(img_paths, mask_paths):
        img  = Image.open(ip).convert("RGB")
        mask = np.array(Image.open(mp).convert("L"))
        # centroid prompt
        ys, xs = np.where(mask > 127)
        cx = int(xs.mean()) if len(xs) else mask.shape[1] // 2
        cy = int(ys.mean()) if len(ys) else mask.shape[0] // 2

        img_t = img_norm(img).unsqueeze(0).to(DEVICE)
        pts   = [(torch.tensor(cx / mask.shape[1] * 512.0),
                  torch.tensor(cy / mask.shape[0] * 512.0))]

        logits = model(img_t, pts)
        pred   = (logits[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
        pred   = np.array(Image.fromarray(pred * 255).resize(
                     (mask.shape[1], mask.shape[0]), Image.NEAREST)) // 255
        gt     = (mask > 127).astype(np.uint8)

        inter = (pred * gt).sum()
        dice  = 2 * inter / (pred.sum() + gt.sum() + 1e-6)
        dice_scores.append(float(dice))

dice_arr = np.array(dice_scores)
print(f"\n=== RepViT + CA-SAM2 decoder (no ARCADE training) — 200 held-out test images ===")
print(f"  Dice mean: {dice_arr.mean():.3f}")
print(f"  Dice std:  {dice_arr.std():.3f}")
PYEOF

echo "Done $(date)"
