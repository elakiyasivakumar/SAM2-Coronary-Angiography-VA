#!/bin/bash
# Evaluate out-of-the-box (untrained) MobileSAM on ARCADE test set (200 images)
set -euo pipefail
LOG=/tmp/vanilla_eval.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/vanilla_mobilesam_eval.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== Vanilla MobileSAM eval on ARCADE test set $(date) ==="

apt-get install -y -q git
pip3 install -q scikit-image huggingface_hub timm
pip3 install -q git+https://github.com/ChaoningZhang/MobileSAM.git

# Download vanilla MobileSAM weights (no fine-tuning)
wget -q -O /tmp/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
echo "Vanilla MobileSAM weights ready."

# Download ARCADE test set (200 held-out images)
mkdir -p /tmp/arcade_test/images /tmp/arcade_test/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/images/* /tmp/arcade_test/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/masks/*  /tmp/arcade_test/masks/
echo "Test data ready: $(ls /tmp/arcade_test/images | wc -l) images"

python3 - <<'PYEOF'
import os, glob, json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024
img_norm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from mobile_sam import sam_model_registry
model = sam_model_registry["vit_t"](checkpoint="/tmp/mobile_sam.pt").to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

img_paths  = sorted(glob.glob("/tmp/arcade_test/images/*.png"))
mask_paths = sorted(glob.glob("/tmp/arcade_test/masks/*.png"))
print(f"Evaluating on {len(img_paths)} images...")

dice_scores, iou_scores = [], []

with torch.no_grad():
    for i, (ip, mp) in enumerate(zip(img_paths, mask_paths)):
        img    = Image.open(ip).convert("RGB")
        mask   = np.array(Image.open(mp).convert("L"))

        # centroid click prompt
        ys, xs = np.where(mask > 0)
        cx = int(xs.mean()) if len(xs) else mask.shape[1] // 2
        cy = int(ys.mean()) if len(ys) else mask.shape[0] // 2

        img_t = img_norm(img).unsqueeze(0).to(DEVICE)

        # scale click to model input size
        cx_n = cx / mask.shape[1] * IMG_SIZE
        cy_n = cy / mask.shape[0] * IMG_SIZE

        pt       = torch.tensor([[[cx_n, cy_n]]], dtype=torch.float32, device=DEVICE)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=DEVICE)

        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                        if DEVICE == "cuda" else torch.no_grad())
        with autocast_ctx:
            image_embed = model.image_encoder(img_t)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=(pt, pt_label), boxes=None, masks=None)
            logits, _ = model.mask_decoder(
                image_embeddings=image_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

        pred = (logits[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
        pred = np.array(Image.fromarray(pred * 255).resize(
            (mask.shape[1], mask.shape[0]), Image.NEAREST)) > 127

        gt    = mask > 0
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        denom = pred.sum() + gt.sum()
        dice  = (2 * inter / denom) if denom > 0 else 1.0
        iou   = (inter / union)     if union > 0 else 1.0
        dice_scores.append(dice)
        iou_scores.append(iou)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/200  running Dice: {np.mean(dice_scores):.4f}", flush=True)

mean_dice = float(np.mean(dice_scores))
std_dice  = float(np.std(dice_scores))
mean_iou  = float(np.mean(iou_scores))

print(f"\n=== Vanilla MobileSAM (no fine-tuning) — ARCADE test set (200 images) ===")
print(f"  Dice: {mean_dice:.3f} ± {std_dice:.3f}")
print(f"  IoU:  {mean_iou:.3f}")

result = {"model": "vanilla_mobilesam", "dice_mean": mean_dice,
          "dice_std": std_dice, "iou_mean": mean_iou, "n": len(dice_scores)}
with open("/tmp/results_vanilla_mobilesam.json", "w") as f:
    json.dump(result, f, indent=2)
PYEOF

gsutil cp /tmp/results_vanilla_mobilesam.json \
  gs://coronary-angio-v2/results/distillation/results_vanilla_mobilesam.json
echo "Done $(date)"
