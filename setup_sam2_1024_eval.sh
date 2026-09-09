#!/bin/bash
# Evaluate SAM2.1 Hiera-Tiny fine-tuned checkpoint on 200 held-out ARCADE test images.
# Uses centroid click prompt (deterministic, matches all other model comparisons).
set -euo pipefail
LOG=/tmp/sam2_1024_eval.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/sam2_1024_eval.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== SAM2.1 1024x1024 eval on 200 held-out ARCADE test images $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"

# ── Deps ──────────────────────────────────────────────────────────────────────
apt-get install -y -q git python3-pip
pip3 install -q -U pip setuptools wheel
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub timm

git clone https://github.com/facebookresearch/sam2.git /opt/sam2_meta
pip3 install -e /opt/sam2_meta --no-build-isolation

mkdir -p /home/jupyter

# ── HF token ──────────────────────────────────────────────────────────────────
export HUGGING_FACE_HUB_TOKEN=$(curl -sf \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token" \
  -H "Metadata-Flavor: Google" || echo "")

# ── Checkpoints ───────────────────────────────────────────────────────────────
python3 -c "
from huggingface_hub import hf_hub_download
import os
# SAM2.1 base weights (needed for build_sam2)
hf_hub_download(repo_id='facebook/sam2.1-hiera-tiny',
                filename='sam2.1_hiera_tiny.pt',
                local_dir='/home/jupyter')
# Our fine-tuned checkpoint
hf_hub_download(repo_id='Elakiya17/fluoroscopy-sam2',
                filename='sam2_fluoroscopy_1024.pt',
                local_dir='/home/jupyter')
print('Checkpoints ready:')
for f in ['sam2.1_hiera_tiny.pt', 'sam2_fluoroscopy_1024.pt']:
    mb = os.path.getsize(f'/home/jupyter/{f}') // 1_000_000
    print(f'  {f}: {mb} MB')
"

# ── Test data (200 held-out ARCADE images) ────────────────────────────────────
mkdir -p /home/jupyter/arcade_test/images /home/jupyter/arcade_test/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/images/* /home/jupyter/arcade_test/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/masks/*  /home/jupyter/arcade_test/masks/
echo "Test images: $(ls /home/jupyter/arcade_test/images | wc -l)"

# ── Eval ──────────────────────────────────────────────────────────────────────
export PYTHONPATH=/opt/sam2_meta

python3 - <<'PYEOF'
import os, glob, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/opt/sam2_meta")
SAM2_REPO = "/opt/sam2_meta"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BASE_CKPT = "/home/jupyter/sam2.1_hiera_tiny.pt"
FTUNE_CKPT= "/home/jupyter/sam2_fluoroscopy_1024.pt"

print(f"Device: {DEVICE}")

# ── Build model (same Hydra cwd fix as training) ──────────────────────────────
from sam2.build_sam import build_sam2
_orig = os.getcwd()
os.chdir(SAM2_REPO)
try:
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", BASE_CKPT,
                       device=DEVICE, apply_postprocessing=False)
finally:
    os.chdir(_orig)

# Load fine-tuned weights
sd = torch.load(FTUNE_CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ── Image preprocessing (matches training) ────────────────────────────────────
normalize = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Forward pass (same conv_s0/s1 fix as training) ───────────────────────────
def forward_sam2(model, img_t, cx, cy):
    with torch.no_grad():
        backbone_out = model.image_encoder(img_t)
        if getattr(model, "use_high_res_features_in_sam", False):
            backbone_out["backbone_fpn"][0] = model.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = model.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1])
        _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
        if getattr(model, "directly_add_no_mem_embed", False):
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        B = img_t.shape[0]
        feats = [feat.permute(1,2,0).view(B, -1, *fs)
                 for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
        image_embed    = feats[-1]
        high_res_feats = feats[:-1]

        point    = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=DEVICE)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=DEVICE)
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(point, pt_label), boxes=None, masks=None)
        lm, _, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False, repeat_image=False,
            high_res_features=[f for f in high_res_feats]
                if getattr(model, "use_high_res_features_in_sam", False) else None,
        )
    return lm  # [1, 1, H, W]

# ── Centroid prompt (deterministic, matches all other model comparisons) ───────
def centroid_click(mask_np):
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        return mask_np.shape[1] // 2, mask_np.shape[0] // 2
    return int(xs.mean()), int(ys.mean())

# ── Evaluate ──────────────────────────────────────────────────────────────────
img_paths  = sorted(glob.glob("/home/jupyter/arcade_test/images/*.png"))
mask_paths = sorted(glob.glob("/home/jupyter/arcade_test/masks/*.png"))
print(f"Evaluating on {len(img_paths)} test images...")

dice_scores, iou_scores = [], []

for i, (ip, mp) in enumerate(zip(img_paths, mask_paths)):
    img  = Image.open(ip).convert("RGB")
    gt   = np.array(Image.open(mp).convert("L"))

    # Centroid of original-resolution mask → scale to 1024
    cx_orig, cy_orig = centroid_click(gt)
    cx = cx_orig / gt.shape[1] * 1024.0
    cy = cy_orig / gt.shape[0] * 1024.0

    img_t = normalize(img).unsqueeze(0).to(DEVICE)

    logits = forward_sam2(model, img_t, cx, cy)

    # Resize logits back to original mask resolution
    logits_up = F.interpolate(logits, size=(gt.shape[0], gt.shape[1]),
                              mode="bilinear", align_corners=False)
    pred = (logits_up[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
    gt_b = (gt > 127).astype(np.uint8)

    inter  = (pred * gt_b).sum()
    union  = pred.sum() + gt_b.sum() - inter
    dice   = 2 * inter / (pred.sum() + gt_b.sum() + 1e-6)
    iou    = inter / (union + 1e-6)

    dice_scores.append(float(dice))
    iou_scores.append(float(iou))

    if (i + 1) % 20 == 0:
        print(f"  [{i+1:3d}/200] running Dice: {np.mean(dice_scores):.4f}")

dice_arr = np.array(dice_scores)
iou_arr  = np.array(iou_scores)

print(f"\n{'='*55}")
print(f"SAM2.1 Hiera-Tiny 1024x1024 — 200 held-out ARCADE test images")
print(f"{'='*55}")
print(f"  Dice  mean ± std : {dice_arr.mean():.4f} ± {dice_arr.std():.4f}")
print(f"  Dice  median     : {np.median(dice_arr):.4f}")
print(f"  Dice  min / max  : {dice_arr.min():.4f} / {dice_arr.max():.4f}")
print(f"  IoU   mean       : {iou_arr.mean():.4f} ± {iou_arr.std():.4f}")
print(f"{'='*55}")
print(f"\nComparison:")
print(f"  CA-SAM2 teacher (SAM2.1, 512px)  : 0.767")
print(f"  MobileSAM v4 (GT only, 1024px)   : 0.806 ± 0.086")
print(f"  SAM2.1 fine-tune (this, 1024px)  : {dice_arr.mean():.3f} ± {dice_arr.std():.3f}")
PYEOF

echo "=== Done $(date) ==="
