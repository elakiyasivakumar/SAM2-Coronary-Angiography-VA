#!/bin/bash
# Video benchmark: MobileSAM v4 vs CA-SAM2 teacher on CPU-only 8GB RAM
# Uses 200 ARCADE val images split into 4 pseudo-videos of 50 frames each
# Reports per-model: latency, peak RAM, FPS; saves side-by-side overlay images
set -euo pipefail
LOG=/tmp/video_compare.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/video_compare.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== Video benchmark: MobileSAM v4 vs CA-SAM2 (CPU, 8GB RAM) $(date) ==="
echo "CPU: $(nproc) cores | RAM: $(free -h | awk '/^Mem:/{print $2}')"

apt-get install -y -q git python3 python3-pip curl ffmpeg
pip3 install -q -U pip setuptools wheel
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub timm Pillow numpy

git clone https://github.com/ChaoningZhang/MobileSAM.git /opt/MobileSAM
pip3 install -q -e /opt/MobileSAM --no-build-isolation

git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
pip3 install -q -e /opt/MedSAM2 --no-build-isolation

git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2

mkdir -p /home/jupyter
ln -sfn /opt/MedSAM2 /home/jupyter/MedSAM2

# Download checkpoints
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Elakiya17/medsam_distill_models', filename='mobilesam_v4.pt', local_dir='/home/jupyter')
hf_hub_download(repo_id='Elakiya17/CA-SAM2', filename='medsam2_arcade_v2.pt', local_dir='/home/jupyter')
print('Checkpoints ready.')
"

# Download MobileSAM base weights
wget -q -O /tmp/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# Test frames: 200 ARCADE val images
mkdir -p /home/jupyter/frames/images /home/jupyter/frames/masks
gsutil -m cp gs://coronary-angio-v2/datasets/arcade/val/images/* /home/jupyter/frames/images/
gsutil -m cp gs://coronary-angio-v2/datasets/arcade/val/masks/*  /home/jupyter/frames/masks/

export PYTHONPATH=/opt/MedSAM2

python3 - <<'PYEOF'
import os, sys, glob, time, resource
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/opt/MedSAM2")

DEVICE = "cpu"
FRAME_DIR   = "/home/jupyter/frames/images"
MASK_DIR    = "/home/jupyter/frames/masks"
OUT_DIR     = "/home/jupyter/video_out"
os.makedirs(OUT_DIR, exist_ok=True)

frame_paths = sorted(glob.glob(FRAME_DIR + "/*.png"))
mask_paths  = sorted(glob.glob(MASK_DIR  + "/*.png"))
# 4 pseudo-videos of 50 frames each
videos = [frame_paths[i*50:(i+1)*50] for i in range(4)]
vmasks = [mask_paths[i*50:(i+1)*50]  for i in range(4)]

def peak_ram_mb():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1e3  # Linux: KB → MB

def centroid(mask_np):
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        return mask_np.shape[1]//2, mask_np.shape[0]//2
    return int(xs.mean()), int(ys.mean())

# ── MobileSAM v4 inference ────────────────────────────────────────────────────
print("\n=== Loading MobileSAM v4 ===")
from mobile_sam import sam_model_registry
model_v4 = sam_model_registry["vit_t"](checkpoint="/tmp/mobile_sam.pt")
sd = torch.load("/home/jupyter/mobilesam_v4.pt", map_location="cpu", weights_only=False)
model_v4.load_state_dict(sd, strict=False)
model_v4.eval()
print(f"  Params: {sum(p.numel() for p in model_v4.parameters())/1e6:.1f}M")

from torchvision import transforms
norm_1024 = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def infer_mobilesam(model, img_pil, cx, cy):
    img_t = norm_1024(img_pil).unsqueeze(0)
    with torch.no_grad():
        emb = model.image_encoder(img_t)
        if emb.shape[-1] != 64:
            emb = F.interpolate(emb, (64,64), mode="bilinear", align_corners=False)
        pt = torch.tensor([[[cx/img_pil.width*1024, cy/img_pil.height*1024]]], dtype=torch.float32)
        lbl = torch.ones(1,1,dtype=torch.int)
        sp, dp = model.prompt_encoder(points=(pt,lbl), boxes=None, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dp,
            multimask_output=False)
    return (logits[0,0].numpy() > 0.0).astype(np.uint8)

# ── CA-SAM2 teacher inference ─────────────────────────────────────────────────
print("=== Loading CA-SAM2 teacher ===")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import hydra
from omegaconf import OmegaConf

cfg_path = "/opt/MedSAM2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_model = build_sam2("sam2.1_hiera_t.yaml", "/home/jupyter/medsam2_arcade_v2.pt",
                         device=DEVICE, apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2_model)
print(f"  Params: {sum(p.numel() for p in sam2_model.parameters())/1e6:.1f}M")

norm_512 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def infer_casam2(predictor, img_pil, cx, cy):
    img_np = np.array(img_pil.resize((512,512)))
    with torch.no_grad():
        predictor.set_image(img_np)
        masks, _, _ = predictor.predict(
            point_coords=np.array([[cx/img_pil.width*512, cy/img_pil.height*512]]),
            point_labels=np.array([1]),
            multimask_output=False)
    return masks[0].astype(np.uint8)

# ── Run comparison on all 4 pseudo-videos ─────────────────────────────────────
def overlay(img_pil, mask, color=(255,50,50), alpha=0.45):
    out = np.array(img_pil.convert("RGB")).copy()
    out[mask==1, 0] = np.clip(out[mask==1,0]*( 1-alpha) + color[0]*alpha, 0, 255).astype(np.uint8)
    out[mask==1, 1] = np.clip(out[mask==1,1]*(1-alpha) + color[1]*alpha, 0, 255).astype(np.uint8)
    out[mask==1, 2] = np.clip(out[mask==1,2]*(1-alpha) + color[2]*alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

all_v4_lat, all_ca_lat = [], []
all_v4_dice, all_ca_dice = [], []

for vid_idx, (fps, mps) in enumerate(zip(videos, vmasks)):
    vid_dir = os.path.join(OUT_DIR, f"video_{vid_idx+1}")
    os.makedirs(vid_dir, exist_ok=True)
    v4_lats, ca_lats = [], []
    v4_dice, ca_dice = [], []

    for i, (fp, mp) in enumerate(zip(fps, mps)):
        img = Image.open(fp).convert("RGB")
        gt  = np.array(Image.open(mp).convert("L"))
        cx, cy = centroid(gt)

        # MobileSAM v4
        t0 = time.time()
        m_v4 = infer_mobilesam(model_v4, img, cx, cy)
        v4_lats.append((time.time()-t0)*1000)
        m_v4_full = np.array(Image.fromarray(m_v4*255).resize((gt.shape[1],gt.shape[0]),Image.NEAREST))//255
        inter = (m_v4_full*(gt>127)).sum()
        v4_dice.append(2*inter/(m_v4_full.sum()+(gt>127).sum()+1e-6))

        # CA-SAM2
        t0 = time.time()
        m_ca = infer_casam2(predictor, img, cx, cy)
        ca_lats.append((time.time()-t0)*1000)
        m_ca_full = np.array(Image.fromarray(m_ca*255).resize((gt.shape[1],gt.shape[0]),Image.NEAREST))//255
        inter = (m_ca_full*(gt>127)).sum()
        ca_dice.append(2*inter/(m_ca_full.sum()+(gt>127).sum()+1e-6))

        # Side-by-side: original | v4 | CA-SAM2 | GT
        ov_v4 = overlay(img, m_v4_full, color=(255,80,80))
        ov_ca = overlay(img, m_ca_full, color=(80,80,255))
        ov_gt = overlay(img, (gt>127).astype(np.uint8), color=(80,255,80))
        W, H = img.size
        composite = Image.new("RGB", (W*4, H))
        composite.paste(img,   (0,   0))
        composite.paste(ov_v4, (W,   0))
        composite.paste(ov_ca, (W*2, 0))
        composite.paste(ov_gt, (W*3, 0))
        composite.save(os.path.join(vid_dir, f"frame_{i:03d}.png"))

    all_v4_lat += v4_lats
    all_ca_lat += ca_lats
    all_v4_dice += v4_dice
    all_ca_dice += ca_dice
    print(f"Video {vid_idx+1}/4: v4 Dice={np.mean(v4_dice):.3f} ({np.mean(v4_lats):.0f}ms/frame) | "
          f"CA-SAM2 Dice={np.mean(ca_dice):.3f} ({np.mean(ca_lats):.0f}ms/frame)")

    # Make mp4 from frames
    os.system(f"ffmpeg -y -r 5 -i {vid_dir}/frame_%03d.png "
              f"-vcodec libx264 -pix_fmt yuv420p {vid_dir}/compare.mp4 -loglevel quiet")

ram = peak_ram_mb()
print(f"\n=== OVERALL RESULTS (200 frames, 4 videos, CPU) ===")
print(f"{'':20s}  {'MobileSAM v4':>16s}  {'CA-SAM2':>16s}")
print(f"{'Test Dice':20s}  {np.mean(all_v4_dice):>16.3f}  {np.mean(all_ca_dice):>16.3f}")
print(f"{'Latency mean (ms)':20s}  {np.mean(all_v4_lat):>16.1f}  {np.mean(all_ca_lat):>16.1f}")
print(f"{'Latency p95 (ms)':20s}  {np.percentile(all_v4_lat,95):>16.1f}  {np.percentile(all_ca_lat,95):>16.1f}")
print(f"{'FPS':20s}  {1000/np.mean(all_v4_lat):>16.2f}  {1000/np.mean(all_ca_lat):>16.2f}")
print(f"{'Peak RAM (MB)':20s}  {ram:>16.0f}  {'(shared)':>16s}")
print(f"{'Model size (MB)':20s}  {'38.8':>16s}  {'~160':>16s}")
PYEOF

# Upload overlay frames and videos
gsutil -m cp -r /home/jupyter/video_out gs://coronary-angio-v2/results/video_compare/
echo "Done $(date)"
