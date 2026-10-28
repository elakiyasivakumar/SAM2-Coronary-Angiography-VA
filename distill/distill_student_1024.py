"""
Knowledge Distillation with 1024x1024 Teacher (CA-SAM2 / SAM2.1 Hiera-Tiny).

Key fixes from legacy 512px distillation:
  1. Resolution Parity: Both Teacher and Student operate at native 1024x1024.
  2. Spatial Feature Alignment: Teacher embedding is [B, 256, 64, 64] and Student is [B, 256, 64, 64].
     Zero downsampling interpolation needed for Feature KD!
  3. Prompt Synchronization: The teacher is evaluated dynamically on the exact same
     prompt point (cx, cy) and image batch as the student during training.
     Eliminates the prompt mismatch bug where student was penalized against a different prompt.
  4. Disentangled Ablations:
     --ablation 1: GT only (Dice + wBCE)
     --ablation 2: GT + Response KD (prompt-aligned)
     --ablation 3: GT + clDice
     --ablation 4: GT + Response KD + clDice
     --ablation 5: GT + Response KD + clDice + Feature KD
  5. Continuous Logit Upsampling: Evaluates by bilinearly upsampling continuous logits
     to full resolution before thresholding.
"""

import argparse
import glob
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as cp
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Paths & Environment ───────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024
DATA_DIR = os.environ.get("DATA_DIR", "/home/jupyter/arcade_train")
TEST_DIR = os.environ.get("TEST_DIR", "/home/jupyter/arcade_test")
BUCKET = os.environ.get("GCS_BUCKET", "gs://coronary-angio-v2")
MOBILE_CKPT = os.environ.get("MOBILE_CKPT", "/home/jupyter/mobile_sam.pt")
TEACHER_CKPT = os.environ.get("TEACHER_CKPT", "/home/jupyter/sam2_fluoroscopy_1024.pt")
TEACHER_BASE_CKPT = os.environ.get("TEACHER_BASE_CKPT", "/home/jupyter/sam2.1_hiera_tiny.pt")
SAM2_REPO = os.environ.get("SAM2_REPO", "/opt/sam2_meta")

# ── Normalization ─────────────────────────────────────────────────────────────
IMG_NORM_1024 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

NUM_GEOM_AUGS = 5

def _geom_pil(pil_img, aug_idx):
    if aug_idx == 0:
        return pil_img
    if aug_idx == 1:
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if aug_idx == 2:
        return pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    if aug_idx == 3:
        return pil_img.rotate(20)
    if aug_idx == 4:
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(20)
    return pil_img

def random_color_aug(img_pil):
    """X-ray exposure variation simulation."""
    if random.random() < 0.6:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.7, 1.35))
    if random.random() < 0.6:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.7, 1.35))
    if random.random() < 0.4:
        arr = np.array(img_pil).astype(np.float32)
        sigma = random.uniform(4, 18)
        arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(arr)
    return img_pil

def centroid_click(mask_np, jitter=5):
    ys, xs = np.where(mask_np > 0)
    if len(ys) == 0:
        return (mask_np.shape[1] // 2, mask_np.shape[0] // 2)
    idx = np.random.randint(len(ys))
    cy = int(ys[idx]) + random.randint(-jitter, jitter)
    cx = int(xs[idx]) + random.randint(-jitter, jitter)
    cy = max(0, min(mask_np.shape[0] - 1, cy))
    cx = max(0, min(mask_np.shape[1] - 1, cx))
    return (cx, cy)

class PromptAlignedDataset(Dataset):
    def __init__(self, img_paths, mask_paths, color_aug=True, jitter=5):
        self.pairs = []
        for ip, mp in zip(img_paths, mask_paths):
            for aug_idx in range(NUM_GEOM_AUGS):
                self.pairs.append((ip, mp, aug_idx))
        self.color_aug = color_aug
        self.jitter = jitter

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp, aug_idx = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")

        img = _geom_pil(img, aug_idx)
        mask = _geom_pil(mask, aug_idx)

        if self.color_aug:
            img = random_color_aug(img)

        mask_np = np.array(mask)
        cx, cy = centroid_click(mask_np, jitter=self.jitter)
        h, w = mask_np.shape
        cx_1024 = cx / w * IMG_SIZE
        cy_1024 = cy / h * IMG_SIZE

        img_t = IMG_NORM_1024(img)
        mask_t = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)
        pt = torch.tensor([cx_1024, cy_1024], dtype=torch.float32)

        return img_t, mask_t, pt

# ── Losses ────────────────────────────────────────────────────────────────────
def dice_loss(logits, target, smooth=1e-5):
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()

def soft_erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)

def soft_dilate(x):
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))

def soft_open(x):
    return soft_dilate(soft_erode(x))

def _skel_inner(x):
    x = x.float()
    skel = F.relu(x - soft_open(x))
    for _ in range(10):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel

def soft_skel(x):
    return cp.checkpoint(_skel_inner, x, use_reentrant=False)

def soft_cldice_loss(probs, target, smooth=1.0):
    skel_pred = soft_skel(probs)
    with torch.no_grad():
        skel_true = _skel_inner(target)
    tprec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_true * probs) + smooth) / (torch.sum(skel_true) + smooth)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

def combined_loss(logits, gt, pos_weight, cldice_w):
    if gt.shape != logits.shape:
        gt = F.interpolate(gt, size=logits.shape[-2:], mode="nearest")
    wbce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight.to(logits.device))
    d = dice_loss(logits, gt)
    hard = 0.5 * d + 0.2 * wbce

    if cldice_w > 0:
        probs = torch.sigmoid(logits).float()
        cl = soft_cldice_loss(probs, gt.float())
    else:
        cl = torch.tensor(0.0, device=logits.device)

    return hard + cldice_w * cl, d.item(), (cl.item() if cldice_w > 0 else 0.0)

# ── Teacher Model (SAM2.1 1024x1024) ──────────────────────────────────────────
def load_1024_teacher(ckpt_path=None):
    from huggingface_hub import hf_hub_download
    for p in ["/opt/sam2_meta", SAM2_REPO]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    from sam2.build_sam import build_sam2

    base_ckpt = TEACHER_BASE_CKPT
    if not os.path.exists(base_ckpt):
        print("Downloading base SAM2.1 Hiera-Tiny weights...")
        hf_hub_download(repo_id="facebook/sam2.1-hiera-tiny",
                        filename="sam2.1_hiera_tiny.pt",
                        local_dir=os.path.dirname(base_ckpt) or ".")

    ft_ckpt = ckpt_path or TEACHER_CKPT
    if not os.path.exists(ft_ckpt):
        print(f"Downloading fine-tuned 1024 teacher checkpoint from Elakiya17/fluoroscopy-sam2...")
        hf_hub_download(repo_id="Elakiya17/fluoroscopy-sam2",
                        filename="sam2_fluoroscopy_1024.pt",
                        local_dir=os.path.dirname(ft_ckpt) or ".")

    sd = torch.load(ft_ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(sd, dict) and "model" in sd):
        print("Wrapping teacher checkpoint with {'model': state_dict} for build_sam2...")
        torch.save({"model": sd}, ft_ckpt)

    _orig = os.getcwd()
    if os.path.isdir(SAM2_REPO):
        os.chdir(SAM2_REPO)
    try:
        model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", ft_ckpt,
                           device=DEVICE, apply_postprocessing=False)
    finally:
        os.chdir(_orig)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("1024x1024 SAM2.1 Teacher loaded and frozen.")
    return model

def forward_sam2(model, imgs, pts_list):
    """
    Teacher forward pass at 1024x1024.
    Returns:
      logits: [B, 1, 256, 256]
      image_embed: [B, 256, 64, 64]
    """
    B = imgs.shape[0]
    backbone_out = model.image_encoder(imgs)
    if getattr(model, "use_high_res_features_in_sam", False):
        backbone_out["backbone_fpn"][0] = model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1])
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    if getattr(model, "directly_add_no_mem_embed", False):
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    feats = [feat.permute(1, 2, 0).view(B, -1, *fs)
             for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
    image_embed = feats[-1]      # [B, 256, 64, 64]
    high_res_feats = feats[:-1]

    logits_list = []
    for i, pt in enumerate(pts_list):
        cx, cy = pt[0].item(), pt[1].item()
        point = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=DEVICE)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=DEVICE)
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(point, pt_label), boxes=None, masks=None)
        lm, _, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed[i].unsqueeze(0),
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False, repeat_image=False,
            high_res_features=[f[i].unsqueeze(0) for f in high_res_feats],
        )
        logits_list.append(lm)
    return torch.cat(logits_list, dim=0), image_embed

# ── Student Model (MobileSAM 1024x1024) ───────────────────────────────────────
def build_mobilesam(teacher_ckpt, transplant_decoder=True):
    from mobile_sam import sam_model_registry
    if not os.path.exists(MOBILE_CKPT):
        print("Downloading MobileSAM base checkpoint...")
        subprocess.run([
            "wget", "-q", "-O", MOBILE_CKPT,
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        ], check=True)
    model = sam_model_registry["vit_t"](checkpoint=MOBILE_CKPT)

    if transplant_decoder and os.path.exists(teacher_ckpt):
        teacher_sd = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        if isinstance(teacher_sd, dict) and "model" in teacher_sd:
            teacher_sd = teacher_sd["model"]
        decoder_sd = {k.replace("sam_mask_decoder.", ""): v
                      for k, v in teacher_sd.items()
                      if k.startswith("sam_mask_decoder.")}
        prompt_sd = {k.replace("sam_prompt_encoder.", ""): v
                     for k, v in teacher_sd.items()
                     if k.startswith("sam_prompt_encoder.")}
        model.mask_decoder.load_state_dict(decoder_sd, strict=False)
        model.prompt_encoder.load_state_dict(prompt_sd, strict=False)
        print("  Decoder & Prompt encoder weights partially transplanted from teacher.")

    return model.to(DEVICE)

def forward_mobilesam(model, imgs, pts_list):
    """
    Student forward pass at 1024x1024.
    Returns:
      logits: [B, 1, 256, 256]
      image_embed: [B, 256, 64, 64]
    """
    B = imgs.shape[0]
    image_embed = model.image_encoder(imgs)  # [B, 256, 64, 64]
    if getattr(model, "neck_adapter", None) is not None:
        image_embed = model.neck_adapter(image_embed)

    logits_list = []
    for i, pt in enumerate(pts_list):
        cx, cy = pt[0].item(), pt[1].item()
        point = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=DEVICE)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=DEVICE)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=(point, pt_label), boxes=None, masks=None)
        lm, _ = model.mask_decoder(
            image_embeddings=image_embed[i].unsqueeze(0),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        logits_list.append(lm)
    return torch.cat(logits_list, dim=0), image_embed

class FeatureAdapter(nn.Module):
    """1x1 conv projecting student embedding into teacher embedding space."""
    def __init__(self, in_ch=256, out_ch=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        nn.init.eye_(self.proj.weight.squeeze(-1).squeeze(-1))

    def forward(self, x):
        return self.proj(x)

# ── Training & Evaluation Loop ────────────────────────────────────────────────
def evaluate(model, test_img_paths, test_mask_paths):
    model.eval()
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for ip, mp in zip(test_img_paths, test_mask_paths):
            img = Image.open(ip).convert("RGB")
            mask = np.array(Image.open(mp).convert("L"))
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                cx, cy = mask.shape[1] // 2, mask.shape[0] // 2
            else:
                cx, cy = int(xs.mean()), int(ys.mean())

            img_t = IMG_NORM_1024(img).unsqueeze(0).to(DEVICE)
            pt = torch.tensor([[cx / mask.shape[1] * IMG_SIZE,
                                cy / mask.shape[0] * IMG_SIZE]], dtype=torch.float32)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                logits, _ = forward_mobilesam(model, img_t, [pt[0]])

            # Continuous bilinear upsampling before thresholding (preserves fine vessels)
            logits_up = F.interpolate(logits, size=(mask.shape[0], mask.shape[1]),
                                      mode="bilinear", align_corners=False)
            pred = (logits_up[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
            gt = (mask > 0).astype(np.uint8)

            inter = (pred * gt).sum()
            union = pred.sum() + gt.sum() - inter
            dice = (2.0 * inter / (pred.sum() + gt.sum())) if (pred.sum() + gt.sum()) > 0 else 1.0
            iou = (inter / union) if union > 0 else 1.0
            dice_scores.append(float(dice))
            iou_scores.append(float(iou))

    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    mean_iou = np.mean(iou_scores)
    return mean_dice, std_dice, mean_iou

def train(args):
    EPOCHS = int(os.environ.get("MAX_EPOCHS", args.epochs))
    PATIENCE = int(os.environ.get("ES_PATIENCE", 10))
    BATCH = int(os.environ.get("BATCH_SIZE", args.batch_size))
    BASE_LR = 3e-4
    WARMUP_EP = 3
    WD = 0.01
    GRAD_CLIP = 0.5
    FG_FRAC = 0.05
    pos_weight = torch.tensor([(1 - FG_FRAC) / FG_FRAC]).to(DEVICE)

    # Resolve ablation settings
    use_kd = args.ablation in [2, 4, 5] or args.use_kd
    use_cldice = args.ablation in [3, 4, 5] or args.use_cldice
    use_feat_kd = args.ablation == 5 or args.use_feat_kd
    FEAT_KD_W = float(os.environ.get("FEAT_KD_W", 0.1))

    print(f"\nConfiguration:")
    print(f"  Ablation {args.ablation} | KD Response: {use_kd} | clDice: {use_cldice} | Feature KD: {use_feat_kd}")
    print(f"  Device: {DEVICE} | Batch size: {BATCH} | Epochs: {EPOCHS}")

    # Dataset paths
    img_paths_all = sorted(glob.glob(DATA_DIR + "/images/*.png"))
    mask_paths_all = sorted(glob.glob(DATA_DIR + "/masks/*.png"))
    if not img_paths_all:
        raise RuntimeError(f"No training images found in {DATA_DIR}/images/")

    indices = list(range(len(img_paths_all)))
    random.Random(42).shuffle(indices)
    val_idx = indices[:100]
    trn_idx = indices[100:]
    img_train = [img_paths_all[i] for i in trn_idx]
    mask_train = [mask_paths_all[i] for i in trn_idx]
    img_val = [img_paths_all[i] for i in val_idx]
    mask_val = [mask_paths_all[i] for i in val_idx]

    ds = PromptAlignedDataset(img_train, mask_train, color_aug=True, jitter=5)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    num_workers=4 if os.name != 'nt' else 0,
                    pin_memory=(DEVICE == "cuda"), drop_last=True)
    print(f"  Train set: {len(ds)} samples ({len(img_train)} images x {NUM_GEOM_AUGS} augs)")

    # Build models
    student = build_mobilesam(TEACHER_CKPT, transplant_decoder=not args.no_transplant)
    trainable = [p for p in student.parameters() if p.requires_grad]

    feat_adapter = None
    teacher = None
    if use_kd or use_feat_kd:
        teacher = load_1024_teacher(TEACHER_CKPT)
        if use_feat_kd:
            feat_adapter = FeatureAdapter(256, 256).to(DEVICE)
            trainable += list(feat_adapter.parameters())

    optimizer = optim.AdamW(trainable, lr=BASE_LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    suffix = f"mobilesam_1024_abl{args.ablation}"
    ckpt_dir = "/workspace/checkpoints" if os.path.exists("/workspace/checkpoints") else "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_out = os.path.join(ckpt_dir, f"{suffix}.pt")
    best_val_dice = -1.0
    no_improve = 0

    print("\nStarting Training...")
    for epoch in range(1, EPOCHS + 1):
        if epoch <= WARMUP_EP:
            lr = BASE_LR * (epoch / WARMUP_EP)
            for g in optimizer.param_groups:
                g["lr"] = lr

        cldice_w = 0.0
        if use_cldice:
            cldice_w = min(0.3, 0.3 * max(0, epoch - 3) / 5)

        student.train()
        epoch_loss = 0.0
        t0 = time.time()

        for imgs, masks, pts in dl:
            imgs_d = imgs.to(DEVICE)
            masks_d = masks.to(DEVICE)
            pts_list = [pts[i] for i in range(len(pts))]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                # 1. Student forward pass
                s_logits, s_embed = forward_mobilesam(student, imgs_d, pts_list)

                # 2. Hard ground-truth loss
                hard_loss, d_val, cl_val = combined_loss(s_logits, masks_d, pos_weight, cldice_w)
                loss = hard_loss

                # 3. Dynamic Prompt-Aligned Teacher Supervision
                if use_kd or use_feat_kd:
                    with torch.no_grad():
                        t_logits, t_embed = forward_sam2(teacher, imgs_d, pts_list)

                    if use_kd:
                        soft_targets = torch.sigmoid(t_logits)
                        kd_loss = F.binary_cross_entropy_with_logits(s_logits, soft_targets, pos_weight=pos_weight)
                        loss = 0.5 * kd_loss + 0.5 * hard_loss

                    if use_feat_kd and feat_adapter is not None:
                        s_proj = feat_adapter(s_embed)
                        fg = F.interpolate(masks_d.float(), size=t_embed.shape[-2:], mode="nearest")
                        fg_w = 1.0 + 9.0 * fg
                        feat_loss = ((s_proj - t_embed) ** 2 * fg_w).mean()
                        loss = loss + FEAT_KD_W * feat_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        if epoch > WARMUP_EP:
            scheduler.step()

        val_dice, _, _ = evaluate(student, img_val, mask_val)
        avg_loss = epoch_loss / len(dl)
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f} | Time: {time.time()-t0:.0f}s", flush=True)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            no_improve = 0
            os.makedirs(os.path.dirname(ckpt_out), exist_ok=True)
            torch.save(student.state_dict(), ckpt_out)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val Dice: {best_val_dice:.4f})")
                break

    # Load best checkpoint and evaluate on held-out test set
    if os.path.exists(ckpt_out):
        student.load_state_dict(torch.load(ckpt_out, map_location=DEVICE, weights_only=False))

    test_imgs = sorted(glob.glob(TEST_DIR + "/images/*.png"))
    test_masks = sorted(glob.glob(TEST_DIR + "/masks/*.png"))
    if test_imgs:
        test_dice, test_std, test_iou = evaluate(student, test_imgs, test_masks)
        print(f"\n{'='*60}")
        print(f"Held-out ARCADE Test Results ({len(test_imgs)} images):")
        print(f"  Test Dice: {test_dice:.4f} ± {test_std:.4f}")
        print(f"  Test IoU:  {test_iou:.4f}")
        print(f"{'='*60}\n")
    return best_val_dice

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=int, choices=[1, 2, 3, 4, 5], default=1,
                        help="1=GT only, 2=GT+Response KD, 3=GT+clDice, 4=Full KD+clDice, 5=KD+clDice+FeatKD")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_kd", action="store_true", help="Enable prompt-aligned response KD")
    parser.add_argument("--use_cldice", action="store_true", help="Enable centerline Dice loss")
    parser.add_argument("--use_feat_kd", action="store_true", help="Enable foreground-weighted feature KD")
    parser.add_argument("--no_transplant", action="store_true", help="Do not transplant decoder weights")
    args = parser.parse_args()

    train(args)
