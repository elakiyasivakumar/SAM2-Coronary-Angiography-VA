"""
Stage 2/3: Knowledge distillation student training.

Usage:
  python distill_student.py --student mobilesam --ablation 4
  python distill_student.py --student repvitsam --ablation 4

--ablation:
  1 = GT only (no KD, no clDice)          baseline
  2 = KD response + GT (no clDice)
  3 = KD response + GT + clDice
  4 = Full (KD response + GT + clDice)    final model  [default]

Loss:
  L = 0.5 * KD_BCE(soft_logits) + 0.4 * (0.5*Dice + 0.2*wBCE) + clDice_w * clDice

Data split (v3+):
  900 train / 100 val (for early stopping) from 1000 ARCADE train images (seed=42)
  200 ARCADE val images = held-out test set (never touched during training)
"""

import argparse, os, sys, glob, subprocess, random, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance

# ── paths ────────────────────────────────────────────────────────────────────
BUCKET        = "gs://coronary-angio-v2"
DATA_DIR      = "/home/jupyter/arcade_train"
SOFT_DIR      = "/tmp/soft_labels"   # generated at runtime from HF teacher
MOBILE_CKPT   = "/home/jupyter/mobile_sam.pt"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZE    = 512   # teacher / repvit input size
MOBILE_SIZE   = int(os.environ.get("MOBILE_SIZE", 1024))  # 1024 default, set to 512 to match teacher resolution
HIRES         = MODEL_SIZE // 4
BB_FEAT_SIZES = [[HIRES // (2**k)] * 2 for k in range(3)]

# ── augmentation ─────────────────────────────────────────────────────────────
# Geometric aug by index — applied identically to image, mask, and soft label
# so there's no label/prediction misalignment after flipping/rotating.
NUM_GEOM_AUGS = 5  # 0=none 1=flip_lr 2=flip_tb 3=rot20 4=flip_lr+rot20

def _geom_pil(pil_img, aug_idx):
    if aug_idx == 0: return pil_img
    if aug_idx == 1: return pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if aug_idx == 2: return pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    if aug_idx == 3: return pil_img.rotate(20)
    if aug_idx == 4: return pil_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(20)
    return pil_img

def _geom_np(arr_2d, aug_idx):
    """Apply geometric aug to a 2D float array (for soft labels)."""
    if aug_idx == 0: return arr_2d
    pil = Image.fromarray(arr_2d.astype(np.float32), mode='F')
    return np.array(_geom_pil(pil, aug_idx), dtype=np.float32)

def random_color_aug(img_pil):
    """Random brightness, contrast, and Gaussian noise — X-ray exposure simulation."""
    if random.random() < 0.6:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.7, 1.35))
    if random.random() < 0.6:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.7, 1.35))
    if random.random() < 0.4:
        arr   = np.array(img_pil).astype(np.float32)
        sigma = random.uniform(4, 18)
        arr   = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(arr)
    return img_pil

IMG_NORM_512 = transforms.Compose([
    transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMG_NORM_1024 = transforms.Compose([
    transforms.Resize((MOBILE_SIZE, MOBILE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── teacher (for on-the-fly soft label generation) ───────────────────────────
def load_teacher(ckpt_path):
    import sys as _sys
    for p in ["/opt/MedSAM2", "/home/jupyter/MedSAM2"]:
        if os.path.isdir(p) and p not in _sys.path:
            _sys.path.insert(0, p)
    # build_sam2 expects {"model": state_dict} — wrap if raw
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(sd, dict) and "model" in sd):
        torch.save({"model": sd}, ckpt_path)
        print("Checkpoint wrapped for build_sam2")
    from sam2.build_sam import build_sam2
    model = build_sam2("configs/sam2.1_hiera_t512.yaml", ckpt_path, device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _run_teacher_batch(model, imgs_t, points_list):
    B = imgs_t.shape[0]
    imgs_dev = imgs_t.to(DEVICE)
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if DEVICE == "cuda" else torch.no_grad())
    with torch.no_grad(), autocast_ctx:
        backbone_out = model.forward_image(imgs_dev)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(B, -1, *fs)
            for feat, fs in zip(vision_feats[::-1], BB_FEAT_SIZES[::-1])
        ][::-1]
        image_embed  = feats[-1]
        high_res_feats = feats[:-1]
        logits_list = []
        for i, (cx, cy) in enumerate(points_list):
            pt = torch.tensor([[[cx / MODEL_SIZE * MODEL_SIZE,
                                  cy / MODEL_SIZE * MODEL_SIZE]]],
                               dtype=torch.float32, device=DEVICE)
            pt_label = torch.ones(1, 1, dtype=torch.int, device=DEVICE)
            sparse_emb, dense_emb = model.sam_prompt_encoder(
                points=(pt, pt_label), boxes=None, masks=None)
            low_res, _, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed[i].unsqueeze(0),
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=[f[i].unsqueeze(0) for f in high_res_feats],
            )
            logits_list.append(low_res.cpu().float())
    return logits_list  # list of [1, 1, 256, 256]


def generate_soft_labels(teacher, img_paths, mask_paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    TBATCH = 8 if DEVICE == "cuda" else 4
    done = 0
    for start in range(0, len(img_paths), TBATCH):
        b_imgs, b_masks = img_paths[start:start+TBATCH], mask_paths[start:start+TBATCH]
        imgs_t, pts, stems = [], [], []
        for ip, mp in zip(b_imgs, b_masks):
            stem = os.path.splitext(os.path.basename(ip))[0]
            if os.path.exists(os.path.join(out_dir, f"{stem}_logits.npy")):
                done += 1
                continue
            img  = Image.open(ip).convert("RGB")
            mask = np.array(Image.open(mp).convert("L"))
            ys, xs = np.where(mask > 0)
            cx = int(xs.mean()) if len(xs) else mask.shape[1] // 2
            cy = int(ys.mean()) if len(ys) else mask.shape[0] // 2
            imgs_t.append(IMG_NORM_512(img))
            pts.append((cx, cy))
            stems.append(stem)
        if not stems:
            continue
        for stem, logits in zip(stems, _run_teacher_batch(teacher, torch.stack(imgs_t), pts)):
            np.save(os.path.join(out_dir, f"{stem}_logits.npy"),
                    logits[0].numpy().astype(np.float16))
            done += 1
        print(f"  Soft labels: {done}/{len(img_paths)}", flush=True)


# ── dataset ───────────────────────────────────────────────────────────────────
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


class DistillDataset(Dataset):
    def __init__(self, img_paths, mask_paths, soft_dir, img_size=512,
                 color_aug=False, require_soft=True):
        pairs = []
        for ip, mp in zip(img_paths, mask_paths):
            stem = os.path.splitext(os.path.basename(ip))[0]
            soft = os.path.join(soft_dir, f"{stem}_logits.npy")
            if not require_soft or os.path.exists(soft):
                for aug_idx in range(NUM_GEOM_AUGS):
                    pairs.append((ip, mp, soft, aug_idx, stem))
        self.pairs     = pairs
        self.img_norm  = IMG_NORM_1024 if img_size == 1024 else IMG_NORM_512
        self.img_size  = img_size
        self.color_aug = color_aug

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp, soft_path, aug_idx, stem = self.pairs[idx]
        img  = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")

        # same geometric transform applied to image, mask, AND soft label
        img  = _geom_pil(img,  aug_idx)
        mask = _geom_pil(mask, aug_idx)

        # color/noise augmentation on image only (no soft label alignment needed)
        if self.color_aug:
            img = random_color_aug(img)

        mask_np  = np.array(mask)
        cx, cy   = centroid_click(mask_np, jitter=5)
        h, w     = mask_np.shape
        cx_n     = cx / w * self.img_size
        cy_n     = cy / h * self.img_size

        img_t    = self.img_norm(img)
        mask_256 = np.array(mask.resize((256, 256), Image.NEAREST))
        mask_t   = torch.from_numpy((mask_256 > 0).astype(np.float32)).unsqueeze(0)

        # soft label: same geometric aug applied for alignment (zeros if file absent)
        if os.path.exists(soft_path):
            soft_raw = np.load(soft_path).astype(np.float32)  # [1, 256, 256]
            soft_aug = _geom_np(soft_raw[0], aug_idx)
        else:
            soft_aug = np.zeros((256, 256), dtype=np.float32)
        soft = torch.from_numpy(soft_aug).unsqueeze(0)  # [1, 256, 256]

        return img_t, mask_t, torch.tensor([cx_n, cy_n], dtype=torch.float32), soft


# ── losses (verbatim from arcade_v2.ipynb Cell 8) ────────────────────────────
def dice_loss(logits, target, smooth=1e-5):
    pred  = torch.sigmoid(logits)
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
    x    = x.float()
    skel = F.relu(x - soft_open(x))
    for _ in range(10):
        x    = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel  = skel + F.relu(delta - skel * delta)
    return skel


import torch.utils.checkpoint as cp

def soft_skel(x):
    return cp.checkpoint(_skel_inner, x, use_reentrant=False)


def soft_cldice_loss(probs, target, smooth=1.0):
    skel_pred = soft_skel(probs)
    with torch.no_grad():
        skel_true = _skel_inner(target)
    tprec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_true * probs)  + smooth) / (torch.sum(skel_true) + smooth)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


def combined_loss(logits, gt, pos_weight, cldice_w):
    if gt.shape != logits.shape:
        gt = F.interpolate(gt, size=logits.shape[-2:], mode="nearest")
    wbce   = F.binary_cross_entropy_with_logits(
        logits, gt, pos_weight=pos_weight.to(logits.device))
    d      = dice_loss(logits, gt)
    hard   = 0.5 * d + 0.2 * wbce

    if cldice_w > 0:
        probs = torch.sigmoid(logits).float()
        cl    = soft_cldice_loss(probs, gt.float())
    else:
        cl = torch.tensor(0.0)

    return hard + cldice_w * cl, d.item(), (cl.item() if cldice_w > 0 else 0.0)


def distill_loss(student_logits, teacher_logits, gt, pos_weight, cldice_w, use_kd):
    hard_loss, d_val, cl_val = combined_loss(
        student_logits, gt, pos_weight, cldice_w)

    if use_kd:
        soft_targets = torch.sigmoid(teacher_logits)
        if soft_targets.shape != student_logits.shape:
            soft_targets = F.interpolate(soft_targets, size=student_logits.shape[-2:],
                                         mode="bilinear", align_corners=False)
        kd = F.binary_cross_entropy_with_logits(
            student_logits, soft_targets,
            pos_weight=pos_weight.to(student_logits.device))
        total = 0.5 * kd + 0.4 * hard_loss
    else:
        kd    = torch.tensor(0.0)
        total = hard_loss

    return total, d_val, cl_val


# ── MobileSAM student ─────────────────────────────────────────────────────────
def build_mobilesam(teacher_ckpt):
    from mobile_sam import sam_model_registry
    if not os.path.exists(MOBILE_CKPT):
        print("Downloading MobileSAM checkpoint...")
        subprocess.run([
            "wget", "-q", "-O", MOBILE_CKPT,
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        ], check=True)
    model = sam_model_registry["vit_t"](checkpoint=MOBILE_CKPT)

    # transplant fine-tuned decoder from teacher
    teacher_sd = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    if isinstance(teacher_sd, dict) and set(teacher_sd.keys()) == {"model"}:
        teacher_sd = teacher_sd["model"]
    decoder_sd  = {k.replace("sam_mask_decoder.", ""): v
                   for k, v in teacher_sd.items()
                   if k.startswith("sam_mask_decoder.")}
    prompt_sd   = {k.replace("sam_prompt_encoder.", ""): v
                   for k, v in teacher_sd.items()
                   if k.startswith("sam_prompt_encoder.")}
    missing, unexpected = model.mask_decoder.load_state_dict(decoder_sd, strict=False)
    print(f"  Decoder transplant — missing: {missing[:3]}, unexpected: {unexpected[:3]}")
    missing, _ = model.prompt_encoder.load_state_dict(prompt_sd, strict=False)

    return model.to(DEVICE)


def forward_mobilesam(model, imgs, pts):
    B = imgs.shape[0]
    image_embed = model.image_encoder(imgs)  # [B, 256, 64, 64] at 1024; [B, 256, 32, 32] at 512
    if image_embed.shape[-1] != 64:
        image_embed = F.interpolate(image_embed, size=(64, 64), mode="bilinear", align_corners=False)
    logits_list = []
    for i, (cx, cy) in enumerate(pts):
        pt = torch.tensor([[[cx.item(), cy.item()]]],
                          dtype=torch.float32, device=DEVICE)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=DEVICE)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=(pt, pt_label), boxes=None, masks=None)
        lm, _ = model.mask_decoder(
            image_embeddings=image_embed[i].unsqueeze(0),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        logits_list.append(lm)
    return torch.cat(logits_list, dim=0)  # [B, 1, 256, 256]


# ── RepViT-SAM student ────────────────────────────────────────────────────────
class RepViTSAM(nn.Module):
    def __init__(self, teacher_ckpt):
        super().__init__()
        import timm
        # RepViT-M1.0 at 512×512 produces [B, 384, 16, 16] at stride 32
        # We need [B, 256, 32, 32] for SAM decoder (stride 16 at 512 input)
        # Use repvit_m1 and extract at stride-16 stage
        self.encoder = timm.create_model(
            "repvit_m1", pretrained=True, features_only=True,
            out_indices=(3,))  # stage index giving ~stride 16 output

        # project to [B, 256, H, W] for SAM decoder
        # repvit_m1 stage-3 output: [B, 192, H/16, W/16]
        encoder_ch = self.encoder.feature_info.channels()[-1]
        self.neck = nn.Conv2d(encoder_ch, 256, kernel_size=1, bias=False)

        # SAM decoder from mobile-sam
        from mobile_sam import sam_model_registry
        _sam = sam_model_registry["vit_t"](checkpoint=MOBILE_CKPT)
        self.mask_decoder   = _sam.mask_decoder
        self.prompt_encoder = _sam.prompt_encoder

        # transplant fine-tuned decoder
        teacher_sd = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        if isinstance(teacher_sd, dict) and set(teacher_sd.keys()) == {"model"}:
            teacher_sd = teacher_sd["model"]
        decoder_sd  = {k.replace("sam_mask_decoder.", ""): v
                       for k, v in teacher_sd.items()
                       if k.startswith("sam_mask_decoder.")}
        prompt_sd   = {k.replace("sam_prompt_encoder.", ""): v
                       for k, v in teacher_sd.items()
                       if k.startswith("sam_prompt_encoder.")}
        self.mask_decoder.load_state_dict(decoder_sd, strict=False)
        self.prompt_encoder.load_state_dict(prompt_sd, strict=False)

    def forward(self, imgs, pts):
        B = imgs.shape[0]
        feats = self.encoder(imgs)[-1]      # [B, C, H/16, W/16]
        embed = self.neck(feats)            # [B, 256, H/16, W/16]
        # upsample to 64×64 to match MobileSAM prompt encoder's expected size
        embed = F.interpolate(embed, size=(64, 64), mode="bilinear", align_corners=False)

        logits_list = []
        for i, (cx, cy) in enumerate(pts):
            pt = torch.tensor([[[cx.item(), cy.item()]]],
                              dtype=torch.float32, device=imgs.device)
            pt_label = torch.ones(1, 1, dtype=torch.int, device=imgs.device)
            sparse_emb, dense_emb = self.prompt_encoder(
                points=(pt, pt_label), boxes=None, masks=None)
            lm, _ = self.mask_decoder(
                image_embeddings=embed[i].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            logits_list.append(lm)
        return torch.cat(logits_list, dim=0)  # [B, 1, 256, 256]


def build_repvitsam(teacher_ckpt):
    if not os.path.exists(MOBILE_CKPT):
        print("Downloading MobileSAM checkpoint...")
        subprocess.run([
            "wget", "-q", "-O", MOBILE_CKPT,
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        ], check=True)
    model = RepViTSAM(teacher_ckpt).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  RepViT-SAM total params: {n_params:.1f}M")
    return model


# ── feature adapter (training-only, discarded at inference) ──────────────────
class FeatureAdapter(nn.Module):
    """Projects student image embedding into teacher feature space via 1×1 conv.
    Added to the optimizer during training; never saved with the student checkpoint."""
    def __init__(self, in_ch=256, out_ch=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)


def extract_teacher_embed(teacher, imgs_1024):
    """Run teacher encoder on a batch, return image_embed [B, 256, H, W]."""
    B = imgs_1024.shape[0]
    imgs_512 = F.interpolate(imgs_1024, size=(MODEL_SIZE, MODEL_SIZE),
                             mode="bilinear", align_corners=False)
    backbone_out = teacher.forward_image(imgs_512)
    _, vision_feats, _, _ = teacher._prepare_backbone_features(backbone_out)
    if teacher.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + teacher.no_mem_embed
    feats = [
        feat.permute(1, 2, 0).view(B, -1, *fs)
        for feat, fs in zip(vision_feats[::-1], BB_FEAT_SIZES[::-1])
    ][::-1]
    return feats[-1].float()  # [B, 256, 32, 32]


# ── GCS uploader (background thread, verbatim pattern from arcade_v2.ipynb) ──
import threading, queue

_gcs_q: queue.Queue = queue.Queue()

def _gcs_worker():
    while True:
        item = _gcs_q.get()
        if item is None:
            break
        local, remote = item
        subprocess.run(["gsutil", "cp", local, remote], capture_output=True)
        _gcs_q.task_done()

_gcs_thread = threading.Thread(target=_gcs_worker, daemon=True)
_gcs_thread.start()

def upload_async(local, remote):
    _gcs_q.put((local, remote))


# ── training ──────────────────────────────────────────────────────────────────
def train(args):
    EPOCHS     = int(os.environ.get("MAX_EPOCHS", 50))
    PATIENCE   = int(os.environ.get("ES_PATIENCE", 10))
    BATCH      = int(os.environ.get("BATCH_SIZE", 4))
    BASE_LR    = 3e-4
    WARMUP_EP  = 3
    WD         = 0.01
    GRAD_CLIP  = 0.5
    FG_FRAC    = 0.05
    pos_weight = torch.tensor([(1 - FG_FRAC) / FG_FRAC])

    teacher_ckpt = "/home/jupyter/medsam2_arcade_v2.pt"
    if not os.path.exists(teacher_ckpt):
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="Elakiya17/CA-SAM2", filename="medsam2_arcade_v2.pt",
                        local_dir="/home/jupyter")

    use_kd     = args.ablation >= 2
    use_cldice = args.ablation >= 3

    img_paths_all  = sorted(glob.glob(DATA_DIR + "/images/*.png"))
    mask_paths_all = sorted(glob.glob(DATA_DIR + "/masks/*.png"))

    # only generate soft labels when KD loss is actually used
    if use_kd:
        if not os.path.exists(SOFT_DIR) or len(glob.glob(SOFT_DIR + "/*.npy")) < len(img_paths_all):
            print("Generating soft labels from HF teacher (one-time)...")
            teacher = load_teacher(teacher_ckpt)
            generate_soft_labels(teacher, img_paths_all, mask_paths_all, SOFT_DIR)
            del teacher
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print("Soft labels ready.")
    else:
        print("Ablation=1 (GT only) — skipping soft label generation.")

    # ── 900/100 split from 1000 training images (seed=42 for reproducibility) ──
    # The 200 ARCADE val images are the held-out test set and are NEVER used here.
    indices = list(range(len(img_paths_all)))
    random.Random(42).shuffle(indices)
    n_val    = 100
    val_idx  = indices[:n_val]
    trn_idx  = indices[n_val:]
    img_paths_train  = [img_paths_all[i]  for i in trn_idx]
    mask_paths_train = [mask_paths_all[i] for i in trn_idx]
    img_paths_val    = [img_paths_all[i]  for i in val_idx]
    mask_paths_val   = [mask_paths_all[i] for i in val_idx]
    print(f"Split: {len(img_paths_train)} train / {len(img_paths_val)} val (early-stop) / "
          f"200 held-out test (arcade_test, never touched)")

    img_size = MOBILE_SIZE if args.student == "mobilesam" else MODEL_SIZE
    ds  = DistillDataset(img_paths_train, mask_paths_train, SOFT_DIR,
                         img_size=img_size, color_aug=True, require_soft=use_kd)
    dl  = DataLoader(ds, batch_size=BATCH, shuffle=True,
                     num_workers=4, pin_memory=True, drop_last=True)
    print(f"Train dataset: {len(ds)} samples ({len(img_paths_train)} images × {NUM_GEOM_AUGS} geom augs + color aug), "
          f"{len(dl)} batches/epoch")

    # build model
    if args.student == "mobilesam":
        model = build_mobilesam(teacher_ckpt)
    else:
        model = build_repvitsam(teacher_ckpt)

    trainable = [p for p in model.parameters() if p.requires_grad]

    # feature KD: adapter + frozen teacher encoder
    use_feat_kd = bool(int(os.environ.get("USE_FEAT_KD", 0)))
    FEAT_KD_W   = float(os.environ.get("FEAT_KD_W", 0.1))
    feat_adapter = None
    teacher_feat = None
    if use_feat_kd:
        feat_adapter = FeatureAdapter(256, 256).to(DEVICE)
        trainable   += list(feat_adapter.parameters())
        teacher_feat = load_teacher(teacher_ckpt)
        teacher_feat.eval()
        for p in teacher_feat.parameters():
            p.requires_grad_(False)
        print(f"Feature KD enabled — adapter 256→256, weight={FEAT_KD_W}")

    n_train = sum(p.numel() for p in trainable) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Trainable: {n_train:.1f}M / {n_total:.1f}M")

    optimizer = optim.AdamW(trainable, lr=BASE_LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    version   = os.environ.get("MODEL_VERSION", "v3")
    suffix    = f"{args.student}_abl{args.ablation}_{version}"
    ckpt_path = f"/home/jupyter/{suffix}.pt"
    best_dice  = -1.0
    no_improve = 0

    # val set = 100-image internal split from train (early stopping signal only)
    img_norm_val = IMG_NORM_1024 if args.student == "mobilesam" else IMG_NORM_512
    val_img_size = MOBILE_SIZE if args.student == "mobilesam" else MODEL_SIZE

    def quick_val_dice(model):
        model.eval()
        scores = []
        with torch.no_grad():
            for ip, mp in zip(img_paths_val, mask_paths_val):
                img  = Image.open(ip).convert("RGB")
                mask = np.array(Image.open(mp).convert("L"))
                ys, xs = np.where(mask > 0)
                cx = int(xs.mean()) if len(xs) else mask.shape[1] // 2
                cy = int(ys.mean()) if len(ys) else mask.shape[0] // 2
                img_t = img_norm_val(img).unsqueeze(0).to(DEVICE)
                pts   = [(torch.tensor(cx / mask.shape[1] * val_img_size),
                          torch.tensor(cy / mask.shape[0] * val_img_size))]
                autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                                if DEVICE == "cuda" else torch.no_grad())
                with autocast_ctx:
                    logits = (forward_mobilesam(model, img_t, pts)
                              if args.student == "mobilesam" else model(img_t, pts))
                pred = (logits[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
                pred = np.array(Image.fromarray(pred * 255).resize(
                    (mask.shape[1], mask.shape[0]), Image.NEAREST)) > 127
                gt   = mask > 0
                denom = pred.sum() + gt.sum()
                scores.append((2 * (pred & gt).sum() / denom) if denom > 0 else 1.0)
        model.train()
        return float(np.mean(scores))

    for epoch in range(1, EPOCHS + 1):
        # warmup
        if epoch <= WARMUP_EP:
            lr = BASE_LR * (epoch / WARMUP_EP)
            for g in optimizer.param_groups:
                g["lr"] = lr

        cldice_w = 0.0
        if use_cldice:
            cldice_w = min(0.3, 0.3 * max(0, epoch - 3) / 5)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for imgs, masks, pts, soft_logits in dl:
            imgs_d   = imgs.to(DEVICE)
            masks_d  = masks.to(DEVICE)
            soft_d   = soft_logits.to(DEVICE)
            pts_list = [(pts[i][0], pts[i][1]) for i in range(len(pts))]

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                            if DEVICE == "cuda" else torch.no_grad())
            with autocast_ctx:
                if args.student == "mobilesam":
                    logits = forward_mobilesam(model, imgs_d, pts_list)
                else:
                    logits = model(imgs_d, pts_list)

                loss, _, _ = distill_loss(
                    logits, soft_d, masks_d, pos_weight, cldice_w, use_kd)

                # foreground-weighted encoder feature matching
                if use_feat_kd and feat_adapter is not None:
                    with torch.no_grad():
                        t_embed = extract_teacher_embed(teacher_feat, imgs_d)
                    s_embed  = model.image_encoder(imgs_d)       # [B, 256, 64, 64]
                    s_proj   = feat_adapter(s_embed)             # [B, 256, 64, 64]
                    s_down   = F.interpolate(s_proj, size=t_embed.shape[-2:],
                                             mode="bilinear", align_corners=False)
                    fg       = F.interpolate(masks_d, size=t_embed.shape[-2:],
                                             mode="nearest")
                    fg_w     = 1.0 + 9.0 * fg                   # 10× weight on vessel pixels
                    loss_feat = ((s_down - t_embed) ** 2 * fg_w).mean()
                    loss      = loss + FEAT_KD_W * loss_feat

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        if epoch > WARMUP_EP:
            scheduler.step()

        avg = epoch_loss / len(dl)
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"cldice_w={cldice_w:.3f}  "
              f"t={time.time()-t0:.0f}s", flush=True)

        val_dice = quick_val_dice(model)
        print(f"  → val Dice: {val_dice:.4f}  best: {best_dice:.4f}  patience: {no_improve}/{PATIENCE}", flush=True)

        if val_dice > best_dice:
            best_dice  = val_dice
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            upload_async(ckpt_path, f"{BUCKET}/checkpoints/{suffix}.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    _gcs_q.join()
    print(f"Training complete. Best val Dice: {best_dice:.4f}")
    print(f"Checkpoint: {BUCKET}/checkpoints/{suffix}.pt")


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(args):
    # Evaluate on the 200 held-out ARCADE images (arcade_test).
    # These images are never downloaded to the training VM during train().
    img_paths  = sorted(glob.glob("/home/jupyter/arcade_test/images/*.png"))
    mask_paths = sorted(glob.glob("/home/jupyter/arcade_test/masks/*.png"))
    if not img_paths:
        raise RuntimeError("arcade_test not found — did the setup script download it?")
    cc = centroid_click  # reuse the function defined above

    teacher_ckpt = "/home/jupyter/medsam2_arcade_v2.pt"
    version   = os.environ.get("MODEL_VERSION", "")
    suffix    = f"{args.student}_abl{args.ablation}" + (f"_{version}" if version else "")
    ckpt_path = f"/home/jupyter/{suffix}.pt"
    if not os.path.exists(ckpt_path):
        subprocess.run(["gsutil", "cp",
                        f"{BUCKET}/checkpoints/{suffix}.pt", ckpt_path], check=True)

    if args.student == "mobilesam":
        model = build_mobilesam(teacher_ckpt)
    else:
        model = build_repvitsam(teacher_ckpt)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False),
                          strict=False)
    model.eval()

    img_size = MOBILE_SIZE if args.student == "mobilesam" else MODEL_SIZE
    img_norm  = IMG_NORM_1024 if img_size == 1024 else IMG_NORM_512

    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for ip, mp in zip(img_paths, mask_paths):
            img    = Image.open(ip).convert("RGB")
            mask   = np.array(Image.open(mp).convert("L"))
            cx, cy = cc(mask, jitter=0)

            img_t  = img_norm(img).unsqueeze(0).to(DEVICE)
            pts    = [(torch.tensor(cx / mask.shape[1] * img_size),
                       torch.tensor(cy / mask.shape[0] * img_size))]

            autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                            if DEVICE == "cuda" else torch.no_grad())
            with autocast_ctx:
                if args.student == "mobilesam":
                    logits = forward_mobilesam(model, img_t, pts)
                else:
                    logits = model(img_t, pts)

            pred = (logits[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
            pred = np.array(Image.fromarray(pred * 255).resize(
                (mask.shape[1], mask.shape[0]), Image.NEAREST)) > 127

            gt   = (mask > 0)
            inter = (pred & gt).sum()
            union = (pred | gt).sum()
            denom = pred.sum() + gt.sum()

            dice = (2 * inter / denom) if denom > 0 else 1.0
            iou  = (inter / union) if union > 0 else 1.0
            dice_scores.append(dice)
            iou_scores.append(iou)

    mean_dice = np.mean(dice_scores)
    std_dice  = np.std(dice_scores)
    mean_iou  = np.mean(iou_scores)
    print(f"\n=== {suffix} — HELD-OUT TEST SET (200 ARCADE val images, never seen during training) ===")
    print(f"  Dice: {mean_dice:.3f} ± {std_dice:.3f}")
    print(f"  IoU:  {mean_iou:.3f}")

    result = {"student": args.student, "ablation": args.ablation,
              "dice_mean": mean_dice, "dice_std": std_dice, "iou_mean": mean_iou}
    out_path = f"/home/jupyter/results_{suffix}.json"
    import json
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    upload_async(out_path, f"{BUCKET}/results/distillation/results_{suffix}.json")
    _gcs_q.join()
    return mean_dice


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student",  choices=["mobilesam", "repvitsam"],
                        default="mobilesam")
    parser.add_argument("--ablation", type=int, choices=[1, 2, 3, 4], default=4)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Student: {args.student}  |  Ablation: {args.ablation}")
    print(f"{'='*60}\n")

    train(args)
    evaluate(args)
