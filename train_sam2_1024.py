"""
SAM2.1 Hiera-Tiny fine-tuned at 1024x1024 on ARCADE coronary angiography.

Architecture: facebookresearch/sam2 (Meta, NOT MedSAM2).
Config: sam2.1_hiera_t.yaml → image_size: 1024 (native).
Base weights: facebook/sam2.1-hiera-tiny (HuggingFace).

Protocol:
  - Frozen encoder except last 2 Hiera blocks (blocks[10,11]) + FPN neck
  - Discriminative LRs: decoder 5e-5 | neck 1e-5 | trunk blocks[10,11] 5e-6
  - Loss: 0.5×Dice + 0.2×wBCE + clDice (warmup epochs 3-8 → 0.3×)
  - 10× offline geometric augmentation (900 images → 9,000 training samples)
  - 50 epochs max, early stopping patience=10 on 100-image internal val
  - Centroid click prompt (jitter ±5px in original image space)

Goal: isolate whether 1024×1024 resolution (not distillation) explains MobileSAM v4's
0.806 Dice advantage over the 512×512 CA-SAM2 teacher at 0.767.
"""
import os, sys, glob, time, random, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as cp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

SAM2_REPO = "/opt/sam2_meta"
sys.path.insert(0, SAM2_REPO)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = 1024
DATA_DIR  = "/home/jupyter/arcade_train"
TEST_DIR  = "/home/jupyter/arcade_test"
BUCKET    = "gs://coronary-angio-v2"
HF_REPO   = os.environ.get("HF_REPO", "Elakiya17/fluoroscopy-sam2")
VERSION   = os.environ.get("MODEL_VERSION", "sam2_fluoroscopy_1024")
EPOCHS    = int(os.environ.get("MAX_EPOCHS", 50))
PATIENCE  = int(os.environ.get("ES_PATIENCE", 10))
BATCH     = int(os.environ.get("BATCH_SIZE", 4))
BASE_LR   = 5e-5

# Hiera-Tiny: stages [1,2,7,2] = 12 blocks total. blocks[10,11] = stage 4 (deepest).
UNFROZEN_BLOCKS = {"blocks.10", "blocks.11"}

IMG_NORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Augmentation: 10× offline (1 original + 9 geometric transforms) ──────────
# Same transform applied identically to image and mask (replay-style consistency).
# Transforms chosen for fluoroscopy: gantry angle variation (rotations),
# patient orientation (flips), and scale variation (zoom).
SUFFIXES = [
    "orig",          # 0 — identity
    "hf",            # 1 — flip left-right
    "vf",            # 2 — flip top-bottom
    "hf_vf",         # 3 — flip both (= 180° rotation)
    "rm20",          # 4 — rotate −20°
    "rp20",          # 5 — rotate +20°
    "hf_rm20",       # 6 — flip-LR then rotate −20°
    "hf_rp20",       # 7 — flip-LR then rotate +20°
    "vf_rm10",       # 8 — flip-TB then rotate −10°
    "zoom085",       # 9 — zoom out to 85%, pad to original size
]

def _zoom(img_pil, msk_pil, scale=0.85):
    W, H = img_pil.size
    nw, nh = int(W * scale), int(H * scale)
    img_s = img_pil.resize((nw, nh), Image.BICUBIC)
    msk_s = msk_pil.resize((nw, nh), Image.NEAREST)
    pad_l, pad_t = (W - nw) // 2, (H - nh) // 2
    img_p = Image.new("RGB", (W, H), (0, 0, 0))
    msk_p = Image.new("L",   (W, H), 0)
    img_p.paste(img_s, (pad_l, pad_t))
    msk_p.paste(msk_s, (pad_l, pad_t))
    return img_p, msk_p

def augment_pair(img_pil, msk_pil):
    hf  = lambda i, m: (i.transpose(Image.FLIP_LEFT_RIGHT),
                        m.transpose(Image.FLIP_LEFT_RIGHT))
    vf  = lambda i, m: (i.transpose(Image.FLIP_TOP_BOTTOM),
                        m.transpose(Image.FLIP_TOP_BOTTOM))
    rot = lambda i, m, a: (i.rotate(a, resample=Image.BICUBIC),
                           m.rotate(a, resample=Image.NEAREST))
    i_hf,  m_hf  = hf(img_pil, msk_pil)
    i_vf,  m_vf  = vf(img_pil, msk_pil)
    i_hfvf, m_hfvf = vf(i_hf, m_hf)
    i_rm20, m_rm20 = rot(img_pil, msk_pil, -20)
    i_rp20, m_rp20 = rot(img_pil, msk_pil,  20)
    i_z,   m_z   = _zoom(img_pil, msk_pil, 0.85)
    return [
        (img_pil,  msk_pil),
        (i_hf,     m_hf),
        (i_vf,     m_vf),
        (i_hfvf,   m_hfvf),
        (i_rm20,   m_rm20),
        (i_rp20,   m_rp20),
        rot(i_hf,  m_hf,  -20),
        rot(i_hf,  m_hf,   20),
        rot(i_vf,  m_vf,  -10),
        (i_z,      m_z),
    ]

def generate_aug10k(src_img_paths, src_msk_paths, out_dir):
    img_out = os.path.join(out_dir, "images")
    msk_out = os.path.join(out_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)
    stem0 = os.path.splitext(os.path.basename(src_img_paths[0]))[0]
    if os.path.exists(os.path.join(img_out, f"{stem0}_orig.png")):
        print("  Aug10k already present — skipping generation")
        return
    n = len(src_img_paths) * len(SUFFIXES)
    print(f"  Generating {n} augmented images ({len(src_img_paths)} × {len(SUFFIXES)})...")
    for ip, mp in zip(src_img_paths, src_msk_paths):
        stem = os.path.splitext(os.path.basename(ip))[0]
        img  = Image.open(ip).convert("RGB")
        msk  = Image.open(mp).convert("L")
        for suffix, (ai, am) in zip(SUFFIXES, augment_pair(img, msk)):
            ai.save(os.path.join(img_out, f"{stem}_{suffix}.png"))
            mb = (np.array(am) > 0).astype(np.uint8) * 255
            Image.fromarray(mb).save(os.path.join(msk_out, f"{stem}_{suffix}.png"))
    print(f"  Done: {len(os.listdir(img_out))} images")


def centroid_click(mask_np, jitter=5):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return mask_np.shape[1]//2, mask_np.shape[0]//2
    # Random foreground pixel + jitter — matches v4 distillation training exactly
    idx = np.random.randint(len(xs))
    cx  = int(xs[idx]) + random.randint(-jitter, jitter)
    cy  = int(ys[idx]) + random.randint(-jitter, jitter)
    return np.clip(cx, 0, mask_np.shape[1]-1), np.clip(cy, 0, mask_np.shape[0]-1)


class ArcadeDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img     = Image.open(self.img_paths[idx]).convert("RGB")
        msk_pil = Image.open(self.mask_paths[idx]).convert("L")
        msk_np  = np.array(msk_pil)
        img_t   = IMG_NORM(img)
        msk_t   = torch.from_numpy((msk_np > 0).astype(np.float32)).unsqueeze(0)
        cx, cy  = centroid_click(msk_np, jitter=10)
        pt      = torch.tensor([cx / msk_np.shape[1] * IMG_SIZE,
                                 cy / msk_np.shape[0] * IMG_SIZE], dtype=torch.float32)
        return img_t, msk_t, pt


# ── Losses ────────────────────────────────────────────────────────────────────
def dice_loss(logits, target, smooth=1e-5):
    pred  = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


def soft_erode(x):
    return torch.min(-F.max_pool2d(-x, (3,1), (1,1), (1,0)),
                     -F.max_pool2d(-x, (1,3), (1,1), (0,1)))

def soft_dilate(x):
    return F.max_pool2d(x, (3,3), (1,1), (1,1))

def soft_open(x):
    return soft_dilate(soft_erode(x))

def _skel_inner(x):
    x    = x.float()
    skel = F.relu(x - soft_open(x))
    for _ in range(10):
        x     = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel  = skel + F.relu(delta - skel * delta)
    return skel

def soft_skel(x):
    return cp.checkpoint(_skel_inner, x, use_reentrant=False)

def cldice_loss(logits, target, smooth=1.0):
    with torch.autocast(device_type="cuda", enabled=False):
        probs     = torch.sigmoid(logits.float())
        skel_pred = soft_skel(probs)
        with torch.no_grad():
            skel_true = _skel_inner(target.float())
        tprec = (skel_pred * target).sum() + smooth
        tprec /= skel_pred.sum() + smooth
        tsens = (skel_true * probs).sum() + smooth
        tsens /= skel_true.sum() + smooth
        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)

def combined_loss(logits, target, pos_weight, cldice_w):
    d = dice_loss(logits, target)
    b = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    cl = cldice_loss(logits, target) if cldice_w > 0 else torch.tensor(0.0, device=logits.device)
    return 0.5 * d + 0.2 * b + cldice_w * cl, d.item(), cl.item()


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(base_ckpt):
    from sam2.build_sam import build_sam2
    # Uses Meta's sam2.1_hiera_t.yaml → image_size: 1024
    model = build_sam2("sam2.1_hiera_t.yaml", base_ckpt,
                       device=DEVICE, apply_postprocessing=False)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  SAM2.1 Hiera-Tiny: {n:.1f}M params  |  image_size=1024  |  device={DEVICE}")
    return model


def build_param_groups(model):
    decoder_p, neck_p, trunk_p = [], [], []
    for name, p in model.named_parameters():
        if "image_encoder.trunk" in name:
            if any(b in name for b in UNFROZEN_BLOCKS):
                p.requires_grad_(True)
                trunk_p.append(p)
            else:
                p.requires_grad_(False)
        elif "image_encoder.neck" in name:
            p.requires_grad_(True)
            neck_p.append(p)
        else:
            p.requires_grad_(True)
            decoder_p.append(p)

    def split_wd(params):
        decay, no_decay = [], []
        for p in params:
            (no_decay if p.ndim <= 1 else decay).append(p)
        return decay, no_decay

    groups = []
    for params, lr in [(decoder_p, BASE_LR), (neck_p, BASE_LR*0.2), (trunk_p, BASE_LR*0.1)]:
        d, nd = split_wd(params)
        groups += [{"params": d,  "lr": lr, "weight_decay": 0.01},
                   {"params": nd, "lr": lr, "weight_decay": 0.0}]

    n_train = sum(p.numel() for g in groups for p in g["params"]) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Trainable: {n_train:.1f}M / {n_total:.1f}M")
    print(f"  Decoder LR: {BASE_LR:.0e} | Neck LR: {BASE_LR*0.2:.0e} | Trunk blocks[10,11] LR: {BASE_LR*0.1:.0e}")
    return groups


def set_train_mode(model):
    model.train()
    model.image_encoder.trunk.eval()
    model.image_encoder.trunk.blocks[10].train()
    model.image_encoder.trunk.blocks[11].train()
    model.image_encoder.neck.train()


# ── SAM2 forward ──────────────────────────────────────────────────────────────
def forward_sam2(model, imgs, pts, high_res=True):
    B = imgs.shape[0]
    # Call image_encoder directly (not forward_image) so gradients flow through
    # unfrozen trunk blocks[10,11] and neck during training.
    backbone_out = model.image_encoder(imgs)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    # no_mem_embed is a video-mode attribute; safe to skip in image-only fine-tuning
    if getattr(model, "directly_add_no_mem_embed", False):
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    # vision_feats: list of [HW, B, C], finest→coarsest; feat_sizes matches
    feats = [feat.permute(1,2,0).view(B, -1, *fs)
             for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
    image_embed    = feats[-1]
    high_res_feats = feats[:-1]

    logits_list = []
    for i, pt in enumerate(pts):
        cx, cy   = pt[0].item(), pt[1].item()
        point    = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=DEVICE)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=DEVICE)
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(point, pt_label), boxes=None, masks=None)
        lm, _, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed[i].unsqueeze(0),
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False, repeat_image=False,
            high_res_features=[f[i].unsqueeze(0) for f in high_res_feats] if high_res else None,
        )
        logits_list.append(lm)
    return torch.cat(logits_list, dim=0)


# ── Val / test eval ───────────────────────────────────────────────────────────
def eval_dice(model, img_paths, mask_paths, split_name="val"):
    model.eval()
    scores = []
    with torch.no_grad():
        for ip, mp in zip(img_paths, mask_paths):
            img  = Image.open(ip).convert("RGB")
            mask = np.array(Image.open(mp).convert("L"))
            cx, cy = centroid_click(mask, jitter=0)
            img_t  = IMG_NORM(img).unsqueeze(0).to(DEVICE)
            pt     = [torch.tensor([cx/mask.shape[1]*IMG_SIZE,
                                    cy/mask.shape[0]*IMG_SIZE])]
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(DEVICE=="cuda")):
                logits = forward_sam2(model, img_t, pt)
            pred = (logits[0,0].float().cpu().numpy() > 0.0).astype(np.uint8)
            pred = np.array(Image.fromarray(pred*255).resize(
                (mask.shape[1], mask.shape[0]), Image.NEAREST)) // 255
            gt   = (mask > 0).astype(np.uint8)
            denom = pred.sum() + gt.sum()
            scores.append(float(2*(pred*gt).sum()/denom) if denom > 0 else 1.0)
    model.train()
    return float(np.mean(scores)), np.array(scores)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    base_ckpt = "/home/jupyter/sam2.1_hiera_tiny.pt"
    if not os.path.exists(base_ckpt):
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="facebook/sam2.1-hiera-tiny",
                        filename="sam2.1_hiera_tiny.pt",
                        local_dir="/home/jupyter")

    model = build_model(base_ckpt)

    # 900/100 split (seed=42, same as distillation experiments)
    img_paths_all  = sorted(glob.glob(DATA_DIR + "/images/*.png"))
    msk_paths_all  = sorted(glob.glob(DATA_DIR + "/masks/*.png"))
    indices = list(range(len(img_paths_all)))
    random.Random(42).shuffle(indices)
    val_idx = indices[:100]
    trn_idx = indices[100:]
    img_trn = [img_paths_all[i] for i in trn_idx]
    msk_trn = [msk_paths_all[i] for i in trn_idx]
    img_val = [img_paths_all[i] for i in val_idx]
    msk_val = [msk_paths_all[i] for i in val_idx]
    print(f"Split: {len(img_trn)} train / {len(img_val)} internal val / 200 held-out test")

    # 10× offline augmentation (900 images → 9,000 training samples)
    aug_dir = "/home/jupyter/arcade_aug10k"
    generate_aug10k(img_trn, msk_trn, aug_dir)
    aug_img = sorted(glob.glob(aug_dir + "/images/*.png"))
    aug_msk = sorted(glob.glob(aug_dir + "/masks/*.png"))
    print(f"Augmented training set: {len(aug_img)} images")

    fg_mean    = float(np.mean([(np.array(Image.open(m).convert("L").resize(
                   (256,256), Image.NEAREST)) > 0).mean() for m in aug_msk[:200]]))
    pos_weight = torch.tensor([(1-fg_mean)/fg_mean]).to(DEVICE)
    print(f"Foreground: {fg_mean*100:.1f}%  pos_weight: {pos_weight.item():.1f}×")

    ds = ArcadeDataset(aug_img, aug_msk)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    print(f"DataLoader: {len(ds)} samples | {len(dl)} batches/epoch | batch_size={BATCH}")

    param_groups = build_param_groups(model)
    optimizer    = optim.AdamW(param_groups, betas=(0.9, 0.999))
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler       = torch.amp.GradScaler("cuda", enabled=(DEVICE=="cuda"))

    ckpt_path  = f"/home/jupyter/{VERSION}.pt"
    best_dice  = -1.0
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        set_train_mode(model)
        cldice_w   = min(0.3, 0.3 * max(0, epoch-3) / 5)
        epoch_loss = 0.0
        t0         = time.time()

        for imgs, masks, pts in dl:
            imgs_d   = imgs.to(DEVICE)
            masks_d  = masks.to(DEVICE)
            pts_list = [pts[i] for i in range(len(pts))]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(DEVICE=="cuda")):
                logits = forward_sam2(model, imgs_d, pts_list)
                tgt    = F.interpolate(masks_d.float(), size=logits.shape[-2:], mode="nearest")
                loss, _, _ = combined_loss(logits, tgt, pos_weight, cldice_w)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if torch.isfinite(loss):
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], 0.5)
                scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        val_dice, _ = eval_dice(model, img_val, msk_val)
        improved    = val_dice > best_dice
        if improved:
            best_dice  = val_dice
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss/len(dl):.4f}  "
              f"cldice_w={cldice_w:.2f}  lr={scheduler.get_last_lr()[0]:.2e}  t={int(time.time()-t0)}s")
        print(f"  → val Dice: {val_dice:.4f}  best: {best_dice:.4f}  patience: {no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")

    # ── Held-out test eval ────────────────────────────────────────────────────
    print(f"\n=== {VERSION} — HELD-OUT TEST (200 ARCADE val images) ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    test_img  = sorted(glob.glob(TEST_DIR + "/images/*.png"))
    test_msk  = sorted(glob.glob(TEST_DIR + "/masks/*.png"))
    dice_mean, dice_arr = eval_dice(model, test_img, test_msk, "test")
    iou_scores = []
    # recompute IoU
    model.eval()
    with torch.no_grad():
        for ip, mp in zip(test_img, test_msk):
            img  = Image.open(ip).convert("RGB")
            mask = np.array(Image.open(mp).convert("L"))
            cx, cy = centroid_click(mask, jitter=0)
            img_t  = IMG_NORM(img).unsqueeze(0).to(DEVICE)
            pt     = [torch.tensor([cx/mask.shape[1]*IMG_SIZE, cy/mask.shape[0]*IMG_SIZE])]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
                logits = forward_sam2(model, img_t, pt)
            pred = (logits[0,0].float().cpu().numpy() > 0.0).astype(np.uint8)
            pred = np.array(Image.fromarray(pred*255).resize(
                (mask.shape[1], mask.shape[0]), Image.NEAREST)) // 255
            gt   = (mask > 0).astype(np.uint8)
            iou_scores.append(float((pred*gt).sum() / ((pred+gt-pred*gt).sum()+1e-6)))

    print(f"  Dice: {dice_arr.mean():.3f} ± {dice_arr.std():.3f}")
    print(f"  IoU:  {np.mean(iou_scores):.3f}")
    print(f"  Checkpoint: {ckpt_path}")

    subprocess.run(["gsutil", "cp", ckpt_path,
                    f"{BUCKET}/checkpoints/{VERSION}.pt"], check=True)
    print(f"  GCS: {BUCKET}/checkpoints/{VERSION}.pt")

    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=ckpt_path,
                            path_in_repo=f"{VERSION}.pt",
                            repo_id=HF_REPO, repo_type="model")
        print(f"  HuggingFace: {HF_REPO}/{VERSION}.pt")
    except Exception as e:
        print(f"  HF push skipped (create repo '{HF_REPO}' first): {e}")


if __name__ == "__main__":
    train()
