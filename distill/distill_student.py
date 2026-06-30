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
"""

import argparse, os, sys, glob, subprocess, random, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── paths ────────────────────────────────────────────────────────────────────
BUCKET        = "gs://coronary-angio-v2"
DATA_DIR      = "/home/jupyter/arcade_train"
SOFT_DIR      = "/home/jupyter/soft_labels"
MOBILE_CKPT   = "/home/jupyter/mobile_sam.pt"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZE    = 512   # teacher / repvit input size
MOBILE_SIZE   = 1024  # MobileSAM's native input size

# ── augmentation ─────────────────────────────────────────────────────────────
AUG_TRANSFORMS = [
    lambda img, mask: (img, mask),
    lambda img, mask: (img.transpose(Image.FLIP_LEFT_RIGHT),
                       mask.transpose(Image.FLIP_LEFT_RIGHT)),
    lambda img, mask: (img.transpose(Image.FLIP_TOP_BOTTOM),
                       mask.transpose(Image.FLIP_TOP_BOTTOM)),
    lambda img, mask: (img.rotate(20), mask.rotate(20)),
    lambda img, mask: (img.transpose(Image.FLIP_LEFT_RIGHT).rotate(20),
                       mask.transpose(Image.FLIP_LEFT_RIGHT).rotate(20)),
]

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
    def __init__(self, img_paths, mask_paths, soft_dir, img_size=512):
        pairs = []
        for ip, mp in zip(img_paths, mask_paths):
            stem = os.path.splitext(os.path.basename(ip))[0]
            soft = os.path.join(soft_dir, f"{stem}_logits.npy")
            if os.path.exists(soft):
                for aug_fn in AUG_TRANSFORMS:
                    pairs.append((ip, mp, soft, aug_fn, stem))
        self.pairs    = pairs
        self.img_norm = IMG_NORM_1024 if img_size == 1024 else IMG_NORM_512
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp, soft_path, aug_fn, stem = self.pairs[idx]
        img  = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")
        img, mask = aug_fn(img, mask)

        mask_np  = np.array(mask)
        cx, cy   = centroid_click(mask_np, jitter=5)
        # normalise click to [0, img_size]
        h, w     = mask_np.shape
        cx_n     = cx / w * self.img_size
        cy_n     = cy / h * self.img_size

        img_t    = self.img_norm(img)
        mask_256 = np.array(mask.resize((256, 256), Image.NEAREST))
        mask_t   = torch.from_numpy((mask_256 > 0).astype(np.float32)).unsqueeze(0)

        soft     = torch.from_numpy(
            np.load(soft_path).astype(np.float32))  # [1, 256, 256]

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
    image_embed = model.image_encoder(imgs)  # [B, 256, 64, 64]
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
    EPOCHS     = 30
    BATCH      = 16
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

    # download soft labels
    if not os.path.exists(SOFT_DIR) or len(glob.glob(SOFT_DIR + "/*.npy")) < 100:
        os.makedirs(SOFT_DIR, exist_ok=True)
        subprocess.run(["gsutil", "-m", "cp", "-r",
                        f"{BUCKET}/soft_labels/train/*", SOFT_DIR + "/"],
                       check=True)

    img_size   = MOBILE_SIZE if args.student == "mobilesam" else MODEL_SIZE
    img_paths  = sorted(glob.glob(DATA_DIR + "/images/*.png"))
    mask_paths = sorted(glob.glob(DATA_DIR + "/masks/*.png"))

    ds  = DistillDataset(img_paths, mask_paths, SOFT_DIR, img_size=img_size)
    dl  = DataLoader(ds, batch_size=BATCH, shuffle=True,
                     num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataset: {len(ds)} samples, {len(dl)} batches/epoch")

    # build model
    if args.student == "mobilesam":
        model = build_mobilesam(teacher_ckpt)
    else:
        model = build_repvitsam(teacher_ckpt)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train   = sum(p.numel() for p in trainable) / 1e6
    n_total   = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Trainable: {n_train:.1f}M / {n_total:.1f}M")

    optimizer = optim.AdamW(trainable, lr=BASE_LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler("cuda")

    use_kd    = args.ablation >= 2
    use_cldice = args.ablation >= 3

    suffix    = f"{args.student}_abl{args.ablation}"
    ckpt_path = f"/home/jupyter/{suffix}.pt"
    best_loss = float("inf")

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

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if args.student == "mobilesam":
                    logits = forward_mobilesam(model, imgs_d, pts_list)
                else:
                    logits = model(imgs_d, pts_list)

                loss, _, _ = distill_loss(
                    logits, soft_d, masks_d, pos_weight, cldice_w, use_kd)

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

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt_path)
            upload_async(ckpt_path,
                         f"{BUCKET}/checkpoints/{suffix}.pt")

    _gcs_q.join()
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {BUCKET}/checkpoints/{suffix}.pt")


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(args):
    from stage1_softlabels import centroid_click as cc
    img_paths  = sorted(glob.glob("/home/jupyter/arcade_val/images/*.png"))
    mask_paths = sorted(glob.glob("/home/jupyter/arcade_val/masks/*.png"))

    teacher_ckpt = "/home/jupyter/medsam2_arcade_v2.pt"
    suffix    = f"{args.student}_abl{args.ablation}"
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
            cx, cy = cc(mask)

            img_t  = img_norm(img).unsqueeze(0).to(DEVICE)
            pts    = [(torch.tensor(cx / mask.shape[1] * img_size),
                       torch.tensor(cy / mask.shape[0] * img_size))]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
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
    print(f"\n=== {suffix} — ARCADE val (200 images) ===")
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
