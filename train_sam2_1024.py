"""
Fine-tune SAM2.1 Hiera-Tiny at 1024x1024 on ARCADE coronary angiography.
Starts from SAM2.1 base pretrained weights (not the 512-tuned CA-SAM2 checkpoint).
Same training protocol as CA-SAM2 but at double resolution.
Goal: isolate whether resolution explains MobileSAM v4's advantage over the teacher.
"""
import os, sys, glob, time, random, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/opt/MedSAM2")

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = 1024
DATA_DIR  = "/home/jupyter/arcade_train"
TEST_DIR  = "/home/jupyter/arcade_test"
BUCKET    = "gs://coronary-angio-v2"
HF_REPO   = os.environ.get("HF_REPO", "Elakiya17/fluoroscopy-sam2")

IMG_NORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

NUM_GEOM_AUGS = 5

def _geom_pil(img, mask, seed):
    import torchvision.transforms.functional as TF
    rng = random.Random(seed)
    if rng.random() < 0.5:
        img  = TF.hflip(img)
        mask = TF.hflip(mask)
    angle = rng.uniform(-15, 15)
    img   = TF.rotate(img,  angle)
    mask  = TF.rotate(mask, angle)
    return img, mask

def random_color_aug(img):
    import torchvision.transforms.functional as TF
    img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
    noise = torch.randn_like(transforms.ToTensor()(img)) * 0.02
    t = transforms.ToTensor()(img)
    t = (t + noise).clamp(0, 1)
    return transforms.ToPILImage()(t)

def centroid_click(mask_np, jitter=10):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return mask_np.shape[1]//2, mask_np.shape[0]//2
    cx = int(xs.mean()) + random.randint(-jitter, jitter)
    cy = int(ys.mean()) + random.randint(-jitter, jitter)
    return np.clip(cx, 0, mask_np.shape[1]-1), np.clip(cy, 0, mask_np.shape[0]-1)


class ArcadeDataset(Dataset):
    def __init__(self, img_paths, mask_paths, color_aug=True):
        self.items = []
        for ip, mp in zip(img_paths, mask_paths):
            for aug_idx in range(NUM_GEOM_AUGS):
                self.items.append((ip, mp, aug_idx))
        self.color_aug = color_aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ip, mp, aug_idx = self.items[idx]
        img  = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")
        img, mask = _geom_pil(img, mask, seed=idx * 137 + aug_idx)
        if self.color_aug:
            img = random_color_aug(img)
        mask_np = np.array(mask)
        cx, cy  = centroid_click(mask_np, jitter=10)
        img_t   = IMG_NORM(img)
        mask_t  = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)
        pt      = torch.tensor([cx / mask_np.shape[1] * IMG_SIZE,
                                 cy / mask_np.shape[0] * IMG_SIZE], dtype=torch.float32)
        return img_t, mask_t, pt


def dice_loss(pred, target, smooth=1e-6):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    return 1 - (2 * inter + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)


def build_sam2_1024(base_ckpt):
    from sam2.build_sam import build_sam2
    # SAM2.1 Hiera-Tiny default config is 1024x1024
    model = build_sam2("sam2.1_hiera_t.yaml", base_ckpt,
                       device=DEVICE, apply_postprocessing=False)
    return model


def forward_sam2(model, imgs, pts):
    """Run SAM2 image encoder + mask decoder with centroid point prompts."""
    B = imgs.shape[0]
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
        backbone_out = model.forward_image(imgs)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

        # build image embeddings per-sample
        feat_sizes = [(IMG_SIZE // 32, IMG_SIZE // 32)]  # Hiera-Tiny stride-32 final
        image_embed = vision_feats[-1].permute(1, 2, 0).view(
            B, -1, IMG_SIZE // 32, IMG_SIZE // 32)

    logits_list = []
    for i, pt in enumerate(pts):
        cx, cy = pt[0].item(), pt[1].item()
        point  = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=DEVICE)
        label  = torch.ones(1, 1, dtype=torch.int, device=DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            sparse_emb, dense_emb = model.sam_prompt_encoder(
                points=(point, label), boxes=None, masks=None)
            lm, _ = model.sam_mask_decoder(
                image_embeddings=image_embed[i].unsqueeze(0),
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=None,
            )
        logits_list.append(lm)
    return torch.cat(logits_list, dim=0)  # [B, 1, 256, 256]


def quick_val(model, img_paths_val, mask_paths_val):
    model.eval()
    scores = []
    with torch.no_grad():
        for ip, mp in zip(img_paths_val, mask_paths_val):
            img  = Image.open(ip).convert("RGB")
            mask = np.array(Image.open(mp).convert("L"))
            cx, cy = centroid_click(mask, jitter=0)
            img_t  = IMG_NORM(img).unsqueeze(0).to(DEVICE)
            pt     = [torch.tensor([cx/mask.shape[1]*IMG_SIZE,
                                    cy/mask.shape[0]*IMG_SIZE])]
            logits = forward_sam2(model, img_t, pt)
            pred   = (logits[0,0].float().cpu().numpy() > 0.0).astype(np.uint8)
            pred   = np.array(Image.fromarray(pred*255).resize(
                (mask.shape[1], mask.shape[0]), Image.NEAREST)) // 255
            gt     = (mask > 0).astype(np.uint8)
            denom  = pred.sum() + gt.sum()
            scores.append((2*(pred*gt).sum()/denom) if denom > 0 else 1.0)
    model.train()
    return float(np.mean(scores))


def train():
    EPOCHS   = int(os.environ.get("MAX_EPOCHS", 50))
    PATIENCE = int(os.environ.get("ES_PATIENCE", 10))
    BATCH    = int(os.environ.get("BATCH_SIZE", 2))
    BASE_LR  = 1e-4
    WD       = 0.01
    VERSION  = os.environ.get("MODEL_VERSION", "sam2_1024")
    FG_FRAC  = 0.05

    base_ckpt = "/home/jupyter/sam2.1_hiera_tiny.pt"
    if not os.path.exists(base_ckpt):
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="facebook/sam2.1-hiera-tiny",
                        filename="sam2.1_hiera_tiny.pt",
                        local_dir="/home/jupyter")

    print(f"Building SAM2.1 Hiera-Tiny at {IMG_SIZE}x{IMG_SIZE}...")
    model = build_sam2_1024(base_ckpt)
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total params: {n_total:.1f}M  |  Device: {DEVICE}")

    # freeze encoder, train decoder + prompt encoder (same protocol as CA-SAM2)
    for p in model.image_encoder.parameters():
        p.requires_grad_(False)
    trainable = ([p for p in model.sam_mask_decoder.parameters()] +
                 [p for p in model.sam_prompt_encoder.parameters()])
    n_train = sum(p.numel() for p in trainable) / 1e6
    print(f"  Trainable (decoder+prompt enc): {n_train:.1f}M")

    img_paths_all  = sorted(glob.glob(DATA_DIR + "/images/*.png"))
    mask_paths_all = sorted(glob.glob(DATA_DIR + "/masks/*.png"))
    indices = list(range(len(img_paths_all)))
    random.Random(42).shuffle(indices)
    val_idx  = indices[:100]
    trn_idx  = indices[100:]
    img_trn  = [img_paths_all[i] for i in trn_idx]
    msk_trn  = [mask_paths_all[i] for i in trn_idx]
    img_val  = [img_paths_all[i] for i in val_idx]
    msk_val  = [mask_paths_all[i] for i in val_idx]
    print(f"Split: {len(img_trn)} train / {len(img_val)} val / 200 held-out test")

    ds = ArcadeDataset(img_trn, msk_trn, color_aug=True)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)

    pos_weight = torch.tensor([(1-FG_FRAC)/FG_FRAC]).to(DEVICE)
    optimizer  = optim.AdamW(trainable, lr=BASE_LR, weight_decay=WD)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler     = torch.amp.GradScaler("cuda", enabled=(DEVICE=="cuda"))

    ckpt_path  = f"/home/jupyter/{VERSION}.pt"
    best_dice  = -1.0
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        for p in model.image_encoder.parameters():
            p.requires_grad_(False)
        epoch_loss = 0.0
        t0 = time.time()

        for imgs, masks, pts in dl:
            imgs_d  = imgs.to(DEVICE)
            masks_d = masks.to(DEVICE)
            pts_list = [pts[i] for i in range(len(pts))]

            optimizer.zero_grad(set_to_none=True)
            logits = forward_sam2(model, imgs_d, pts_list)
            logits_up = F.interpolate(logits.float(), size=masks_d.shape[-2:],
                                      mode="bilinear", align_corners=False)
            loss_bce  = F.binary_cross_entropy_with_logits(
                logits_up, masks_d, pos_weight=pos_weight)
            loss_dice = dice_loss(logits_up, masks_d).mean()
            loss = 0.5 * loss_dice + 0.2 * loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        val_dice = quick_val(model, img_val, msk_val)
        improved = val_dice > best_dice
        if improved:
            best_dice  = val_dice
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss/len(dl):.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  t={int(time.time()-t0)}s")
        print(f"  → val Dice: {val_dice:.4f}  best: {best_dice:.4f}  patience: {no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Training complete. Best val Dice: {best_dice:.4f}")

    # ── Held-out test eval ────────────────────────────────────────────────────
    print(f"\n=== {VERSION} — HELD-OUT TEST SET (200 ARCADE val images) ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()
    test_img   = sorted(glob.glob(TEST_DIR + "/images/*.png"))
    test_mask  = sorted(glob.glob(TEST_DIR + "/masks/*.png"))
    dice_scores = []
    with torch.no_grad():
        for ip, mp in zip(test_img, test_mask):
            img  = Image.open(ip).convert("RGB")
            mask = np.array(Image.open(mp).convert("L"))
            cx, cy = centroid_click(mask, jitter=0)
            img_t  = IMG_NORM(img).unsqueeze(0).to(DEVICE)
            pt     = [torch.tensor([cx/mask.shape[1]*IMG_SIZE,
                                    cy/mask.shape[0]*IMG_SIZE])]
            logits = forward_sam2(model, img_t, pt)
            pred   = (logits[0,0].float().cpu().numpy() > 0.0).astype(np.uint8)
            pred   = np.array(Image.fromarray(pred*255).resize(
                (mask.shape[1], mask.shape[0]), Image.NEAREST)) // 255
            gt     = (mask > 0).astype(np.uint8)
            inter  = (pred*gt).sum()
            dice_scores.append(2*inter/(pred.sum()+gt.sum()+1e-6))

    arr = np.array(dice_scores)
    print(f"  Dice: {arr.mean():.3f} ± {arr.std():.3f}")
    print(f"  IoU:  {np.mean([(pred*gt).sum()/((pred+gt-pred*gt).sum()+1e-6) for pred,gt in [(np.array(Image.fromarray((torch.sigmoid(torch.load(ckpt_path,map_location='cpu',weights_only=False).get('sam_mask_decoder.pred_obj_scores.weight',torch.zeros(1))).numpy()>0.5).astype(np.uint8)*255).resize((512,512),Image.NEAREST))//255,(mask>0).astype(np.uint8))]):.3f}")

    # push checkpoint to GCS
    subprocess.run(["gsutil", "cp", ckpt_path,
                    f"{BUCKET}/checkpoints/{VERSION}.pt"], check=True)
    print(f"Checkpoint: {BUCKET}/checkpoints/{VERSION}.pt")

    # push to HF if repo exists
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(path_or_fileobj=ckpt_path,
                        path_in_repo=f"{VERSION}.pt",
                        repo_id=HF_REPO, repo_type="model")
        print(f"Pushed to HuggingFace: {HF_REPO}/{VERSION}.pt")
    except Exception as e:
        print(f"HF push skipped (repo may not exist yet): {e}")


if __name__ == "__main__":
    train()
