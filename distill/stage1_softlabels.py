"""
Stage 1: Generate soft labels from fine-tuned MedSAM2 teacher.

Runs on all 1,000 ARCADE train images. Saves per image:
  {stem}_logits.npy  — raw logits [1, 256, 256] float16
Output goes to gs://coronary-angio-v2/soft_labels/train/
"""

import os, sys, glob, subprocess, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

BUCKET      = "gs://coronary-angio-v2"
CKPT_PATH   = "/home/jupyter/medsam2_arcade_v2.pt"
MEDSAM2_DIR = "/home/jupyter/MedSAM2"
CONFIG      = "configs/sam2.1_hiera_t512.yaml"
MODEL_SIZE  = 512
HIRES       = MODEL_SIZE // 4
BB_FEAT_SIZES = [[HIRES // (2**k)] * 2 for k in range(3)]

DATA_DIR    = "/home/jupyter/arcade_train"
OUT_DIR     = "/home/jupyter/soft_labels"
BATCH       = 16
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

IMG_NORM = transforms.Compose([
    transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def download_data():
    os.makedirs(DATA_DIR + "/images", exist_ok=True)
    os.makedirs(DATA_DIR + "/masks", exist_ok=True)
    subprocess.run(
        ["gsutil", "-m", "cp", "-r",
         f"{BUCKET}/datasets/arcade/train/images/*", DATA_DIR + "/images/"],
        check=True)
    subprocess.run(
        ["gsutil", "-m", "cp", "-r",
         f"{BUCKET}/datasets/arcade/train/masks/*", DATA_DIR + "/masks/"],
        check=True)


def download_checkpoint():
    if not os.path.exists(CKPT_PATH):
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="Elakiya17/CA-SAM2", filename="medsam2_arcade_v2.pt",
                        local_dir=os.path.dirname(CKPT_PATH))
    # build_sam2 expects {"model": state_dict} — wrap if raw
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if not (isinstance(sd, dict) and "model" in sd):
        torch.save({"model": sd}, CKPT_PATH)
        print("Checkpoint wrapped for build_sam2")


def centroid_click(mask_np):
    ys, xs = np.where(mask_np > 0)
    if len(ys) == 0:
        return (mask_np.shape[1] // 2, mask_np.shape[0] // 2)
    return (int(xs.mean()), int(ys.mean()))


def load_teacher():
    sys.path.insert(0, MEDSAM2_DIR)
    from sam2.build_sam import build_sam2
    model = build_sam2("configs/sam2.1_hiera_t512.yaml", CKPT_PATH, device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_teacher_batch(model, imgs_t, points_list):
    """imgs_t: [B,3,512,512], points_list: list of (cx,cy) ints"""
    B = imgs_t.shape[0]
    imgs_dev = imgs_t.to(DEVICE)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        backbone_out = model.forward_image(imgs_dev)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(B, -1, *fs)
            for feat, fs in zip(vision_feats[::-1], BB_FEAT_SIZES[::-1])
        ][::-1]
        image_embed = feats[-1]
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


def main():
    print("=== Stage 1: Soft Label Generation ===")
    download_data()
    download_checkpoint()
    os.makedirs(OUT_DIR, exist_ok=True)

    img_paths  = sorted(glob.glob(DATA_DIR + "/images/*.png"))
    mask_paths = sorted(glob.glob(DATA_DIR + "/masks/*.png"))
    assert len(img_paths) == len(mask_paths) == 1000, \
        f"Expected 1000 pairs, got {len(img_paths)} imgs / {len(mask_paths)} masks"

    model = load_teacher()
    print(f"Teacher loaded on {DEVICE}")

    done = 0
    for start in range(0, len(img_paths), BATCH):
        batch_imgs   = img_paths[start:start + BATCH]
        batch_masks  = mask_paths[start:start + BATCH]
        imgs_t, pts, stems = [], [], []

        for ip, mp in zip(batch_imgs, batch_masks):
            stem = os.path.splitext(os.path.basename(ip))[0]
            out  = os.path.join(OUT_DIR, f"{stem}_logits.npy")
            if os.path.exists(out):
                done += 1
                continue

            img  = Image.open(ip).convert("RGB")
            mask = np.array(Image.open(mp).convert("L"))
            cx, cy = centroid_click(mask)

            imgs_t.append(IMG_NORM(img))
            pts.append((cx, cy))
            stems.append(stem)

        if not stems:
            continue

        logits_list = run_teacher_batch(model, torch.stack(imgs_t), pts)

        for stem, logits in zip(stems, logits_list):
            arr = logits[0].numpy().astype(np.float16)  # [1, 256, 256]
            np.save(os.path.join(OUT_DIR, f"{stem}_logits.npy"), arr)
            done += 1

        print(f"  {done}/{len(img_paths)} done", flush=True)

    print(f"Uploading soft labels to {BUCKET}/soft_labels/train/ ...")
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", OUT_DIR + "/*",
         f"{BUCKET}/soft_labels/train/"],
        check=True)
    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
