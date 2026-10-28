"""
DCA1 External Domain Shift Evaluation.
Evaluates SAM2.1 1024 Teacher vs MobileSAM 1024 Student on 134 unseen clinical coronary angiograms.

Neither model saw any DCA1 images during training (trained strictly on ARCADE).
DCA1: 134 images (300x300 px), from Cervantes-Sanchez et al.
"""

import os
import glob
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DCA1_DIR = os.environ.get("DCA1_DIR", "/workspace/data/dca1")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/results/dca1")
TEACHER_CKPT = os.environ.get("TEACHER_CKPT", "/workspace/checkpoints/sam2_fluoroscopy_1024.pt")
TEACHER_BASE_CKPT = os.environ.get("TEACHER_BASE_CKPT", "/workspace/checkpoints/sam2.1_hiera_tiny.pt")
STUDENT_CKPT = os.environ.get("STUDENT_CKPT", "/workspace/checkpoints/mobilesam_abl1_v6.pt")
STUDENT_BASE_CKPT = os.environ.get("STUDENT_BASE_CKPT", "/workspace/checkpoints/mobile_sam.pt")
SAM2_REPO = os.environ.get("SAM2_REPO", "/workspace/sam2_meta")

IMG_NORM_1024 = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_teacher():
    import sys
    for p in [SAM2_REPO, "/opt/sam2_meta"]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    from sam2.build_sam import build_sam2

    sd = torch.load(TEACHER_CKPT, map_location="cpu", weights_only=False)
    if not (isinstance(sd, dict) and "model" in sd):
        torch.save({"model": sd}, TEACHER_CKPT)

    _orig = os.getcwd()
    if os.path.isdir(SAM2_REPO):
        os.chdir(SAM2_REPO)
    try:
        model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", TEACHER_CKPT,
                           device=DEVICE, apply_postprocessing=False)
    finally:
        os.chdir(_orig)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("✓ Teacher (SAM2.1 Hiera-Tiny 1024) loaded successfully.")
    return model

def load_student():
    from mobile_sam import sam_model_registry
    model = sam_model_registry["vit_t"](checkpoint=STUDENT_BASE_CKPT)
    sd = torch.load(STUDENT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    print("✓ Student (MobileSAM 1024) loaded successfully.")
    return model

def forward_sam2(model, img_t, cx, cy):
    with torch.no_grad():
        backbone_out = model.image_encoder(img_t)
        if getattr(model, "use_high_res_features_in_sam", False):
            backbone_out["backbone_fpn"][0] = model.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = model.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
        if getattr(model, "directly_add_no_mem_embed", False):
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

        feats = [feat.permute(1, 2, 0).view(1, -1, *fs)
                 for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
        image_embed = feats[-1]
        high_res_feats = feats[:-1]

        point = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=DEVICE)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=DEVICE)
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(point, pt_label), boxes=None, masks=None)
        lm, _, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False, repeat_image=False,
            high_res_features=[f for f in high_res_feats],
        )
    return lm

def forward_mobilesam(model, img_t, cx, cy):
    with torch.no_grad():
        image_embed = model.image_encoder(img_t)
        if getattr(model, "neck_adapter", None) is not None:
            image_embed = model.neck_adapter(image_embed)
        point = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=DEVICE)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=DEVICE)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=(point, pt_label), boxes=None, masks=None)
        lm, _ = model.mask_decoder(
            image_embeddings=image_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
    return lm

def compute_metrics(pred, gt):
    inter = (pred * gt).sum()
    p_sum = pred.sum()
    g_sum = gt.sum()
    union = p_sum + g_sum - inter
    dice = (2.0 * inter / (p_sum + g_sum)) if (p_sum + g_sum) > 0 else 1.0
    iou = (inter / union) if union > 0 else 1.0
    prec = (inter / p_sum) if p_sum > 0 else 1.0
    rec = (inter / g_sum) if g_sum > 0 else 1.0
    return float(dice), float(iou), float(prec), float(rec)

def overlay_mask(img_np, mask_np, color=(0, 255, 0), alpha=0.5):
    """Overlay binary mask on RGB image."""
    vis = img_np.copy()
    c_arr = np.array(color, dtype=np.float32)
    idx = mask_np > 0
    vis[idx] = (vis[idx].astype(np.float32) * (1 - alpha) + c_arr * alpha).clip(0, 255).astype(np.uint8)
    return vis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_overlays", action="store_true", default=True)
    parser.add_argument("--num_overlays", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    overlay_dir = os.path.join(OUT_DIR, "overlays")
    if args.save_overlays:
        os.makedirs(overlay_dir, exist_ok=True)

    print("=" * 65)
    print(" DCA1 External Domain Shift Evaluation (Unseen Clinical Dataset)")
    print("=" * 65)

    # 1. Pair images and ground truth masks
    # Match numbers: e.g. '1.pgm' with '1_gt.pgm'
    all_files = glob.glob(os.path.join(DCA1_DIR, "*.pgm"))
    pairs = []
    for i in range(1, 135):
        img_p = os.path.join(DCA1_DIR, f"{i}.pgm")
        msk_p = os.path.join(DCA1_DIR, f"{i}_gt.pgm")
        if os.path.exists(img_p) and os.path.exists(msk_p):
            pairs.append((img_p, msk_p, i))

    print(f"Loaded {len(pairs)} validated image/mask pairs from DCA1.\n")
    if len(pairs) == 0:
        raise RuntimeError(f"No valid DCA1 pairs found in {DCA1_DIR}")

    # 2. Load Models
    teacher = load_teacher()
    student = load_student()

    t_dices, t_ious, t_precs, t_recs = [], [], [], []
    s_dices, s_ious, s_precs, s_recs = [], [], [], []

    print(f"\nRunning inference across all {len(pairs)} images on {DEVICE}...")
    t0 = time.time()

    for idx, (ip, mp, num) in enumerate(pairs):
        # Load grayscale PGM and convert to RGB
        img_pil = Image.open(ip).convert("RGB")
        msk_pil = Image.open(mp).convert("L")
        img_np = np.array(img_pil)
        msk_np = np.array(msk_pil)
        gt = (msk_np > 0).astype(np.uint8)

        # Centroid click prompt in original image coordinates
        ys, xs = np.where(gt > 0)
        if len(xs) == 0:
            cx, cy = msk_np.shape[1] // 2, msk_np.shape[0] // 2
        else:
            cx, cy = int(xs.mean()), int(ys.mean())

        # Scale prompt to 1024
        cx_1024 = cx / msk_np.shape[1] * 1024.0
        cy_1024 = cy / msk_np.shape[0] * 1024.0

        img_t = IMG_NORM_1024(img_pil).unsqueeze(0).to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            # Teacher inference
            t_lm = forward_sam2(teacher, img_t, cx_1024, cy_1024)
            t_up = F.interpolate(t_lm, size=(msk_np.shape[0], msk_np.shape[1]), mode="bilinear", align_corners=False)
            t_pred = (t_up[0, 0].cpu().numpy() > 0.0).astype(np.uint8)

            # Student inference
            s_lm = forward_mobilesam(student, img_t, cx_1024, cy_1024)
            s_up = F.interpolate(s_lm, size=(msk_np.shape[0], msk_np.shape[1]), mode="bilinear", align_corners=False)
            s_pred = (s_up[0, 0].cpu().numpy() > 0.0).astype(np.uint8)

        t_dice, t_iou, t_prec, t_rec = compute_metrics(t_pred, gt)
        s_dice, s_iou, s_prec, s_rec = compute_metrics(s_pred, gt)

        t_dices.append(t_dice)
        t_ious.append(t_iou)
        t_precs.append(t_prec)
        t_recs.append(t_rec)

        s_dices.append(s_dice)
        s_ious.append(s_iou)
        s_precs.append(s_prec)
        s_recs.append(s_rec)

        # Save composite overlay for first N images
        if args.save_overlays and idx < args.num_overlays:
            # 4-panel: [Original + Click] | [Teacher] | [Student] | [Ground Truth]
            vis_orig = img_np.copy()
            # Draw prompt click dot
            import cv2
            cv2.circle(vis_orig, (cx, cy), 5, (255, 0, 0), -1)

            vis_t = overlay_mask(img_np, t_pred, color=(0, 140, 255))   # Orange
            vis_s = overlay_mask(img_np, s_pred, color=(255, 50, 50))   # Red
            vis_gt = overlay_mask(img_np, gt, color=(50, 220, 50))      # Green

            W, H = img_pil.size
            panel = Image.new("RGB", (W * 4, H))
            panel.paste(Image.fromarray(vis_orig), (0, 0))
            panel.paste(Image.fromarray(vis_t), (W, 0))
            panel.paste(Image.fromarray(vis_s), (W * 2, 0))
            panel.paste(Image.fromarray(vis_gt), (W * 3, 0))
            panel.save(os.path.join(overlay_dir, f"dca1_comp_{num:03d}.png"))

        if (idx + 1) % 25 == 0 or idx + 1 == len(pairs):
            print(f"  [{idx+1:3d}/{len(pairs)}] Running Dice -> Teacher: {np.mean(t_dices):.4f} | Student: {np.mean(s_dices):.4f}")

    total_time = time.time() - t0
    t_dice_arr, s_dice_arr = np.array(t_dices), np.array(s_dices)
    t_iou_arr, s_iou_arr = np.array(t_ious), np.array(s_ious)

    print("\n" + "=" * 65)
    print(" EXTERNAL EVALUATION RESULTS (134 Unseen DCA1 Angiograms)")
    print("=" * 65)
    print(f"{'Metric':<20} | {'Teacher (SAM2.1 1024)':<20} | {'Student (MobileSAM 1024)':<22}")
    print("-" * 65)
    print(f"{'Dice Mean ± Std':<20} | {t_dice_arr.mean():.4f} ± {t_dice_arr.std():.4f}{'':<4} | {s_dice_arr.mean():.4f} ± {s_dice_arr.std():.4f}")
    print(f"{'Dice Median':<20} | {np.median(t_dice_arr):.4f}{'':<14} | {np.median(s_dice_arr):.4f}")
    print(f"{'Dice Min / Max':<20} | {t_dice_arr.min():.4f} / {t_dice_arr.max():.4f}{'':<5} | {s_dice_arr.min():.4f} / {s_dice_arr.max():.4f}")
    print(f"{'IoU Mean ± Std':<20} | {t_iou_arr.mean():.4f} ± {t_iou_arr.std():.4f}{'':<4} | {s_iou_arr.mean():.4f} ± {s_iou_arr.std():.4f}")
    print(f"{'Precision':<20} | {np.mean(t_precs):.4f}{'':<14} | {np.mean(s_precs):.4f}")
    print(f"{'Recall':<20} | {np.mean(t_recs):.4f}{'':<14} | {np.mean(s_recs):.4f}")
    print("-" * 65)
    diff = s_dice_arr.mean() - t_dice_arr.mean()
    print(f"Student vs Teacher Margin: {diff:+.4f} Dice (p-value paired t-test: {from_scipy_ttest(s_dice_arr, t_dice_arr):.4f})")
    print(f"Total inference time for 134 images: {total_time:.1f}s ({total_time/len(pairs)*1000:.1f} ms/image)")
    print("=" * 65)

    results = {
        "dataset": "DCA1 (134 coronary angiograms, 300x300, external domain shift)",
        "num_images": len(pairs),
        "teacher": {
            "model": "SAM2.1 Hiera-Tiny 1024 (ARCADE fine-tuned)",
            "params": "38.9M",
            "dice_mean": float(t_dice_arr.mean()),
            "dice_std": float(t_dice_arr.std()),
            "dice_median": float(np.median(t_dice_arr)),
            "iou_mean": float(t_iou_arr.mean()),
            "iou_std": float(t_iou_arr.std()),
            "precision": float(np.mean(t_precs)),
            "recall": float(np.mean(t_recs))
        },
        "student": {
            "model": "MobileSAM 1024 (ARCADE distilled/fine-tuned)",
            "params": "10.1M",
            "dice_mean": float(s_dice_arr.mean()),
            "dice_std": float(s_dice_arr.std()),
            "dice_median": float(np.median(s_dice_arr)),
            "iou_mean": float(s_iou_arr.mean()),
            "iou_std": float(s_iou_arr.std()),
            "precision": float(np.mean(s_precs)),
            "recall": float(np.mean(s_recs))
        },
        "margin": float(diff),
        "per_image_results": [
            {"id": num, "teacher_dice": t_dices[i], "student_dice": s_dices[i]}
            for i, (_, _, num) in enumerate(pairs)
        ]
    }

    out_json = os.path.join(OUT_DIR, "dca1_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out_json}")
    if args.save_overlays:
        print(f"Saved visual overlays to: {overlay_dir}")

def from_scipy_ttest(a, b):
    try:
        from scipy import stats
        return float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        return 1.0

if __name__ == "__main__":
    main()
