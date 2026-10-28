"""
Video Fluoroscopy Benchmark: SAM2.1 1024 Teacher vs MobileSAM 1024 Student.
Runs inference across real coronary fluoroscopy video frames (Study 5, CoronaryDominance),
measures per-frame latency/FPS, generates a 3-panel comparison video (.mp4),
and exports screenshots at key temporal phases (wash-in, peak fill, wash-out).
"""

import os
import glob
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAMES_DIR = os.environ.get("FRAMES_DIR", "/workspace/data/vid5_frames")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/results/vid5_eval")
TEACHER_CKPT = os.environ.get("TEACHER_CKPT", "/workspace/checkpoints/sam2_fluoroscopy_1024.pt")
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
    print("✓ Teacher (SAM2.1 Hiera-Tiny 1024) loaded.")
    return model

def load_student():
    from mobile_sam import sam_model_registry
    model = sam_model_registry["vit_t"](checkpoint=STUDENT_BASE_CKPT)
    sd = torch.load(STUDENT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    print("✓ Student (MobileSAM 1024) loaded.")
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

def overlay_mask(img_np, mask_np, color=(0, 165, 255), alpha=0.45):
    """color in BGR format."""
    vis = img_np.copy()
    idx = mask_np > 0
    c_arr = np.array(color, dtype=np.float32)
    vis[idx] = (vis[idx].astype(np.float32) * (1 - alpha) + c_arr * alpha).clip(0, 255).astype(np.uint8)
    return vis

def auto_detect_prompt(frames):
    """
    Finds the peak contrast frame by pixel variance and selects the centroid
    of the darkest contrast region (vessel bolus).
    """
    vars_list = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        vars_list.append(np.var(gray))
    peak_idx = int(np.argmax(vars_list))
    peak_frame = cv2.cvtColor(frames[peak_idx], cv2.COLOR_BGR2GRAY)

    # Inverted intensity: contrast dye is dark, so inverted is bright
    inv = 255 - peak_frame
    # Gaussian center weighting
    H, W = peak_frame.shape
    y_grid, x_grid = np.ogrid[:H, :W]
    center_y, center_x = H / 2, W / 2
    sigma = min(H, W) / 3
    gaussian_weight = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))
    weighted_inv = inv.astype(np.float32) * gaussian_weight

    # Top 15% brightest pixels (strongest dye fill)
    thresh = np.percentile(weighted_inv, 88)
    ys, xs = np.where(weighted_inv > thresh)
    if len(xs) == 0:
        cx, cy = W // 2, H // 2
    else:
        cx, cy = int(xs.mean()), int(ys.mean())

    return peak_idx, cx, cy

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sc_dir = os.path.join(OUT_DIR, "screenshots")
    os.makedirs(sc_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.png")))
    if not frame_files:
        raise RuntimeError(f"No frames found in {FRAMES_DIR}")

    frames = [cv2.imread(f) for f in frame_files]
    H, W, _ = frames[0].shape
    print(f"Loaded {len(frames)} frames ({W}x{H}) from {FRAMES_DIR}")

    # Detect prompt
    peak_idx, prompt_x, prompt_y = auto_detect_prompt(frames)
    print(f"Auto-selected Prompt Frame: #{peak_idx:02d} | Prompt Point: ({prompt_x}, {prompt_y})")

    # Load Models
    teacher = load_teacher()
    student = load_student()

    t_times, s_times = [], []
    t_masks, s_masks = [], []

    cx_1024 = prompt_x / W * 1024.0
    cy_1024 = prompt_y / H * 1024.0

    print(f"\nProcessing {len(frames)} video frames...")
    for idx, frame_bgr in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_t = IMG_NORM_1024(img_pil).unsqueeze(0).to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            # 1. Teacher
            t_start = time.time()
            t_lm = forward_sam2(teacher, img_t, cx_1024, cy_1024)
            t_up = F.interpolate(t_lm, size=(H, W), mode="bilinear", align_corners=False)
            t_mask = (t_up[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
            torch.cuda.synchronize()
            t_times.append((time.time() - t_start) * 1000)

            # 2. Student
            s_start = time.time()
            s_lm = forward_mobilesam(student, img_t, cx_1024, cy_1024)
            s_up = F.interpolate(s_lm, size=(H, W), mode="bilinear", align_corners=False)
            s_mask = (s_up[0, 0].cpu().numpy() > 0.0).astype(np.uint8)
            torch.cuda.synchronize()
            s_times.append((time.time() - s_start) * 1000)

        t_masks.append(t_mask)
        s_masks.append(s_mask)

    t_mean_lat = np.mean(t_times)
    s_mean_lat = np.mean(s_times)
    t_fps = 1000.0 / t_mean_lat
    s_fps = 1000.0 / s_mean_lat

    print("\n" + "=" * 65)
    print(" VIDEO INFERENCE BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Teacher Latency: {t_mean_lat:.1f} ms/frame  |  FPS: {t_fps:.1f}")
    print(f"Student Latency: {s_mean_lat:.1f} ms/frame  |  FPS: {s_fps:.1f} (Speedup: {t_mean_lat/s_mean_lat:.2f}x)")
    print("=" * 65)

    # Generate 3-Panel Side-by-Side Video
    # [Original + Click] | [Teacher SAM2.1 (Orange)] | [Student MobileSAM (Red)]
    video_out_path = os.path.join(OUT_DIR, "vid5_teacher_vs_student.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(video_out_path, fourcc, 12.0, (W * 3, H))

    # Key screenshot frames to export
    key_frames = [5, 18, peak_idx, min(48, len(frames)-1), min(60, len(frames)-1)]

    for idx, (frame_bgr, t_m, s_m) in enumerate(zip(frames, t_masks, s_masks)):
        vis_orig = frame_bgr.copy()
        # Draw green prompt click dot on original
        cv2.circle(vis_orig, (prompt_x, prompt_y), 5, (0, 255, 0), -1)

        # Overlays: Teacher = Orange (0, 140, 255), Student = Red (0, 0, 255)
        vis_t = overlay_mask(frame_bgr, t_m, color=(0, 140, 255))
        vis_s = overlay_mask(frame_bgr, s_m, color=(0, 0, 255))

        # Add text labels
        cv2.putText(vis_orig, f"Original (Frame {idx:02d})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_t, f"Teacher SAM2.1 1024 ({t_mean_lat:.0f}ms)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        cv2.putText(vis_s, f"Student MobileSAM 1024 ({s_mean_lat:.0f}ms)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        composite = np.hstack([vis_orig, vis_t, vis_s])
        out_video.write(composite)

        if idx in key_frames:
            sc_path = os.path.join(sc_dir, f"phase_frame_{idx:02d}.png")
            cv2.imwrite(sc_path, composite)
            print(f"Saved keyframe screenshot: {sc_path}")

    out_video.release()
    print(f"\n✓ Comparison video written to: {video_out_path}")

if __name__ == "__main__":
    main()
