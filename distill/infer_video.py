"""
MobileSAM v4 video inference benchmark.
Processes a directory of frames (or a .mp4) and reports:
  - per-frame latency (ms)
  - peak RAM usage (MB)
  - frames per second
  - optional: saves overlay images to output_dir

Usage:
  python3 infer_video.py --ckpt /path/to/mobilesam_abl1_v4.pt \
                         --frames /path/to/frames_dir \
                         --output_dir /tmp/overlay \
                         --device cpu
"""
import argparse
import os
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import resource

MOBILE_CKPT = "/tmp/mobile_sam.pt"
MODEL_SIZE  = 1024


def load_model(ckpt_path, device):
    import subprocess
    if not os.path.exists(MOBILE_CKPT):
        subprocess.run([
            "wget", "-q", "-O", MOBILE_CKPT,
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        ], check=True)

    from mobile_sam import sam_model_registry
    model = sam_model_registry["vit_t"](checkpoint=MOBILE_CKPT)

    # load fine-tuned v4 weights
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(device)


def preprocess(img_path, size=MODEL_SIZE):
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, orig_w, orig_h


def run_inference(model, img_tensor, device):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        image_embed = model.image_encoder(img_tensor)
        if image_embed.shape[-1] != 64:
            image_embed = F.interpolate(image_embed, size=(64, 64),
                                        mode="bilinear", align_corners=False)
        # centroid prompt (centre of image as default)
        cx = torch.tensor([[[MODEL_SIZE / 2, MODEL_SIZE / 2]]],
                          dtype=torch.float32, device=device)
        pt_label = torch.ones(1, 1, dtype=torch.int, device=device)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=(cx, pt_label), boxes=None, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=image_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
    prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8)
    return mask


def peak_ram_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns kilobytes
    if os.uname().sysname == "Darwin":
        return usage / 1e6
    return usage / 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--frames",     required=True,
                        help="Directory of .png/.jpg frames, or a single .mp4")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--max_frames", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model on {device}...")
    t0 = time.time()
    model = load_model(args.ckpt, device)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    if os.path.isdir(args.frames):
        frame_paths = sorted(
            glob.glob(args.frames + "/*.png") +
            glob.glob(args.frames + "/*.jpg")
        )[:args.max_frames]
    else:
        raise ValueError("--frames must be a directory of images")

    if not frame_paths:
        print("No frames found.")
        return

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    latencies = []
    print(f"Running inference on {len(frame_paths)} frames...")
    for i, fp in enumerate(frame_paths):
        img_tensor, ow, oh = preprocess(fp)
        t_start = time.time()
        mask = run_inference(model, img_tensor, device)
        latency_ms = (time.time() - t_start) * 1000
        latencies.append(latency_ms)

        if args.output_dir:
            overlay = Image.open(fp).convert("RGB").resize((MODEL_SIZE, MODEL_SIZE))
            ov_arr  = np.array(overlay)
            ov_arr[mask == 1, 0] = 255  # red channel overlay
            Image.fromarray(ov_arr).save(
                os.path.join(args.output_dir, f"frame_{i:04d}.png"))

        if (i + 1) % 20 == 0:
            print(f"  frame {i+1}/{len(frame_paths)}: {latency_ms:.1f}ms")

    latencies = np.array(latencies)
    ram_mb = peak_ram_mb()
    fps = 1000.0 / np.mean(latencies)

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Device:           {device}")
    print(f"Frames processed: {len(latencies)}")
    print(f"Latency mean:     {np.mean(latencies):.1f} ms")
    print(f"Latency p50:      {np.percentile(latencies, 50):.1f} ms")
    print(f"Latency p95:      {np.percentile(latencies, 95):.1f} ms")
    print(f"FPS (mean):       {fps:.2f}")
    print(f"Peak RAM:         {ram_mb:.0f} MB")
    print(f"Checkpoint size:  {os.path.getsize(args.ckpt) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
