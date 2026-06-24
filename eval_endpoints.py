"""
Evaluate clinical endpoints on ARCADE validation set.

Downloads CA-SAM2 from HuggingFace, runs inference on ARCADE val images,
applies clinical endpoint post-processing, and saves results + overlays to GCS.

Usage (on GCP instance after setup_gcp.sh):
  python eval_endpoints.py
  python eval_endpoints.py --max_images 10   # quick smoke-test
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import center_of_mass

from endpoints import compute_all_endpoints, visualize_endpoints

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
GCS_BUCKET = "gs://coronary-angio-v2"
HF_REPO    = "Elakiya17/CA-SAM2"
HF_FILE    = "medsam2_arcade_v2.pt"


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def pull_arcade_val(data_dir: str):
    if (os.path.exists(os.path.join(data_dir, "images"))
            and len(glob.glob(data_dir + "/images/*.png")) > 0):
        print(f"ARCADE val already local at {data_dir}")
        return
    os.makedirs(data_dir, exist_ok=True)
    print("Pulling ARCADE val from GCS...")
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", f"{GCS_BUCKET}/datasets/arcade/val/", data_dir],
        check=True,
    )


def _find_medsam2() -> str:
    """Locate MedSAM2 root: env var > /opt > ~/MedSAM2."""
    for candidate in [
        os.environ.get("MEDSAM2_PATH", ""),
        "/opt/MedSAM2",
        os.path.expanduser("~/MedSAM2"),
    ]:
        if candidate and os.path.isdir(candidate):
            return candidate
    raise RuntimeError("MedSAM2 not found. Clone it or set MEDSAM2_PATH.")


def load_predictor(ckpt_local: str):
    medsam2_path = _find_medsam2()
    if medsam2_path not in sys.path:
        sys.path.insert(0, medsam2_path)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not os.path.exists(ckpt_local):
        print("Downloading CA-SAM2 checkpoint from HuggingFace...")
        from huggingface_hub import hf_hub_download
        ckpt_local = hf_hub_download(
            repo_id=HF_REPO, filename=HF_FILE,
            local_dir=os.path.dirname(ckpt_local) or ".",
        )

    cfg = os.path.join(medsam2_path, "configs", "sam2.1_hiera_t512.yaml")
    model = build_sam2(cfg, ckpt_local, device=DEVICE)
    model.eval()
    return SAM2ImagePredictor(model)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def centroid_prompt(mask_np: np.ndarray):
    if mask_np.sum() == 0:
        h, w = mask_np.shape
        return w // 2, h // 2
    cy, cx = center_of_mass(mask_np > 0)
    return int(cx), int(cy)


def run_inference(predictor, image_rgb: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Single centroid click → binary mask [H, W] uint8."""
    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(
        point_coords=np.array([[cx, cy]]),
        point_labels=np.array([1]),
        multimask_output=False,
    )
    return (masks[0] > 0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="/home/jupyter/arcade_val")
    parser.add_argument("--output_dir", default="/home/jupyter/endpoint_results")
    parser.add_argument("--ckpt",       default="/home/jupyter/medsam2_arcade_v2.pt")
    parser.add_argument("--gcs_out",    default=f"{GCS_BUCKET}/endpoint_results")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "overlays"), exist_ok=True)

    pull_arcade_val(args.data_dir)
    predictor = load_predictor(args.ckpt)

    img_paths  = sorted(glob.glob(args.data_dir + "/images/*.png"))
    mask_paths = sorted(glob.glob(args.data_dir + "/masks/*.png"))
    if args.max_images:
        img_paths  = img_paths[:args.max_images]
        mask_paths = mask_paths[:args.max_images]

    print(f"Running endpoints on {len(img_paths)} images  [{DEVICE}]")

    all_results = []

    for ip, mp in zip(img_paths, mask_paths):
        stem     = os.path.splitext(os.path.basename(ip))[0]
        img_rgb  = np.array(Image.open(ip).convert("RGB").resize((512, 512)))
        gt_mask  = np.array(Image.open(mp).convert("L").resize((512, 512), Image.NEAREST))

        cx, cy    = centroid_prompt(gt_mask)
        pred_mask = run_inference(predictor, img_rgb, cx, cy)

        ep = compute_all_endpoints(img_rgb, pred_mask)

        row = {
            "stem":              stem,
            "diameter_mean_px":  ep["diameter"]["mean"],
            "diameter_min_px":   ep["diameter"]["min"],
            "diameter_max_px":   ep["diameter"]["max"],
            "n_stenosis":        len(ep["stenosis_candidates"]),
            "n_occlusion":       len(ep["occlusion_candidates"]),
            "n_stent_lines":     len(ep["stent_candidates"]),
            "stenosis_candidates":   ep["stenosis_candidates"],
            "occlusion_candidates":  ep["occlusion_candidates"],
        }
        all_results.append(row)

        vis = visualize_endpoints(img_rgb, pred_mask, ep)
        out_path = os.path.join(args.output_dir, "overlays", f"{stem}.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        print(f"  {stem:30s}  diam={row['diameter_mean_px']:.1f}px  "
              f"stenosis={row['n_stenosis']}  "
              f"occlusion={row['n_occlusion']}  "
              f"stent={row['n_stent_lines']}")

    out_json = os.path.join(args.output_dir, "endpoint_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} results → {out_json}")
    print(f"Uploading to {args.gcs_out} ...")
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", args.output_dir + "/", args.gcs_out + "/"],
        check=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
