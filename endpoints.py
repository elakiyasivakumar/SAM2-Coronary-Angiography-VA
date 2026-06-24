"""
Clinical endpoint extraction from coronary vessel segmentation masks.

Four endpoints derived from binary mask + original image:
  1. Vessel diameter  — distance transform along medial axis
  2. Stenosis         — skeleton points where diameter drops >35% below median
  3. Occlusion        — interior skeleton terminations and isolated fragments
  4. Stent candidates — bright rigid linear structures overlapping the mask

All spatial values are in pixels unless px_spacing_mm is provided.
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndlabel
from skimage.morphology import skeletonize, remove_small_objects


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skeleton_and_radii(mask: np.ndarray):
    """Binary mask → (skeleton bool array, (N,2) pts, (N,) radii in px)."""
    binary = (mask > 0).astype(bool)
    binary = remove_small_objects(binary, min_size=64)
    edt = distance_transform_edt(binary)
    skel = skeletonize(binary)
    rows, cols = np.where(skel)
    radii = edt[rows, cols]
    pts = np.column_stack([rows, cols]) if len(rows) else np.empty((0, 2), int)
    return skel, pts, radii


def _neighbor_count(skel: np.ndarray, r: int, c: int) -> int:
    patch = skel[max(0, r - 1):r + 2, max(0, c - 1):c + 2].copy()
    patch[min(r, 1), min(c, 1)] = 0
    return int(patch.sum())


# ---------------------------------------------------------------------------
# Endpoint 1 — Vessel diameter
# ---------------------------------------------------------------------------

def compute_diameters(mask: np.ndarray, px_spacing_mm: float = None) -> dict:
    """
    Returns per-skeleton-point diameters.

    Keys: skel_pts (N,2), radii_px (N,), diameters (N,), mean, min, max, unit
    """
    _, skel_pts, radii_px = _skeleton_and_radii(mask)
    scale = px_spacing_mm if px_spacing_mm else 1.0
    unit = "mm" if px_spacing_mm else "px"
    diameters = radii_px * 2.0 * scale
    return {
        "skel_pts":  skel_pts,
        "radii_px":  radii_px,
        "diameters": diameters,
        "mean":  float(np.mean(diameters)) if len(diameters) else 0.0,
        "min":   float(np.min(diameters))  if len(diameters) else 0.0,
        "max":   float(np.max(diameters))  if len(diameters) else 0.0,
        "unit":  unit,
    }


# ---------------------------------------------------------------------------
# Endpoint 2 — Stenosis candidates
# ---------------------------------------------------------------------------

def find_stenosis_candidates(
    radii_px: np.ndarray,
    skel_pts: np.ndarray,
    drop_fraction: float = 0.35,
) -> list:
    """
    Skeleton points where radius is >drop_fraction below the global median.
    Returns list of dicts: {row, col, diameter_px, drop_fraction}
    """
    if len(radii_px) == 0:
        return []
    median_r = np.median(radii_px)
    if median_r < 1e-6:
        return []
    out = []
    for (r, c), rad in zip(skel_pts, radii_px):
        drop = (median_r - rad) / median_r
        if drop > drop_fraction:
            out.append({
                "row": int(r), "col": int(c),
                "diameter_px": float(rad * 2),
                "drop_fraction": float(drop),
            })
    return out


# ---------------------------------------------------------------------------
# Endpoint 3 — Occlusion candidates
# ---------------------------------------------------------------------------

def find_occlusion_candidates(mask: np.ndarray) -> list:
    """
    Interior skeleton endpoints (degree-1 nodes not at image border) and
    small isolated skeleton fragments — both suggest vessel termination or gap.

    Returns list of dicts: {row, col, type}
      type: "endpoint" | "isolated_fragment"
    """
    skel, skel_pts, _ = _skeleton_and_radii(mask)
    H, W = mask.shape
    BORDER = 8
    out = []

    for r, c in skel_pts:
        if _neighbor_count(skel, r, c) == 1:
            if BORDER < r < H - BORDER and BORDER < c < W - BORDER:
                out.append({"row": int(r), "col": int(c), "type": "endpoint"})

    labeled, n = ndlabel(skel)
    for comp_id in range(1, n + 1):
        comp_pts = np.column_stack(np.where(labeled == comp_id))
        if 0 < len(comp_pts) < 20:
            rm, cm = comp_pts.mean(axis=0)
            if BORDER < rm < H - BORDER and BORDER < cm < W - BORDER:
                out.append({"row": int(rm), "col": int(cm), "type": "isolated_fragment"})

    return out


# ---------------------------------------------------------------------------
# Endpoint 4 — Stent candidates
# ---------------------------------------------------------------------------

def detect_stent_candidates(
    image: np.ndarray,
    mask: np.ndarray,
    min_line_length: int = 25,
    max_line_gap: int = 6,
) -> list:
    """
    Bright, rigid linear structures overlapping the vessel mask → stent candidates.
    Uses bright-pixel threshold + Canny + Probabilistic Hough inside the mask ROI.

    Returns list of (x1, y1, x2, y2) line segments.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()

    vessel_px = gray[mask > 0]
    thresh = int(np.percentile(vessel_px, 80)) if len(vessel_px) > 0 else 200
    _, bright = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    dilated_mask = cv2.dilate(
        (mask > 0).astype(np.uint8) * 255,
        np.ones((7, 7), np.uint8), iterations=2,
    )
    roi = cv2.bitwise_and(bright, dilated_mask)
    edges = cv2.Canny(roi, 40, 120)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=15,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    return [tuple(map(int, l[0])) for l in lines] if lines is not None else []


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_all_endpoints(
    image: np.ndarray,
    mask: np.ndarray,
    px_spacing_mm: float = None,
) -> dict:
    """
    Runs all four endpoint extractors and returns a unified results dict.

    image : uint8 [H, W, 3] RGB (or [H, W] grayscale)
    mask  : uint8 or bool [H, W], vessel = nonzero
    """
    diam = compute_diameters(mask, px_spacing_mm)
    return {
        "diameter":             diam,
        "stenosis_candidates":  find_stenosis_candidates(diam["radii_px"], diam["skel_pts"]),
        "occlusion_candidates": find_occlusion_candidates(mask),
        "stent_candidates":     detect_stent_candidates(image, mask),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_endpoints(image: np.ndarray, mask: np.ndarray, results: dict) -> np.ndarray:
    """
    Returns RGB uint8 overlay with all four endpoints drawn.

    Skeleton: red (thin) → green (wide)
    Stenosis: yellow circles + % drop label
    Occlusion: red circles
    Stent: cyan lines
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()

    overlay = vis.copy()
    overlay[mask > 0] = [0, 200, 0]
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    skel_pts = results["diameter"]["skel_pts"]
    radii_px  = results["diameter"]["radii_px"]
    if len(skel_pts):
        max_r = radii_px.max() + 1e-6
        for (r, c), rad in zip(skel_pts, radii_px):
            t = rad / max_r
            cv2.circle(vis, (c, r), 1, (int(255 * (1 - t)), int(255 * t), 0), -1)

    for s in results["stenosis_candidates"]:
        cv2.circle(vis, (s["col"], s["row"]), 7, (255, 220, 0), 2)
        cv2.putText(vis, f"{s['drop_fraction']:.0%}",
                    (s["col"] + 8, s["row"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1)

    for o in results["occlusion_candidates"]:
        cv2.circle(vis, (o["col"], o["row"]), 9, (255, 60, 60), 2)

    for x1, y1, x2, y2 in results["stent_candidates"]:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)

    return vis
