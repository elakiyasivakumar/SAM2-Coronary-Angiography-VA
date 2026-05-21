# Video Evaluation — CoronaryDominance Fluoroscopic Video

Qualitative comparison of **zero-shot MedSAM2** vs **ARCADE v2 fine-tuned MedSAM2** on coronary X-ray fluoroscopy video from the [CoronaryDominance dataset](https://www.nature.com/articles/s41597-025-04331-6) (Scientific Data, 2025).

## Sample Video

`vid5_comparison.mp4` — three-panel side-by-side across all frames:

| Panel | Description |
|-------|-------------|
| Original | Raw fluoroscopy frames |
| Zero-Shot MedSAM2 | No fine-tuning, centroid click prompt |
| Fine-Tuned MedSAM2 | ARCADE v2 checkpoint, same prompt |

## Evaluation Summary (10 studies)

10 studies (vid1–vid10) were evaluated qualitatively. A single centroid click on the auto-selected prompt frame (highest pixel variance = peak contrast fill) is used for both models. SAM2's memory module propagates the mask forward and backward without re-prompting.

| Finding | Detail |
|---------|--------|
| Vessel tracking | Fine-tuned model tracks the main trunk with tighter boundary precision across frames |
| **Specificity (key result)** | **9/10 studies: fine-tuned model did not falsely segment ribs, stents, or bypass grafts as blood vessels** — zero-shot model frequently did |
| Contrast bolus degradation | Both models degrade when contrast clears between injection cycles; SAM2's 6-frame FIFO memory loses the vessel target |

## Prompt Strategy

- **Prompt frame**: frame with highest pixel variance (approximates peak contrast fill)
- **Centroid**: mean position of pixels above 85th percentile of a Gaussian center-weighted intensity map (σ = min(H,W)/3), downweighting bright border artifacts
- Same centroid-click strategy used at both training and inference

## Full Results

Frame comparison grids and MP4s for all 10 studies available on request — contact [es4033@columbia.edu](mailto:es4033@columbia.edu).

## Dataset

CoronaryDominance provides coronary dominance classification labels only — no per-frame vessel segmentation masks exist. This evaluation is qualitative. Quantitative video Dice requires future clinician-annotated ground truth.

> Danilov et al., *Scientific Data* 2025. [https://doi.org/10.1038/s41597-025-04331-6](https://doi.org/10.1038/s41597-025-04331-6)
