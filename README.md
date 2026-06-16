# Fine-Tuning SAM2 for Coronary Artery Segmentation in X-Ray Fluoroscopy

Fine-tuned MedSAM2 on annotated coronary angiograms (ARCADE dataset) and applied it to fluoroscopic video. The model goes from Dice 0.033 zero-shot to **Dice 0.767 ± 0.082** on the ARCADE validation set (200 images) using a single centroid click prompt.

**Checkpoint:** [Elaks17/CA-SAM2 on HuggingFace](https://huggingface.co/Elaks17/CA-SAM2)

---

## What we did

**Base model:** [MedSAM2](https://github.com/bowang-lab/MedSAM2) (Hiera-Tiny backbone, 38.9M params)

**Dataset:** [ARCADE](https://zenodo.org/records/10390295) — 1,200 X-ray coronary angiogram images (512×512 px), 1,000/200 train/val split. Binary vessel masks from polygon annotations.

**Fine-tuning:**
- Partial encoder unfreeze — last 2 Hiera trunk blocks + FPN neck + mask decoder (~13.5M / 38.9M params trainable)
- Discriminative LRs: 5e-5 (decoder) → 1e-5 (neck) → 5e-6 (trunk blocks)
- Augmentation: 5× geometric variants per image (flip, rotate) → 5,000 effective training images
- Loss: `0.5 × Dice + 0.2 × wBCE + 0.3 × clDice` with clDice warmup (epochs 3→8)
- AdamW, cosine annealing, 40 epochs, FP16, batch 4, NVIDIA L4

**Prompt:** Single centroid click from ground-truth mask at training; auto-centroid from brightest center-weighted pixels at inference.

---

## Results

| Model | Dice | IoU |
|---|---|---|
| Zero-Shot MedSAM2 | 0.033 ± 0.062 | — |
| Frozen Encoder Fine-Tune | 0.727 ± 0.081 | 0.577 |
| **Fine-Tuned MedSAM2 (ours)** | **0.767 ± 0.082** | **0.629** |

---

## Video inference

Evaluated qualitatively on 10 fluoroscopy studies from [CoronaryDominance](https://www.kaggle.com/datasets/zhifanl/coronary-dominance) (no per-frame ground truth available). Fine-tuned weights loaded into MedSAM2's video predictor; prompt frame selected by highest pixel variance (peak contrast fill).

Key finding: in 9 of 10 studies, the fine-tuned model does not activate on ribs, stents, bypass grafts, or implanted devices — failure modes present in the zero-shot model in all 10 studies.

See `video_eval/` for a sample comparison video.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `arcade_v2.ipynb` | Fine-tuning pipeline (data loading, augmentation, training loop) |
| `arcade_evaluation.ipynb` | Dice/IoU evaluation on ARCADE val set |
| `video_eval.ipynb` | Video inference on CoronaryDominance studies |

---

## Usage

```python
from huggingface_hub import hf_hub_download
import torch, sys

# Clone MedSAM2 and install
# git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2 && pip install -e .
sys.path.insert(0, "MedSAM2")
from sam2.build_sam import build_sam2

ckpt = hf_hub_download(repo_id="Elaks17/CA-SAM2", filename="medsam2_arcade_v2.pt")
model = build_sam2("configs/sam2.1_hiera_t512.yaml", ckpt, device="cuda")
model.eval()
```

Prompt with a single centroid click `(cx, cy)` in pixel coordinates on a 512×512 image. See `arcade_evaluation.ipynb` for the full inference loop.

---

## Acknowledgements

- **Compute:** Google Cloud (Vertex AI Workbench, NVIDIA L4)
- **Datasets:** ARCADE ([Zenodo 10390295](https://zenodo.org/records/10390295)), CoronaryDominance
- **Base model:** MedSAM2 — Wang et al., 2024 ([GitHub](https://github.com/bowang-lab/MedSAM2))
- **clDice loss:** Shit et al., 2021 ([arXiv:2003.07311](https://arxiv.org/abs/2003.07311))
- **SAM2:** Ravi et al., 2024 ([arXiv:2408.00714](https://arxiv.org/abs/2408.00714))

---

## Citation

If you use this work, please cite:

```bibtex
@misc{sivakumar2026casam2,
  title   = {Fine-Tuning SAM2 for Coronary Artery Segmentation in X-Ray Fluoroscopy},
  author  = {Sivakumar, Elakiya},
  year    = {2026},
  note    = {Columbia University. Checkpoint: \url{https://huggingface.co/Elaks17/CA-SAM2}}
}
```
