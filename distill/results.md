# Distillation Results

Teacher: CA-SAM2 (Hiera-Tiny SAM2.1, 38.9M params) — Dice 0.767 ± 0.082

| Model | Params | Ablation | Dice | Dice Std | IoU | HuggingFace | Notes |
|---|---|---|---|---|---|---|---|
| MobileSAM | 10.1M | 4 (full) | 0.688 | 0.076 | 0.529 | `Elakiya17/medsam_distill_models/mobilesam_abl4.pt` | L4 GPU, batch 4, 30 epochs |
| RepViT-SAM | 8.9M | 4 (full) | 0.641 | 0.079 | 0.477 | `Elakiya17/medsam_distill_models/repvitsam_abl4.pt` | L4 GPU, batch 4, 30 epochs |

GCS: `gs://coronary-angio-v2/checkpoints/`

## Summary
- MobileSAM: 0.079 Dice gap vs teacher (~10% degradation for 4× size reduction)
- RepViT-SAM: 0.126 Dice gap vs teacher, but smallest model (8.9M) and fastest CPU inference
- Neither model has been hyperparameter tuned — first-run baselines only
