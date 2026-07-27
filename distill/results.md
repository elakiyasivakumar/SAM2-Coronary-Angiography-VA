# Distillation Results

Teacher: CA-SAM2 (Hiera-Tiny SAM2.1, 38.9M params) — Dice 0.767 ± 0.082

| Model | Params | Ablation | Dice | Dice Std | IoU | Notes |
|---|---|---|---|---|---|---|
| MobileSAM | ~10M | 4 (full) | 0.688 | 0.076 | 0.529 | L4 GPU, batch 4, 30 epochs |
| RepViT-SAM | ~14M | 4 (full) | — | — | — | pending |

Checkpoint: `gs://coronary-angio-v2/checkpoints/mobilesam_abl4.pt`

Gap vs teacher: 0.079 Dice (~8 points). Reasonable for 4× smaller model, first run, no hyperparam tuning.
