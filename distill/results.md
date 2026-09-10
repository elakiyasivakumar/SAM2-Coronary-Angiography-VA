# Distillation Results

Teacher: CA-SAM2 (MedSAM2 Hiera-Tiny SAM2.1, 38.9M params, 512×512) — Dice 0.767 ± 0.082

All numbers below are on the **200 held-out ARCADE test images** with a deterministic centroid
click prompt. Source of truth: `gs://coronary-angio-v2/results/distillation/results_*.json`
(pulled 2026-09-10).

## Full Result Table

| Run | Script | Params | Ablation | KD response | clDice | Input | Test Dice | Test IoU |
|---|---|---|---|---|---|---|---|---|
| Vanilla MobileSAM (SA-1B) | `eval_vanilla_mobilesam.sh` | 10.1M | — | — | — | 1024 | 0.1082 ± 0.0908 | 0.060 |
| CA-SAM2 teacher | — | 38.9M | — | — | ✅ | 512 | 0.767 ± 0.082 | 0.629 |
| SAM2.1 fine-tune @1024 | `setup_sam2_1024.sh` | 39.0M | — | — | ✅ | **1024** | **0.7995 ± 0.0751** | 0.672 |
| MobileSAM v1 | `setup_distill.sh` | 10.1M | 4 | ✅ | ✅ | 1024 | 0.6233 ± 0.0898 | 0.459 |
| RepViT-SAM | `setup_repvit.sh` | 8.9M | 4 | ✅ | ✅ | 1024 | 0.6415 ± 0.0792 | 0.477 |
| MobileSAM v3 (clean split) | `setup_mobilesam_v3.sh` | 10.1M | 4 | ✅ | ✅ | 1024 | 0.6869 ± 0.0768 | 0.528 |
| **MobileSAM v4 (GT only)** | `setup_mobilesam_finetune.sh` | 10.1M | **1** | ❌ | ❌ | 1024 | **0.8057 ± 0.0862** | 0.683 |
| MobileSAM v6 (+ feature KD) | `setup_mobilesam_v6.sh` | 10.1M | 1 | ❌ | ❌ | 1024 | 0.8047 ± 0.0804 | 0.680 |
| MobileSAM v7 (+ neck adapter) | `setup_mobilesam_v7.sh` | 10.1M | 1 | ❌ | ❌ | 1024 | 0.8072 ± 0.0788 | 0.684 |

Checkpoints: `gs://coronary-angio-v2/checkpoints/` — `medsam2_arcade_v2.pt`,
`mobilesam_abl1_v6.pt`, `mobilesam_abl1_v7.pt`, `sam2_fluoroscopy_1024.pt`.
HuggingFace: `Elakiya17/CA-SAM2`, `Elakiya17/fluoroscopy-sam2`.
(`Elakiya17/medsam_distill_models`, referenced by `setup_video_compare.sh`, is not publicly
reachable — private or removed. The v4 checkpoint is not in GCS either.)

## Failed / Missing Runs

| Run | Script | Status |
|---|---|---|
| v2 | `setup_mobilesam_v2.sh` | corrupt — dropped |
| v5 (MobileSAM @512) | `setup_mobilesam_v5.sh` | **crashed** — `AssertionError: input feature has wrong size` in `tiny_vit_sam.py:338`; TinyViT window attention is hardcoded for 1024 |
| True fine-tune (no transplant) | `setup_mobilesam_trueft.sh` | **crashed** — `TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'` during MobileSAM pip install |

Both missing runs isolate something the paper currently asserts without measurement (see below).

## Interpretation

**1. Resolution, not distillation, explains the student beating the teacher.**
Holding architecture/split/prompt/loss/augmentation fixed, SAM2.1 goes 0.767 @512 → 0.7995 @1024
(+0.033). MobileSAM v4 at 1024 scores 0.8057. So v4's margin over an *equally-resolved* SAM2.1 is
**+0.006**, against σ ≈ 0.08 — indistinguishable. The headline "10.1M student beats its 38.9M
teacher by 5.1%" is a resolution artifact, not a distillation result.

**2. The ablation 4 → 1 step moves two variables at once.**
`distill_student.py:530-531`: `use_kd = ablation >= 2` and `use_cldice = ablation >= 3`. Going from
4 to 1 removes response KD **and** clDice simultaneously. The +0.119 gain (0.687 → 0.806) cannot be
attributed to KD removal alone. clDice is not neutral — it is credited with the teacher's largest
single-run gain in `results_log.md` (run #7, +0.040).

To isolate, two runs are needed and neither exists:
- `--ablation 3` — clDice on, KD off
- `--ablation 2` — KD on, clDice off

**3. Unresolved: the v1 vs v3 labeling.**
`paper_draft.md` Finding 2 claims v1 (leaky split) = 0.688 and v3 (clean split) = 0.687, concluding
the split correction "changed nothing." But the JSONs say `results_mobilesam_abl4.json` = 0.6233 and
`results_mobilesam_abl4_v3.json` = 0.6869. The previous version of this file listed 0.688/0.076/0.529
for "abl4", which matches the **v3** JSON exactly — suggesting the un-suffixed run is v1 at 0.6233
and the split correction was worth +0.064, not zero. **Needs confirmation before the paper claim
stands.**

**4. Feature KD and the neck adapter are null results.**
v4 0.8057 / v6 0.8047 / v7 0.8072 — a 0.0025 spread across three runs with σ ≈ 0.08. Neither
foreground-weighted feature matching nor an in-path neck adapter moves the needle over GT-only.
