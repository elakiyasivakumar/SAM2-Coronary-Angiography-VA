# SAM2.1 Fluoroscopy 1024×1024 — Architecture & Training Protocol

**Experiment goal:** determine whether 1024×1024 resolution alone explains MobileSAM v4's 0.806 Dice advantage over the 512×512 CA-SAM2 teacher (0.767 Dice). If SAM2.1 at 1024×1024 matches or exceeds 0.806, resolution is the dominant factor; if it lands near 0.767, it is the architecture swap (TinyViT→Hiera).

---

## Architecture: SAM2.1 Hiera-Tiny at 1024×1024

Source: `facebookresearch/sam2` — `sam2.1_hiera_t.yaml` (image_size: **1024**, not MedSAM2's 512-patched config).

```
Input: 1024 × 1024 × 3  (coronary angiogram, grayscale→3ch)
│
▼
┌─────────────────────────────────────────────────────────────────────────┐
│  IMAGE ENCODER: Hiera-Tiny Trunk  (38.9M params total)                 │
│                                                                         │
│  Patch Embed  4×4                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Stage 1   blocks[0–1]    256×256 tokens   stride 4   ❄  FROZEN  │  │
│  │  Stage 2   blocks[2–3]    128×128 tokens   stride 8   ❄  FROZEN  │  │
│  │  Stage 3   blocks[4–9]     64×64 tokens    stride 16  ❄  FROZEN  │  │
│  │  Stage 4   blocks[10–11]   32×32 tokens    stride 32  🔥 UNFROZEN│  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  FPN Neck (3-level pyramid)                            🔥 UNFROZEN     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Scale 0:  128×128 × 32   (from stage 2 skip)                    │  │
│  │  Scale 1:   64×64 × 64   (from stage 3 skip)                     │  │
│  │  Scale 2:   32×32 × 256  (from stage 4, deepest)                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
         │ image_embed 32×32×256
         │ high_res_features [128×128×32, 64×64×64]
▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PROMPT ENCODER                                        🔥 UNFROZEN     │
│  Input: centroid click (x, y) → point token                            │
│  PE: dense sinusoidal embedding for image_embed                         │
└─────────────────────────────────────────────────────────────────────────┘
         │ sparse_emb, dense_emb
▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MASK DECODER (Two-way Transformer)                    🔥 UNFROZEN     │
│  image_embed 32×32  +  high_res_features [128×128, 64×64]              │
│  → Two-way attention (image↔prompt tokens, 2 layers)                   │
│  → MLP → mask logit 256×256                                             │
│  → upsample → 1024×1024 logit → resize to native (512×512 for ARCADE)  │
└─────────────────────────────────────────────────────────────────────────┘
         │
▼
  Binary segmentation mask  (coronary arteries)
```

**Key difference from CA-SAM2 v1 (512×512):**  
At 1024×1024, Stage 4 produces 32×32 tokens instead of 16×16, and the FPN neck Scale 0 is 128×128 instead of 64×64. The mask decoder works with 4× more spatial tokens throughout — this is why resolution matters for thin vessels.

---

## V2 Training Protocol

Mirrors the CA-SAM2 v2 notebook (`arcade_v2_executed.ipynb`) that improved Dice from 0.727 → 0.767.

### Freeze / unfreeze

| Component | Status | Rationale |
|---|---|---|
| Trunk blocks[0–9] | ❄ Frozen | General vision features; 900 ARCADE images too few to retrain |
| Trunk blocks[10–11] | 🔥 Unfrozen | Deepest semantic features; need to adapt to X-ray texture |
| FPN neck | 🔥 Unfrozen | Multi-scale pyramid must adapt; output feeds decoder |
| Prompt encoder | 🔥 Unfrozen | Small (<<1M params); click→embedding needs domain calibration |
| Mask decoder | 🔥 Unfrozen | Core segmentation head; all params trained |

### Discriminative learning rates

```
Decoder  ── 5e-5  (fastest — primary task head)
Neck     ── 1e-5  (moderate — bridges encoder→decoder)
Trunk[10,11] ── 5e-6  (slowest — fine adaptation only)
```

Technique from Howard & Ruder (2018) ULMFiT; validated on SAM2 in MedSAM2 (Wang et al. 2024).

### Loss function

```
L = 0.5 × Dice  +  0.2 × wBCE  +  clDice_w × clDice
```

**clDice warm-up schedule:**

| Epoch | clDice weight |
|---|---|
| 1–3 | 0.00 (off) |
| 4 | 0.06 |
| 5 | 0.12 |
| 6 | 0.18 |
| 7 | 0.24 |
| 8+ | 0.30 (full) |

clDice (Shit et al. 2021, CVPR) optimizes the soft skeleton overlap — critical for coronary vessels where thin branches are topologically important but contribute few pixels. Starting it at zero avoids noisy skeleton gradients before the model converges.

### Data augmentation (5× offline)

| Variant | Transform |
|---|---|
| `orig` | No transform |
| `hf` | Horizontal flip |
| `vf` | Vertical flip |
| `r20` | Rotate −20° (bicubic img, nearest mask) |
| `r20_hf` | Rotate −20° + horizontal flip |

900 training images × 5 = **4,500 augmented images** (same as v2 notebook). Generated once to `/home/jupyter/arcade_aug5k/`.

### Other hyperparameters

| Parameter | Value |
|---|---|
| Max epochs | 50 |
| Early stopping patience | 10 (val Dice) |
| Batch size | 4 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| LR schedule | Cosine annealing (T_max=50) |
| Gradient clip | 0.5 |
| Mixed precision | fp16 autocast + GradScaler |
| Prompt | Centroid click (jitter ±10px during train) |
| Val prompt | Centroid click (no jitter) |

---

## Research rationale

### Is this a sound approach?

**Yes — with caveats.** Key evidence:

1. **Partial-unfreeze SAM fine-tuning on small medical datasets works.**  
   SAMed (Cheng et al. 2023), MedSAM (Ma et al. 2024), SA-Med2D (Ye et al. 2023) all validate that freezing the early encoder and fine-tuning only the deep blocks + decoder prevents catastrophic forgetting on 500–2000-image datasets. Our 900-image ARCADE set is within this range.

2. **clDice is purpose-built for coronary vessels.**  
   Shit et al. (2021) introduced clDice specifically for tubular structure segmentation (vessels, roads, neurons). Their experiments on coronary angiography showed +2–4% Dice improvement over standard Dice loss at the same architecture — directly applicable here.

3. **Discriminative LRs prevent over-adapting deep pretrained features.**  
   Validated across NLP (ULMFiT) and vision fine-tuning (ViT-B on medical imaging). The key insight: the FPN neck must adapt quickly (it bridges two domains — natural images → X-ray), but the trunk blocks should shift minimally so pretraining generalises.

4. **1024×1024 is well-motivated for thin coronary vessels.**  
   Coronary arteries are 1.5–5mm in diameter. At 512px resolution on a typical 200mm field of view, a 1.5mm vessel is ~4px wide. At 1024px it is ~8px — within the attention receptive field of the mask decoder. The resolution hypothesis is plausible and worth testing.

### Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Overfitting to 900 images | Medium | Early stopping (patience=10), partial freeze, weight decay 0.01 |
| `BB_FEAT_SIZES` mismatch (hard-coded) | Low-Medium | Must verify against `model.image_size` at runtime; add assertion |
| Centroid click prompt bias | Low | Jitter ±10px in training; real-world use should use bounding box or multiple clicks for better robustness |
| SAM2 video memory overhead | Low | `apply_postprocessing=False` disables the memory module; we use image-only mode |

### What this experiment decides

```
                    SAM2.1 1024×1024 result
                         /           \
              Dice ≥ 0.800          Dice ≈ 0.767
                   /                       \
    Resolution is the key factor     Resolution not sufficient
    MobileSAM v4's advantage is      Architecture gap (TinyViT encoder)
    primarily from running at 1024   is the primary bottleneck
         |                                 |
    Paper story A:                   Paper story B:
    Resolution > distillation        Decoder transplant is the key
```

---

## Compute and timing

| Hardware | GPU | Estimated time | Cost |
|---|---|---|---|
| L4 VM (current plan) | 24GB VRAM | 20–26 hrs | ~$18 |
| Workbench + A100 40GB | 40GB VRAM | 8–10 hrs | ~$35–40 |

**Why 20–26 hrs vs the previous 6–8 hrs (CA-SAM2 v2):**  
CA-SAM2 v2 ran at **512×512** on a V100. The Hiera-Tiny trunk processes 4× more spatial tokens at 1024×1024, and the FPN neck + mask decoder scale with sequence length. The L4 is ~2× faster than V100 per FLOP, but the 4× resolution increase more than offsets that. Net result: ~2.5–3× more wall-clock per epoch. With a higher epoch cap (50 vs 20), the total is 3–4× longer.

**Recommendation:** if turnaround time matters (e.g., paper deadline), use Workbench with A100 (`n1-standard-8 + a100-40gb` in `us-central1-a`). If cost matters, L4 VM is fine overnight.
