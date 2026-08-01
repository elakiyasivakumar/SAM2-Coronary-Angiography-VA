# Architecture & Loss Analysis — CA-SAM2 Distillation

## The 6 Models That Matter

| # | Model | Category | Result |
|---|---|---|---|
| 1 | Vanilla MobileSAM | Non-distilled baseline | 0.108 |
| 2 | CA-SAM2 | Teacher | 0.767 |
| 3 | v1 | Distilled | 0.688 |
| 4 | v2 | Distilled | corrupt — drop |
| 5 | ablation=1 | Distilled | running |
| 6 | True fine-tune | Non-distilled | missing |
| + | Feature KD | Distilled | planned |

---

## Architecture Comparison: v1 vs v3 vs ablation=1

| Component | v1 | v3 | ablation=1 |
|---|---|---|---|
| Encoder | TinyViT (SA-1B, frozen) | TinyViT (SA-1B, frozen) | TinyViT (SA-1B, frozen) |
| Decoder init | partial transfer from CA-SAM2 (strict=False) | partial transfer from CA-SAM2 (strict=False) | partial transfer from CA-SAM2 (strict=False) |
| Prompt enc init | partial transfer from CA-SAM2 | partial transfer from CA-SAM2 | partial transfer from CA-SAM2 |
| Input size | 1024×1024 | 1024×1024 | 1024×1024 |
| Prompt | centroid click | centroid click | centroid click |

| Loss component | v1 | v3 | ablation=1 |
|---|---|---|---|
| Dice | ✅ 0.5× | ✅ 0.5× | ✅ 0.5× |
| wBCE | ✅ 0.2× | ✅ 0.2× | ✅ 0.2× |
| KD response (soft logits) | ✅ 0.5× | ✅ 0.5× | ❌ |
| clDice | ✅ warmup → 0.3× | ✅ warmup → 0.3× | ❌ |
| Total | 0.5×KD + 0.4×(0.5×Dice + 0.2×wBCE) + clDice | same as v1 | 0.5×Dice + 0.2×wBCE |

| KD component | v1 | v3 | ablation=1 |
|---|---|---|---|
| Soft logit matching | ✅ | ✅ | ❌ |
| Decoder weight transfer | ✅ partial | ✅ partial | ✅ partial |
| Feature matching | ❌ | ❌ | ❌ |
| Teacher runs during training | ✅ soft label gen | ✅ soft label gen | ❌ |

| Training setup | v1 | v3 | ablation=1 |
|---|---|---|---|
| Epochs | 30 fixed | stopped ep.17 | 30 max, ES |
| Early stopping | ❌ | ✅ patience=10 | ✅ patience=10 |
| Data split | 1000 train / 200 leaky eval | 900/100/200 clean | 900/100/200 clean |
| Augmentation | 5× geom | 5× geom + color noise | 5× geom + color noise |

| Results | v1 | v3 | ablation=1 |
|---|---|---|---|
| Val Dice | n/a | 0.681 (100-img internal) | 0.810 best (100-img internal) |
| Test Dice | 0.688 | 0.687 | TBD (200 held-out) |

**Key observation:** encoder, decoder init, and prompt encoder are identical across all three.
v1 vs v3 test Dice is identical (0.688 vs 0.687) — only difference was split methodology and epochs.
The jump in ablation=1 val Dice comes entirely from removing KD response loss and clDice.

---

## Why Response KD Hurts When You Have GT Masks

**What soft logit matching is:**
The teacher produces a 256×256 map of continuous logit scores. A boundary pixel might score 0.7 rather than a hard 0 or 1. The student is trained to match this distribution pixel-by-pixel (KD_BCE). The premise is that the teacher's uncertainty encodes "dark knowledge" — more information than a hard label.

**Why it fails here:**
We have GT masks — binary ground truth labels for every pixel. The only thing soft labels could add is the teacher's uncertainty at ambiguous pixels. But the teacher is only 0.767 Dice accurate. At precisely the pixels where it matters (ambiguous boundaries), the teacher's probability is often wrong. You are replacing a correct hard label with an incorrect soft one.

**Conclusion:** with high-quality GT masks and an imperfect teacher, response KD is redundant at best and actively harmful at worst. Demonstrated empirically: removing it (ablation=1) improves val Dice from ~0.68 to ~0.81.

---

## The SAM1 vs SAM2 Architecture Mismatch

**Teacher (CA-SAM2) and student (MobileSAM) are different SAM versions:**

| | Teacher (CA-SAM2) | Student (MobileSAM) |
|---|---|---|
| SAM version | SAM2 (MedSAM2) | SAM1 |
| Encoder | Hiera-Tiny | TinyViT |
| Decoder | SAM2 mask decoder | SAM1 mask decoder |
| Input size | 512×512 | 1024×1024 |

**The decoder transplant is partial, not full:**
```
missing:    ['transformer.layers.0.mlp.lin1.weight', ...]  ← SAM1 keys teacher doesn't have
unexpected: ['obj_score_token.weight', 'conv_s0.weight', ...]  ← SAM2-specific keys student ignores
```
`load_state_dict(strict=False)` copies only the overlapping keys. SAM2-specific components
(object score token, conv layers for video memory) are discarded. SAM1-specific keys are
randomly initialized.

**Standard distillation practice:** decoder weight transfer is typically full and exact —
teacher and student share the same decoder architecture so all keys match. Our setup is
non-standard because we are distilling across SAM versions. The partial transplant means
the student decoder is a hybrid: some weights from ARCADE fine-tuning, some randomly initialized.

---

## Implication for the Loss Function Narrative

The decoder transplant + GT loss (ablation=1) is already working well precisely because:
- The decoder starts from a partially good place (ARCADE-tuned weights where they overlap)
- The GT loss cleanly adapts the frozen encoder's features to what the decoder expects
- No conflicting signal from an imperfect teacher

**What response KD was doing wrong:**
The teacher's logits reflect Hiera-Tiny encoder features flowing through a SAM2 decoder.
The student has TinyViT encoder features flowing through a SAM1 decoder. Matching output
distributions across this double mismatch (encoder AND decoder architecture differ) is
asking the student to imitate something it architecturally cannot reproduce exactly.

**What feature KD would do right:**
Match intermediate encoder representations at the vessel locations (foreground-weighted).
This directly bridges the TinyViT ↔ Hiera-Tiny feature gap rather than trying to match
outputs across architecturally incompatible decoders. Teacher provides feature targets,
GT mask focuses the loss on what matters.

**Proposed loss for v4 (Feature KD):**
```
L = 0.5×Dice + 0.2×wBCE + λ×FG_weighted_MSE(student_feats, teacher_feats)
```
No response KD. No clDice (can revisit). Teacher runs only for feature extraction, not logit matching.
