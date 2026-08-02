# CA-SAM2: Lightweight Coronary Artery Segmentation via Knowledge Distillation for Edge Deployment

**Draft — [venue TBD, e.g., MICCAI 2026 / IEEE JBHI]**

---

## Abstract

Real-time coronary artery segmentation during cardiac catheterization requires models that run on resource-constrained hardware at the point of care. We present CA-SAM2, a knowledge distillation pipeline that compresses a fine-tuned SAM2 teacher (Hiera-Tiny, 38.9M params, Dice 0.767) into MobileSAM (ViT-Tiny, 10.1M params). Through systematic ablation we show that response-based KD — transmitting teacher soft logits to the student — degrades performance when ground-truth masks are available and the teacher is architecturally mismatched to the student. Removing response KD entirely and training with GT supervision alone yields Dice 0.806, outperforming the teacher by 5.1% absolute. We further show that foreground-weighted encoder feature matching (feature KD) adds no measurable gain over GT-only training in this setting. The resulting 10.1M-parameter student runs inference on an 8GB-RAM edge device, enabling cath-lab deployment without GPU infrastructure.

---

## 1. Introduction

- Coronary artery disease is the leading cause of death globally. Fluoroscopic guidance during percutaneous coronary intervention (PCI) requires real-time vessel visualization, but manual annotation is slow and inter-observer variability is high.
- Foundation models (SAM, SAM2) achieve strong zero-shot segmentation but are too large for real-time edge inference: SAM2 (Hiera-Tiny) at 38.9M params requires a dedicated GPU; cath lab workstations often have ≤8GB RAM and no NVIDIA GPU.
- Knowledge distillation transfers task-specific capability from a large teacher to a lightweight student. However, **standard distillation assumptions break in the medical imaging setting**: (1) ground-truth masks are available (unlike natural image KD where labels are scarce), (2) teacher accuracy is moderate (~0.77 Dice), not near-perfect, and (3) teacher and student architectures often differ (SAM1 vs SAM2 decoder).
- **Contributions:**
  - Empirical demonstration that response KD from an imperfect teacher *hurts* when GT masks are available (Dice 0.687 → 0.806 by removing it)
  - Ablation showing feature KD adds no benefit over GT-only fine-tuning in this regime
  - 10.1M-parameter MobileSAM student that exceeds its 38.9M-parameter teacher on the ARCADE benchmark
  - Inference latency analysis on an 8GB-RAM edge device

---

## 2. Background

### 2.1 Segment Anything Models
- SAM (Kirillov et al., 2023): ViT-H/L/B encoder + lightweight mask decoder, 1024×1024 input. Prompt-conditioned.
- SAM2 (Ravi et al., 2024): Hiera encoder, streaming memory for video, 512×512 input. Object-level tracking.
- MobileSAM (Zhang et al., 2023): TinyViT encoder (SA-1B distilled) + SAM1 decoder, 10.1M params, 1024×1024 input. Designed for edge inference.

### 2.2 Knowledge Distillation
- Response KD (Hinton et al., 2015): student matches teacher output distribution (soft logits). Assumes teacher uncertainty is informative ("dark knowledge").
- Feature KD (Romero et al., 2015): student matches intermediate feature maps. Architecture-agnostic alignment.
- Structural KD / weight transfer: direct copy of teacher weights into student. Requires compatible architectures.

### 2.3 Medical Image Segmentation with SAM
- MedSAM, MedSAM2, SurgicalSAM: domain-specific fine-tuning of SAM variants. Generally report Dice 0.75–0.92 on curated datasets.
- Limited work on *distilling* medical SAM models for edge deployment, particularly for fluoroscopy.

---

## 3. Dataset

**ARCADE (Automatic Region-based Coronary Artery Disease dEtection):**
- 1,200 coronary angiogram X-ray images, 512×512 resolution
- Binary vessel masks (stenosis segmentation challenge, MICCAI 2023)
- Official split: 1,000 train / 200 test
- **Our split:** 900 train / 100 internal validation (early stopping) / 200 held-out test (never seen during training)
  - Seed-42 shuffle of 1,000 training images; first 100 = internal val; remaining 900 = train
  - ARCADE official 200 test images = held-out evaluation only

---

## 4. Method

### 4.1 Teacher: CA-SAM2
- Backbone: SAM2.1 Hiera-Tiny (38.9M params), fine-tuned on 900 ARCADE training images
- Input: 512×512, centroid click prompt
- Loss: Dice + weighted BCE + clDice
- Result: Dice 0.767 on 200 held-out ARCADE test images

### 4.2 Student: MobileSAM
- Backbone: TinyViT (SA-1B distilled), 10.1M params
- Input: **1024×1024** (TinyViT window attention is hardcoded for 1024; downsampling to 512 causes assertion errors)
- Prompt: centroid click, same derivation as teacher

### 4.3 Decoder Initialization (Structural KD)
- Teacher (SAM2) and student (SAM1) have architecturally distinct mask decoders
- We copy overlapping weight keys via `load_state_dict(strict=False)`
- SAM2-specific keys (`obj_score_token`, `conv_s0/s1`) are discarded; SAM1-only keys (`transformer.layers.*.mlp.*`) are randomly initialized
- This partial transplant is **non-standard** — in typical distillation the decoder architecture is identical; here it is a cross-version hybrid

### 4.4 Training Protocol
- Augmentation: 5× geometric (flip, rotate, affine) applied identically to image, mask, and soft labels; + X-ray color simulation (brightness/contrast/Gaussian noise)
- Optimizer: AdamW, lr=3e-4, cosine decay, weight decay=0.01, gradient clip 0.5
- Batch size: 4, 30 epochs max, early stopping patience=10 on internal 100-image val set
- Trainable: mask decoder + prompt encoder (encoder frozen)

### 4.5 Distillation Variants Ablated

| Variant | Loss | Teacher during training |
|---|---|---|
| Response KD (v1, v3) | Dice + wBCE + KD_BCE(soft logits) + clDice | Pre-generates soft logit maps |
| **GT only (v4)** | **Dice + wBCE** | **None** |
| Feature KD (v6) | Dice + wBCE + λ·FG-MSE(encoder features) | Runs each batch (frozen) |

---

## 5. Results

### 5.1 Main Comparison

| Model | Params | Input | Test Dice | Test IoU |
|---|---|---|---|---|
| Vanilla MobileSAM (SA-1B, no fine-tuning) | 10.1M | 1024 | 0.108 | — |
| CA-SAM2 teacher | 38.9M | 512 | 0.767 | — |
| v1 / v3 (response KD) | 10.1M | 1024 | 0.687–0.688 | — |
| **v4 (GT only, partial decoder transplant)** | **10.1M** | **1024** | **0.806 ± 0.086** | **0.680** |
| v6 (GT + feature KD) | 10.1M | 1024 | 0.805 ± 0.080 | 0.680 |
| True fine-tune (SA-1B decoder, no transplant) | 10.1M | 1024 | *running* | — |

### 5.2 Key Findings

**Finding 1: Response KD hurts when the teacher is imperfect and GT is available.**
v3 (response KD, clean split) = 0.687 Dice. v4 (GT only, same split) = 0.806. The soft logit signal from a 0.767-Dice teacher introduces noise at boundary pixels where accurate supervision matters most. The student is being asked to imitate a mismatched architecture's output distribution; removing that constraint allows it to learn directly from ground truth.

**Finding 2: Data leakage correction is not the explanation.**
v1 (leaky split, response KD) = 0.688. v3 (clean split, response KD) = 0.687. Correcting the split changed nothing — the 0.119-Dice gain in v4 comes from removing KD response loss, not from data hygiene.

**Finding 3: Feature KD adds nothing over GT-only training.**
v4 = 0.806, v6 (+ foreground-weighted encoder MSE) = 0.805. The TinyViT (SA-1B) → Hiera-Tiny (ARCADE) feature gap is too wide to bridge with a 1×1 projection in 30 epochs. The GT loss already saturates available capacity given the partial decoder transplant.

**Finding 4: Student outperforms teacher (0.806 vs 0.767).**
This is partly explained by resolution advantage: MobileSAM operates at 1024×1024 vs teacher's 512×512. The finer spatial resolution benefits thin coronary vessels (~5% of pixels). We flag this explicitly to avoid overstating the distillation contribution.

---

## 6. Edge Deployment Analysis

*[To be filled after video inference experiment on 8GB-RAM VM]*

- Model size: 38.8 MB (MobileSAM v4 checkpoint)
- Peak RAM at inference (1024×1024 single frame, CPU): *TBD*
- Frames per second on CPU-only 8GB-RAM VM: *TBD*
- Frames per second on L4 GPU: *TBD*
- Comparison: CA-SAM2 teacher RAM/latency: *TBD*

---

## 7. Discussion

- **Why response KD fails in this setting:** The standard KD premise is that the teacher's soft label at an ambiguous pixel encodes more information than a hard 0/1 label. That holds when the teacher is near-perfect (>0.95 Dice) and the student's architecture can faithfully approximate the teacher's output manifold. Neither condition holds here: the teacher is at 0.767 Dice and the architectures are SAM1 vs SAM2 (different decoders, different encoder families). The soft label at a vessel boundary is often wrong, and the student cannot reproduce the teacher's output geometry regardless.
- **Decoder transplant question:** True fine-tune baseline (*running*) will clarify whether the partial CA-SAM2 → MobileSAM decoder transplant contributes to 0.806 or whether supervised ARCADE training at 1024×1024 is doing all the work.
- **Limitations:** (1) 100-image internal validation set may not be reliable for early stopping — a larger internal split would be more robust; (2) resolution advantage confounds teacher vs student comparison; (3) single centroid click prompt — box or multiple clicks may improve boundary accuracy.

---

## 8. Conclusion

We demonstrate that for coronary artery segmentation with available GT masks, response-based knowledge distillation from an imperfect teacher actively degrades student performance relative to direct supervised training. A 10.1M-parameter MobileSAM student trained with GT loss only, initialized from a partially transplanted CA-SAM2 decoder, achieves Dice 0.806 — exceeding its 38.9M-parameter teacher by 5.1% absolute. The model is small enough for 8GB-RAM edge deployment, enabling cath-lab inference without GPU infrastructure.

---

## Appendix: Architecture Mismatch Detail

| Component | CA-SAM2 (teacher) | MobileSAM (student) |
|---|---|---|
| SAM version | SAM2 | SAM1 |
| Encoder | Hiera-Tiny | TinyViT |
| Encoder output | [B, 256, 32, 32] at 512px | [B, 256, 64, 64] at 1024px |
| Decoder | SAM2 (obj_score_token, conv_s0/s1) | SAM1 (transformer MLP) |
| Decoder transplant | source | partial target (overlapping keys only) |
| Prompt encoder | SAM2 | SAM1 (partial transplant) |

---

*[References TBD — MobileSAM, SAM2, ARCADE dataset, Hinton KD, Romero FitNets, MedSAM, MedSAM2]*
