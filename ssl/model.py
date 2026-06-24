"""V-JEPA model for fluoroscopy video self-supervised learning.

Architecture:
  - Context encoder: MedSAM2 Hiera-Tiny, processes visible frames only
  - Target encoder: EMA copy of context encoder, processes all frames (frozen)
  - Predictor: 4-layer transformer, predicts target features for masked frames

Masking strategy (temporal):
  - Mask the last T//2 frames
  - Context encoder sees frames 0..(T//2 - 1)
  - Predictor predicts target encoder features for frames T//2..(T-1)
  - Directly models contrast fill dynamics: early frames → late frames

This approach preserves SAM2's image encoder API (no internal masking needed).
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight predictor transformer
# ---------------------------------------------------------------------------

class VJEPAPredictor(nn.Module):
    """
    Takes context encoder tokens + learned mask tokens, predicts target
    encoder representations for masked positions.
    """

    def __init__(self, embed_dim: int, predictor_dim: int = 384, depth: int = 4, num_heads: int = 6):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context_tokens: torch.Tensor, n_mask_tokens: int) -> torch.Tensor:
        """
        context_tokens: [B, n_context, embed_dim]
        n_mask_tokens:  number of masked positions to predict
        Returns: [B, n_mask_tokens, embed_dim] predictions
        """
        B = context_tokens.shape[0]
        x = self.input_proj(context_tokens)
        mask_tokens = self.mask_token.expand(B, n_mask_tokens, -1)
        tokens = torch.cat([x, mask_tokens], dim=1)  # [B, n_context + n_mask, predictor_dim]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        preds = self.output_proj(tokens[:, x.shape[1]:])  # [B, n_mask, embed_dim]
        return preds


# ---------------------------------------------------------------------------
# Frame-level encoder wrapper (calls SAM2 image encoder per frame)
# ---------------------------------------------------------------------------

class FrameEncoder(nn.Module):
    """
    Wraps the MedSAM2 image encoder to process a batch of frames independently.
    Returns mean-pooled spatial features per frame: [B, T, C].
    """

    def __init__(self, sam2_image_encoder: nn.Module):
        super().__init__()
        self.encoder = sam2_image_encoder

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips: [B, T, 3, H, W]
        Returns: [B, T, C] — one feature vector per frame
        """
        B, T, C, H, W = clips.shape
        flat = clips.reshape(B * T, C, H, W)

        out = self.encoder(flat)

        # SAM2 image encoder returns a dict or tuple with multi-scale features.
        # Use the last (most semantic) feature map.
        if isinstance(out, dict):
            feats = out.get('vision_features', list(out.values())[-1])
        elif isinstance(out, (list, tuple)):
            feats = out[-1]
        else:
            feats = out  # already a tensor

        # feats: [B*T, C, h, w] → global average pool → [B*T, C]
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])

        feats = feats.reshape(B, T, -1)  # [B, T, C]
        return feats


# ---------------------------------------------------------------------------
# Full V-JEPA model
# ---------------------------------------------------------------------------

class VJEPA(nn.Module):
    """
    V-JEPA for coronary fluoroscopy.

    Training objective: given the first T_context frames, predict the
    target encoder's representations of the last T_mask frames.
    """

    def __init__(self, sam2_image_encoder: nn.Module, ema_momentum: float = 0.996):
        super().__init__()
        self.ema_momentum = ema_momentum

        self.context_encoder = FrameEncoder(sam2_image_encoder)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Infer embed_dim with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 3, 64, 64)
            dummy_feats = self.context_encoder(dummy)
            embed_dim = dummy_feats.shape[-1]

        self.embed_dim = embed_dim
        self.predictor = VJEPAPredictor(
            embed_dim=embed_dim,
            predictor_dim=min(embed_dim * 2, 512),
            depth=4,
            num_heads=max(1, embed_dim // 64),
        )

    @torch.no_grad()
    def update_ema(self, step: int, total_steps: int):
        """Cosine-annealed EMA: momentum goes from ema_momentum → 1.0."""
        m = 1.0 - (1.0 - self.ema_momentum) * (math.cos(math.pi * step / total_steps) + 1) / 2
        for cp, tp in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tp.data.mul_(m).add_((1.0 - m) * cp.data)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips: [B, T, 3, H, W]
        Returns: scalar loss
        """
        B, T, C, H, W = clips.shape
        T_context = T // 2
        T_mask = T - T_context

        context_clips = clips[:, :T_context]  # [B, T_context, 3, H, W]

        # Context encoder: visible frames only
        context_feats = self.context_encoder(context_clips)  # [B, T_context, embed_dim]

        # Target encoder: all frames (no grad)
        with torch.no_grad():
            target_feats = self.target_encoder(clips)  # [B, T, embed_dim]

        target_masked = target_feats[:, T_context:]  # [B, T_mask, embed_dim]

        # Predictor: context features → predict masked frame features
        pred = self.predictor(context_feats, n_mask_tokens=T_mask)  # [B, T_mask, embed_dim]

        # L2 loss on normalized features (per V-JEPA paper)
        pred_norm = F.normalize(pred, dim=-1)
        tgt_norm = F.normalize(target_masked, dim=-1)
        loss = (pred_norm - tgt_norm).pow(2).mean()

        return loss


# ---------------------------------------------------------------------------
# Classification head for Stage 2b
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Lightweight head for downstream binary classification tasks."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """
        frame_feats: [B, T, embed_dim]
        Returns: [B] logits
        """
        pooled = frame_feats.mean(dim=1)  # [B, embed_dim]
        return self.mlp(pooled).squeeze(-1)  # [B]
