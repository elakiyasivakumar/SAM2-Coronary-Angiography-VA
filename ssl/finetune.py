"""Stage 2b: Supervised fine-tuning for downstream binary classification.

Encoder is frozen (from best_encoder.pt). Only the classification head is trained.

Usage:
  python finetune.py \
    --data_dir /tmp/coronary_data \
    --ssl_encoder_ckpt /tmp/ssl_checkpoints/best_encoder.pt \
    --task occlusion \
    --output_dir /tmp/ssl_checkpoints/downstream

Tasks: occlusion | acs | collaterals
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import CoronaryLabeledDataset
from model import ClassificationHead, FrameEncoder


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',          type=str, required=True)
    p.add_argument('--ssl_encoder_ckpt',  type=str, required=True)
    p.add_argument('--task',              type=str, default='occlusion',
                   choices=['occlusion', 'acs', 'collaterals'])
    p.add_argument('--output_dir',        type=str, default='finetune_checkpoints')
    p.add_argument('--epochs',            type=int, default=30)
    p.add_argument('--batch_size',        type=int, default=32)
    p.add_argument('--lr',                type=float, default=1e-3)
    p.add_argument('--clip_len',          type=int, default=8)
    p.add_argument('--clips_per_study',   type=int, default=4)
    p.add_argument('--num_workers',       type=int, default=4)
    return p.parse_args()


def build_encoder_from_ssl_ckpt(ckpt_path: str, device):
    """Reconstruct FrameEncoder from SSL checkpoint (no SAM2 dependency needed)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    # The target encoder (EMA) is more stable than context encoder
    state = ckpt['target_encoder']
    embed_dim = ckpt['embed_dim']

    encoder = FrameEncoder.__new__(FrameEncoder)
    nn.Module.__init__(encoder)
    encoder.load_state_dict(state)
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder, embed_dim


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Frozen encoder
    encoder, embed_dim = build_encoder_from_ssl_ckpt(args.ssl_encoder_ckpt, device)
    print(f'Encoder embed_dim: {embed_dim}  (frozen)')

    # Classification head
    head = ClassificationHead(embed_dim=embed_dim).to(device)

    # Data
    train_ds = CoronaryLabeledDataset(args.data_dir, task=args.task,
                                      clip_len=args.clip_len,
                                      clips_per_study=args.clips_per_study)
    val_ds   = CoronaryLabeledDataset(args.data_dir, task=args.task,
                                      clip_len=args.clip_len, clips_per_study=2)

    # Balanced sampler for imbalanced tasks
    labels = [s[2] for s in train_ds.samples]
    n_pos = max(sum(labels), 1)
    n_neg = max(len(labels) - n_pos, 1)
    sample_weights = [1.0 / n_pos if l == 1 else 1.0 / n_neg for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f'Task: {args.task} | Train: {len(train_ds)} clips | Val: {len(val_ds)} clips')
    print(f'Class balance — pos: {n_pos}  neg: {n_neg}  pos_weight: {n_neg/n_pos:.2f}')

    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    best_auc = 0.0

    for epoch in range(args.epochs):
        # Train
        head.train()
        train_loss = 0.0
        for clips, y in train_loader:
            clips = clips.to(device, non_blocking=True)
            y     = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                with torch.no_grad():
                    feats = encoder(clips)   # [B, T, embed_dim]
                logits = head(feats)          # [B]
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        # Validate
        head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for clips, y in val_loader:
                clips = clips.to(device)
                with autocast():
                    feats  = encoder(clips)
                    logits = head(feats)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(y.numpy())

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.5

        print(f'Epoch {epoch+1:03d}/{args.epochs} | Loss {train_loss:.4f} | AUC {auc:.4f}')

        if auc > best_auc:
            best_auc = auc
            torch.save(dict(head=head.state_dict(), auc=auc,
                            task=args.task, epoch=epoch),
                       os.path.join(args.output_dir, f'best_{args.task}.pt'))
            print(f'  -> New best AUC: {auc:.4f}')

    print(f'Done. Best AUC ({args.task}): {best_auc:.4f}')


if __name__ == '__main__':
    main()
