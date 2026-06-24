"""Evaluate downstream classification heads from Stage 2b.

Reports AUC-ROC + classification report for a given task.

Usage:
  python eval.py \
    --data_dir /tmp/coronary_data \
    --ssl_encoder_ckpt /tmp/ssl_checkpoints/best_encoder.pt \
    --head_ckpt /tmp/ssl_checkpoints/downstream/best_occlusion.pt \
    --task occlusion
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from data import CoronaryLabeledDataset
from model import ClassificationHead, FrameEncoder


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',         type=str, required=True)
    p.add_argument('--ssl_encoder_ckpt', type=str, required=True)
    p.add_argument('--head_ckpt',        type=str, required=True)
    p.add_argument('--task',             type=str, default='occlusion',
                   choices=['occlusion', 'acs', 'collaterals'])
    p.add_argument('--clip_len',         type=int, default=8)
    p.add_argument('--clips_per_study',  type=int, default=8,
                   help='More clips per study → more stable per-study score via averaging')
    p.add_argument('--threshold',        type=float, default=0.5)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load encoder
    ssl_ckpt = torch.load(args.ssl_encoder_ckpt, map_location=device)
    encoder = FrameEncoder.__new__(FrameEncoder)
    nn.Module.__init__(encoder)
    encoder.load_state_dict(ssl_ckpt['target_encoder'])
    encoder = encoder.to(device).eval()
    embed_dim = ssl_ckpt['embed_dim']
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Load head
    head_ckpt = torch.load(args.head_ckpt, map_location=device)
    head = ClassificationHead(embed_dim=embed_dim).to(device).eval()
    head.load_state_dict(head_ckpt['head'])

    # Data
    ds = CoronaryLabeledDataset(args.data_dir, task=args.task,
                                clip_len=args.clip_len,
                                clips_per_study=args.clips_per_study)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, y in loader:
            clips = clips.to(device)
            with autocast():
                feats  = encoder(clips)
                logits = head(feats)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    preds_binary = (all_preds > args.threshold).astype(int)

    print(f'\n=== Evaluation: {args.task} ===')
    print(f'AUC-ROC:   {auc:.4f}')
    print(f'Threshold: {args.threshold}')
    print()
    print(classification_report(all_labels, preds_binary,
                                target_names=['normal', args.task]))

    # Youden's J optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]
    print(f"Youden's J optimal threshold: {best_thresh:.3f}")


if __name__ == '__main__':
    main()
