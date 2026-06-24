"""Stage 2a: V-JEPA SSL pre-training on CoronaryDominance fluoroscopy video.

Usage:
  python pretrain.py \
    --data_dir /tmp/coronary_data \
    --teacher_ckpt /tmp/medsam2_arcade_v2.pt \
    --output_dir /tmp/ssl_checkpoints \
    --gcs_bucket coronary-angio-v2
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data import CoronaryClipDataset
from model import VJEPA


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',     type=str, required=True)
    p.add_argument('--teacher_ckpt', type=str, required=True)
    p.add_argument('--output_dir',   type=str, default='ssl_checkpoints')
    p.add_argument('--epochs',       type=int, default=50)
    p.add_argument('--batch_size',   type=int, default=16)
    p.add_argument('--lr',           type=float, default=1.5e-4)
    p.add_argument('--warmup_epochs',type=int, default=10)
    p.add_argument('--clip_len',     type=int, default=8)
    p.add_argument('--stride',       type=int, default=2)
    p.add_argument('--gcs_bucket',   type=str, default='')
    p.add_argument('--resume',       type=str, default='')
    p.add_argument('--num_workers',  type=int, default=4)
    return p.parse_args()


def cosine_lr(optimizer, epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * max(epoch, 1) / warmup_epochs
    else:
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def upload_gcs(local_path, bucket, remote_path):
    try:
        from google.cloud import storage
        storage.Client().bucket(bucket).blob(remote_path).upload_from_filename(local_path)
        print(f'Uploaded to gs://{bucket}/{remote_path}')
    except Exception as e:
        print(f'GCS upload skipped: {e}')


def load_medsam2_encoder(ckpt_path: str, device):
    """Load MedSAM2 image encoder from fine-tuned checkpoint."""
    sys.path.insert(0, '/tmp/MedSAM2')
    try:
        from sam2.build_sam import build_sam2
        # Build base architecture, then load our weights
        model = build_sam2('sam2_hiera_t.yaml', ckpt_path, device=device)
        return model.image_encoder
    except Exception as e:
        raise RuntimeError(
            f'Could not load MedSAM2 encoder from {ckpt_path}.\n'
            f'Ensure MedSAM2 is cloned at /tmp/MedSAM2 and installed.\n'
            f'Error: {e}'
        )


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data
    train_ds = CoronaryClipDataset(args.data_dir, clip_len=args.clip_len,
                                   stride=args.stride, split='train')
    val_ds   = CoronaryClipDataset(args.data_dir, clip_len=args.clip_len,
                                   stride=args.stride, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f'Train clips: {len(train_ds):,}  Val clips: {len(val_ds):,}')

    # Model
    image_encoder = load_medsam2_encoder(args.teacher_ckpt, device)
    model = VJEPA(sam2_image_encoder=image_encoder).to(device)
    print(f'Embed dim: {model.embed_dim}')

    # Only train context encoder + predictor; target encoder is EMA (no grad)
    trainable = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    scaler = GradScaler()

    start_epoch = 0
    best_val_loss = float('inf')
    total_steps = args.epochs * len(train_loader)
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        global_step = start_epoch * len(train_loader)
        print(f'Resumed from epoch {start_epoch}')

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(optimizer, epoch, args.warmup_epochs, args.epochs, args.lr)

        # Train
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for clips in train_loader:
            clips = clips.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = model(clips)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            model.update_ema(global_step, total_steps)
            global_step += 1
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clips in val_loader:
                clips = clips.to(device, non_blocking=True)
                with autocast():
                    loss = model(clips)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1:03d}/{args.epochs} | LR {lr:.2e} | '
              f'Train {train_loss:.4f} | Val {val_loss:.4f} | {time.time()-t0:.0f}s')

        # Checkpoint
        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    val_loss=val_loss, best_val_loss=best_val_loss)
        torch.save(ckpt, os.path.join(args.output_dir, 'last.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'best_encoder.pt')
            torch.save(dict(
                context_encoder=model.context_encoder.state_dict(),
                target_encoder=model.target_encoder.state_dict(),
                embed_dim=model.embed_dim,
                epoch=epoch,
                val_loss=val_loss,
            ), best_path)
            print(f'  -> New best val loss: {val_loss:.4f}')
            if args.gcs_bucket:
                upload_gcs(best_path, args.gcs_bucket, 'ssl/best_encoder.pt')

    print('Pre-training complete.')
    if args.gcs_bucket:
        upload_gcs(os.path.join(args.output_dir, 'last.pt'),
                   args.gcs_bucket, 'ssl/last.pt')


if __name__ == '__main__':
    main()
