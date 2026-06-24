"""CoronaryDominance data loaders for V-JEPA pre-training (Stage 2a) and
supervised fine-tuning (Stage 2b).

Assumes data is stored as .npz files, one per sequence, with keys:
  frames: uint8 [N, H, W]
  is_artifact: bool
  is_occlusion: bool
  is_collaterals: bool
  is_acs: bool  (may not exist in all sequences)
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

TARGET_SIZE = 512
TASK_LABEL_KEYS = {
    'occlusion':   'is_occlusion',
    'acs':         'is_acs',
    'collaterals': 'is_collaterals',
}


def _load_clip(path: Path, start: int, clip_len: int, stride: int) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    frames = data['frames']  # [N, H, W] uint8
    indices = [min(start + i * stride, len(frames) - 1) for i in range(clip_len)]
    clip = frames[indices].astype(np.float32) / 255.0  # [T, H, W]
    if clip.shape[-1] != TARGET_SIZE or clip.shape[-2] != TARGET_SIZE:
        clip = np.stack([cv2.resize(f, (TARGET_SIZE, TARGET_SIZE)) for f in clip])
    # Grayscale -> 3-channel
    clip = np.stack([clip, clip, clip], axis=1)  # [T, 3, H, W]
    return clip


class CoronaryClipDataset(Dataset):
    """Unlabeled clip dataset for Stage 2a SSL pre-training.

    Produces clips of shape [T, 3, H, W] float32 in [0, 1].
    """

    def __init__(
        self,
        npz_dir: str,
        clip_len: int = 8,
        stride: int = 2,
        split: str = 'train',
        val_fraction: float = 0.1,
        seed: int = 42,
        filter_artifacts: bool = True,
    ):
        self.clip_len = clip_len
        self.stride = stride

        npz_dir = Path(npz_dir)
        all_files = sorted(npz_dir.glob('**/*.npz'))

        if filter_artifacts:
            filtered = []
            for f in all_files:
                try:
                    d = np.load(f, allow_pickle=True)
                    if not bool(d.get('is_artifact', False)):
                        filtered.append(f)
                except Exception:
                    pass
            all_files = filtered

        rng = random.Random(seed)
        all_files = list(all_files)
        rng.shuffle(all_files)
        n_val = max(1, int(len(all_files) * val_fraction))
        self.files = all_files[:n_val] if split == 'val' else all_files[n_val:]

        self.clips = []
        for f in self.files:
            try:
                d = np.load(f, allow_pickle=True)
                n_frames = d['frames'].shape[0]
                window = clip_len * stride
                hop = window // 2
                for start in range(0, max(1, n_frames - window + 1), hop):
                    self.clips.append((f, start))
            except Exception:
                pass

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        path, start = self.clips[idx]
        clip = _load_clip(path, start, self.clip_len, self.stride)
        return torch.from_numpy(clip)


class CoronaryLabeledDataset(Dataset):
    """Labeled dataset for Stage 2b downstream fine-tuning.

    Produces (clip [T, 3, H, W], label float32) pairs.
    Aggregates multiple clips per study for richer coverage.
    """

    def __init__(
        self,
        npz_dir: str,
        task: str = 'occlusion',
        clip_len: int = 8,
        stride: int = 2,
        clips_per_study: int = 4,
    ):
        assert task in TASK_LABEL_KEYS, f'Unknown task: {task}'
        self.clip_len = clip_len
        self.stride = stride
        label_key = TASK_LABEL_KEYS[task]

        npz_dir = Path(npz_dir)
        self.samples = []  # (path, start, label)

        for f in sorted(npz_dir.glob('**/*.npz')):
            try:
                d = np.load(f, allow_pickle=True)
                if label_key not in d:
                    continue
                label = int(bool(d[label_key]))
                n_frames = d['frames'].shape[0]
                max_start = max(0, n_frames - clip_len * stride)
                starts = np.linspace(0, max_start, clips_per_study, dtype=int)
                for start in starts:
                    self.samples.append((f, int(start), label))
            except Exception:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start, label = self.samples[idx]
        clip = _load_clip(path, start, self.clip_len, self.stride)
        return torch.from_numpy(clip), torch.tensor(label, dtype=torch.float32)
