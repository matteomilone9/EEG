# vit_dataset.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src_gaf.gaf_dataset import compute_gaf


class EEGViTDataset(Dataset):
    def __init__(self, eeg_df, labels, index_range,
                 window_size=500, stride=250,
                 gaf_type="GADF", image_size=32,
                 target_size=224, augment=False):
        """
        Args:
            eeg_df:      DataFrame EEG (n_channels x n_samples)
            labels:      array (n_samples,)
            index_range: tupla (start, end)
            window_size: finestra temporale in campioni
            stride:      stride in campioni
            gaf_type:    "GASF" o "GADF"
            image_size:  dimensione GAF intermedia (es. 32)
            target_size: dimensione finale per il ViT (224)
            augment:     rumore leggero solo in training
        """
        self.eeg         = eeg_df.values.astype(np.float32)  # (C, T)
        self.labels      = labels
        self.window_size = window_size
        self.gaf_type    = gaf_type
        self.image_size  = image_size
        self.target_size = target_size
        self.augment     = augment

        start, end = index_range
        self.windows = list(range(start, end - window_size + 1, stride))

        print(f"EEGViTDataset ({gaf_type} {image_size}→{target_size}px): "
              f"{len(self.windows)} finestre | range=[{start},{end}) | stride={stride}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start  = self.windows[idx]
        end    = start + self.window_size
        window = self.eeg[:, start:end].copy()   # (C, T)

        # z-score per canale
        mu     = window.mean(axis=1, keepdims=True)
        sd     = window.std(axis=1, keepdims=True) + 1e-8
        window = (window - mu) / sd

        # augmentation leggera solo in training
        if self.augment:
            window += np.random.normal(0, 0.02, window.shape).astype(np.float32)

        # GAF per ogni canale → (C, image_size, image_size)
        C = window.shape[0]
        gaf_stack = np.zeros((C, self.image_size, self.image_size), dtype=np.float32)
        for ch in range(C):
            gaf_stack[ch] = compute_gaf(window[ch], self.gaf_type, self.image_size)

        # Tensor (C, H, W) → resize a (C, target_size, target_size)
        gaf_tensor = torch.tensor(gaf_stack, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
        gaf_tensor = F.interpolate(
            gaf_tensor, size=(self.target_size, self.target_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)  # (C, target_size, target_size)

        label = self.labels[start + self.window_size // 2]

        return (
            gaf_tensor,
            torch.tensor(label, dtype=torch.long),
        )
