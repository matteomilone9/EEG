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

        self.eeg         = eeg_df.values.astype(np.float32)  # (C, T)
        self.labels      = labels
        self.window_size = window_size
        self.gaf_type    = gaf_type
        self.image_size  = image_size
        self.target_size = target_size
        self.augment     = augment

        start, end = index_range

        # FIX: scarta finestre con label 2 (preparazione) o -1 (non assegnato)
        self.windows       = []
        skipped_prep       = 0
        skipped_unassigned = 0

        for i in range(start, end - window_size + 1, stride):
            center_label = self.labels[i + window_size // 2]
            if center_label == 2:
                skipped_prep += 1
            elif center_label == -1:
                skipped_unassigned += 1
            else:
                self.windows.append(i)

        n_valid = len(self.windows)
        print(f"EEGViTDataset ({gaf_type} {image_size}→{target_size}px): "
              f"{n_valid} finestre valide | range=[{start},{end}) | stride={stride} | "
              f"scartate prep={skipped_prep} | scartate non-assegnate={skipped_unassigned}")

        if n_valid > 0:
            centers      = [w + window_size // 2 for w in self.windows]
            win_labels   = self.labels[centers]
            vals, counts = np.unique(win_labels, return_counts=True)
            label_names  = {0: "Riposo", 1: "Motor Imagery"}
            for v, c in zip(vals, counts):
                print(f"  {label_names.get(int(v), str(v))}: {c} finestre ({100*c/n_valid:.1f}%)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start  = self.windows[idx]
        end    = start + self.window_size
        window = self.eeg[:, start:end].copy()  # (C, T)

        mu     = window.mean(axis=1, keepdims=True)
        sd     = window.std(axis=1, keepdims=True) + 1e-8
        window = (window - mu) / sd

        if self.augment:
            window += np.random.normal(0, 0.02, window.shape).astype(np.float32)

        C         = window.shape[0]
        gaf_stack = np.zeros((C, self.image_size, self.image_size), dtype=np.float32)
        for ch in range(C):
            gaf_stack[ch] = compute_gaf(window[ch], self.gaf_type, self.image_size)

        # Resize a target_size per il ViT
        gaf_tensor = torch.tensor(gaf_stack, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
        gaf_tensor = F.interpolate(
            gaf_tensor, size=(self.target_size, self.target_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)  # (C, target_size, target_size)

        label = self.labels[start + self.window_size // 2]

        return (
            gaf_tensor,
            torch.tensor(label, dtype=torch.long),
        )
