# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np


class EEGDataset(Dataset):
    def __init__(self, eeg_df, labels, index_range, window_size=500, stride=None):
        self.eeg         = eeg_df.values  # shape: (n_channels, n_samples)
        self.labels      = labels
        self.window_size = window_size
        self.stride      = stride if stride is not None else window_size

        start_idx, end_idx = index_range

        # Genera finestre valide:
        # - non sforano i limiti
        # - il campione centrale NON è preparazione (label=2)
        # - il campione centrale NON è non assegnato (label=-1)
        self.windows = []
        skipped_prep   = 0
        skipped_unassigned = 0

        for i in range(start_idx, end_idx - window_size + 1, self.stride):
            center_label = self.labels[i + window_size // 2]
            if center_label == 2:
                skipped_prep += 1
            elif center_label == -1:
                skipped_unassigned += 1
            else:
                self.windows.append(i)

        n_valid = len(self.windows)
        print(
            f"Dataset [{start_idx}, {end_idx}] | "
            f"stride={self.stride} | "
            f"finestre valide={n_valid} | "
            f"scartate prep={skipped_prep} | "
            f"scartate non-assegnate={skipped_unassigned}"
        )

        # Stampa bilanciamento classi
        if n_valid > 0:
            centers      = [w + window_size // 2 for w in self.windows]
            win_labels   = self.labels[centers]
            vals, counts = np.unique(win_labels, return_counts=True)
            label_names  = {0: "Riposo", 1: "Motor Imagery"}
            for v, c in zip(vals, counts):
                name = label_names.get(int(v), f"label={v}")
                print(f"  {name}: {c} finestre ({100 * c / n_valid:.1f}%)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start  = self.windows[idx]
        end    = start + self.window_size
        window = self.eeg[:, start:end].copy()

        # Z-score per canale — normalizza ogni canale indipendentemente
        mean   = window.mean(axis=1, keepdims=True)
        std    = window.std(axis=1, keepdims=True) + 1e-8
        window = (window - mean) / std

        label = self.labels[start + self.window_size // 2]
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label,  dtype=torch.long),
        )
