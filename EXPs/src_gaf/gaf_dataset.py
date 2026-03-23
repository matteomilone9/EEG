# gaf_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_gaf(signal, gaf_type="GASF", image_size=32):
    T = len(signal)
    if T > image_size:
        indices = np.linspace(0, T - 1, image_size).astype(int)
        signal  = signal[indices]

    s_min, s_max = signal.min(), signal.max()
    denom = s_max - s_min
    if denom < 1e-8:
        signal = np.zeros_like(signal)
    else:
        signal = 2 * (signal - s_min) / denom - 1.0
    signal = np.clip(signal, -1.0, 1.0)

    phi   = np.arccos(signal)
    phi_i = phi[:, np.newaxis]
    phi_j = phi[np.newaxis, :]

    if gaf_type == "GASF":
        gaf = np.cos(phi_i + phi_j)
    else:
        gaf = np.sin(phi_i - phi_j)

    return gaf.astype(np.float32)


class EEGGAFDataset(Dataset):
    def __init__(self, eeg_df, labels, index_range,
                 window_size=500, stride=250,
                 gaf_type="GASF", image_size=32,
                 augment=False):

        self.eeg         = eeg_df.values.astype(np.float32)  # (C, T)
        self.labels      = labels
        self.window_size = window_size
        self.gaf_type    = gaf_type
        self.image_size  = image_size
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
        print(f"EEGGAFDataset ({gaf_type} {image_size}x{image_size}): "
              f"{n_valid} finestre valide | range=[{start},{end}) | stride={stride} | "
              f"scartate prep={skipped_prep} | scartate non-assegnate={skipped_unassigned}")

        # Bilanciamento classi
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

        # z-score per canale
        mu     = window.mean(axis=1, keepdims=True)
        sd     = window.std(axis=1, keepdims=True) + 1e-8
        window = (window - mu) / sd

        if self.augment:
            window += np.random.normal(0, 0.02, window.shape).astype(np.float32)

        # GAF per ogni canale → (C, image_size, image_size)
        C         = window.shape[0]
        gaf_stack = np.zeros((C, self.image_size, self.image_size), dtype=np.float32)
        for ch in range(C):
            gaf_stack[ch] = compute_gaf(window[ch], self.gaf_type, self.image_size)

        label = self.labels[start + self.window_size // 2]

        return (
            torch.tensor(gaf_stack, dtype=torch.float32),
            torch.tensor(label,     dtype=torch.long),
        )
