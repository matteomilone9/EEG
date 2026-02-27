# gaf_dataset.py
# EEGDataset che produce Gramian Angular Fields (GASF o GADF)

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_gaf(signal, gaf_type="GASF", image_size=32):
    """
    Converte una serie temporale 1D in una GAF image.

    Args:
        signal:     array 1D (T,)
        gaf_type:   "GASF" (somma) o "GADF" (differenza)
        image_size: dimensione output (image_size x image_size)
                    se < len(signal), fa downsampling con interpolazione

    Returns:
        gaf: array 2D (image_size, image_size) float32
    """
    # 1) Downsampling se necessario (media su segmenti)
    T = len(signal)
    if T > image_size:
        # riduce a image_size punti con media mobile
        indices = np.linspace(0, T - 1, image_size).astype(int)
        signal  = signal[indices]

    # 2) Normalizza in [-1, 1]
    s_min, s_max = signal.min(), signal.max()
    denom = s_max - s_min
    if denom < 1e-8:
        signal = np.zeros_like(signal)
    else:
        signal = 2 * (signal - s_min) / denom - 1.0
    signal = np.clip(signal, -1.0, 1.0)

    # 3) Coordinate polari: phi = arccos(x)
    phi = np.arccos(signal)

    # 4) Gramian matrix
    phi_i = phi[:, np.newaxis]  # (N, 1)
    phi_j = phi[np.newaxis, :]  # (1, N)

    if gaf_type == "GASF":
        gaf = np.cos(phi_i + phi_j)   # cos(phi_i + phi_j)
    else:  # GADF
        gaf = np.sin(phi_i - phi_j)   # sin(phi_i - phi_j)

    return gaf.astype(np.float32)


class EEGGAFDataset(Dataset):
    def __init__(self, eeg_df, labels, index_range,
                 window_size=500, stride=250,
                 gaf_type="GASF", image_size=32,
                 augment=False):
        """
        Args:
            eeg_df:      DataFrame EEG (n_channels, n_samples)
            labels:      array (n_samples,)
            index_range: tupla (start, end) in campioni
            window_size: finestra temporale in campioni
            stride:      stride in campioni
            gaf_type:    "GASF" o "GADF"
            image_size:  dimensione immagine GAF (image_size x image_size)
            augment:     rumore leggero solo in training
        """
        self.eeg         = eeg_df.values.astype(np.float32)  # (C, T)
        self.labels      = labels
        self.window_size = window_size
        self.gaf_type    = gaf_type
        self.image_size  = image_size
        self.augment     = augment

        start, end = index_range
        self.windows = list(range(start, end - window_size + 1, stride))

        print(f"EEGGAFDataset ({gaf_type} {image_size}x{image_size}): "
              f"{len(self.windows)} finestre | range=[{start},{end}) | stride={stride}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start  = self.windows[idx]
        end    = start + self.window_size
        window = self.eeg[:, start:end].copy()   # (C, T)

        # z-score per canale
        mu  = window.mean(axis=1, keepdims=True)
        sd  = window.std(axis=1, keepdims=True) + 1e-8
        window = (window - mu) / sd

        # augmentation leggera solo in training
        if self.augment:
            window += np.random.normal(0, 0.02, window.shape).astype(np.float32)

        # GAF per ogni canale → (C, image_size, image_size)
        C = window.shape[0]
        gaf_stack = np.zeros((C, self.image_size, self.image_size), dtype=np.float32)
        for ch in range(C):
            gaf_stack[ch] = compute_gaf(window[ch], self.gaf_type, self.image_size)

        label = self.labels[start + self.window_size // 2]

        return (
            torch.tensor(gaf_stack, dtype=torch.float32),
            torch.tensor(label,     dtype=torch.long),
        )
