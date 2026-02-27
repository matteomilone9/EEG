# dataset_spectrogram.py
# EEGDataset che produce spettrogrammi (n_channels, freq_bins, time_steps)

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import stft


class EEGSpectrogramDataset(Dataset):
    def __init__(self, eeg_df, labels, index_range, window_size=500,
                 stride=250, fs=500, augment=False):
        """
        Args:
            eeg_df:      DataFrame EEG (n_channels, n_samples)
            labels:      array (n_samples,)
            index_range: tupla (start, end) in campioni
            window_size: finestra temporale in campioni
            stride:      stride in campioni
            fs:          frequenza di campionamento
            augment:     se True, aggiunge rumore leggero (solo training)
        """
        self.eeg        = eeg_df.values.astype(np.float32)  # (C, T)
        self.labels     = labels
        self.window_size = window_size
        self.fs         = fs
        self.augment    = augment

        start, end = index_range
        self.windows = list(range(start, end - window_size + 1, stride))
        print(f"SpectrogramDataset: {len(self.windows)} finestre | "
              f"range=[{start},{end}) | stride={stride}")

    def __len__(self):
        return len(self.windows)

    def _to_spectrogram(self, window):
        """
        window: (C, T) numpy float32
        returns: (C, freq_bins, time_steps) numpy float32

        Usa STFT con:
          - nperseg=64  → risoluzione freq: 64/500 = ~8Hz per bin
          - noverlap=48 → overlap 75%
          - nfft=128    → freq_bins = 65 (0–250 Hz)
        Poi seleziona solo le freq rilevanti (7–40 Hz = banda mu/beta)
        """
        C = window.shape[0]
        specs = []

        for ch in range(C):
            sig = window[ch]

            # z-score per canale
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)

            # STFT
            f, t, Zxx = stft(sig, fs=self.fs, nperseg=64, noverlap=48, nfft=128)

            # magnitudine in dB
            mag = np.abs(Zxx)
            mag = np.log1p(mag)  # log1p più stabile di log su dati piccoli

            # seleziona solo banda mu/beta: 7–40 Hz
            freq_mask = (f >= 7) & (f <= 40)
            mag = mag[freq_mask, :]  # (freq_bins_filtered, time_steps)

            specs.append(mag)

        spec = np.stack(specs, axis=0)  # (C, freq_bins, time_steps)
        return spec.astype(np.float32)

    def __getitem__(self, idx):
        start  = self.windows[idx]
        end    = start + self.window_size
        window = self.eeg[:, start:end].copy()

        # augmentation leggera solo in training
        if self.augment:
            window += np.random.normal(0, 0.02, window.shape).astype(np.float32)

        spec  = self._to_spectrogram(window)   # (C, F, T)
        label = self.labels[start + self.window_size // 2]

        return (
            torch.tensor(spec,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )
