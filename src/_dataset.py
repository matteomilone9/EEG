import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, eeg_df, labels, index_range, window_size=2000, stride=None):
        self.eeg = eeg_df.values          # shape: (n_channels, n_samples)
        self.labels = labels
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size  # default: no overlap

        start_idx, end_idx = index_range
        # genera finestre con stride, senza sforare i limiti
        self.windows = [
            i for i in range(start_idx, end_idx - window_size + 1, self.stride)
        ]
        print(f"Dataset: {len(self.windows)} finestre | stride={self.stride} | range=[{start_idx}, {end_idx}]")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        end = start + self.window_size
        window = self.eeg[:, start:end].copy()

        # z-score per canale — normalizza ogni canale indipendentemente ~ da valutare la norm
        mean = window.mean(axis=1, keepdims=True)
        std = window.std(axis=1, keepdims=True) + 1e-8
        window = (window - mean) / std

        label = self.labels[start + self.window_size // 2]
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )
