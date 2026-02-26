"""
Dataset class per PyTorch
"""
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, señales, labels, indices, window_size=2000):
        # Señales è un DataFrame? Un numpy array?
        # print(f"🔍 Tipo señales: {type(señales)}")
        # print(f"🔍 Shape señales: {señales.shape}")
        # print(f"🔍 Range valori: [{señales.values.min():.4f}, {señales.values.max():.4f}]")
        # print(f"🔍 Media: {señales.values.mean():.4f}, Std: {señales.values.std():.4f}")

        # NON normalizzare qui se i dati sono già filtrati!
        self.eeg = señales.values  # Mantieni i valori originali

        self.labels = labels
        self.indices = indices
        self.window_size = window_size

        # Filtra indici validi
        self.valid_indices = [i for i in indices
                              if i + window_size <= señales.shape[1]]

        print(f"✅ Dataset creato con {len(self.valid_indices)} campioni validi")

    def __len__(self):
        return len(self.valid_indices)


    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.window_size

        window = self.eeg[:, start:end]
        # print(f"📊 Dataset - window shape: {window.shape}, start={start}, end={end}")

        label = self.labels[end - 1]

        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )