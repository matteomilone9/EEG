# models_spectrogram.py
# CNN 2D leggera per classificazione su spettrogrammi EEG

import torch
import torch.nn as nn


class SpectrogramNet(nn.Module):
    """
    CNN 2D per spettrogrammi EEG.

    Input:  (batch, n_channels, freq_bins, time_steps)
    Output: (batch, n_classes)

    Con window_size=500, nperseg=64, noverlap=48, banda 7-40Hz:
      freq_bins ≈ 23, time_steps ≈ 14
    """

    def __init__(self, n_channels=31, n_classes=2, dropout=0.25):
        super().__init__()

        # Blocco 1: convoluzione "spaziale" (sui canali)
        # kernel (n_channels, 1) → fonde i canali EEG
        self.spatial = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )

        # Blocco 2: convoluzione tempo-frequenza
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(dropout),
        )

        # Blocco 3: approfondimento
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Pooling adattivo → dimensione fissa indipendentemente da freq/time
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classificatore
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        # stampa info a init
        total = sum(p.numel() for p in self.parameters())
        print(f"SpectrogramNet: {total:,} parametri | "
              f"n_channels={n_channels} | n_classes={n_classes}")

    def forward(self, x):
        # x: (B, C, F, T)
        x = self.spatial(x)      # (B, 32, F, T)
        x = self.conv1(x)        # (B, 64, F/2, T/2)
        x = self.conv2(x)        # (B, 128, F/2, T/2)
        x = self.adaptive_pool(x)  # (B, 128, 4, 4)
        return self.classifier(x)
