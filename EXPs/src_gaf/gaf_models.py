# gaf_models.py
# CNN 2D leggera per classificazione su immagini GAF

import torch
import torch.nn as nn


class GAFNet(nn.Module):
    """
    CNN 2D per immagini GAF da EEG.

    Input:  (batch, n_channels, image_size, image_size)
    Output: (batch, n_classes)

    Progettata per essere leggera (pochi parametri)
    per evitare overfitting su dataset piccoli.
    """

    def __init__(self, n_channels=31, image_size=32, n_classes=2, dropout=0.5):
        super().__init__()

        # Blocco 1: fusione spaziale dei canali EEG
        self.spatial = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )

        # Blocco 2: pattern locali nella GAF
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2, 2),         # image_size/2
            nn.Dropout(dropout),
        )

        # Blocco 3: pattern più globali
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2, 2),         # image_size/4
            nn.Dropout(dropout),
        )

        # Pooling adattivo → dimensione fissa (4x4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classificatore compatto
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"GAFNet: {total:,} parametri | "
              f"n_channels={n_channels} | image_size={image_size} | n_classes={n_classes}")

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.spatial(x)        # (B, 16, H, W)
        x = self.conv1(x)          # (B, 32, H/2, W/2)
        x = self.conv2(x)          # (B, 64, H/4, W/4)
        x = self.adaptive_pool(x)  # (B, 64, 4, 4)
        return self.classifier(x)
