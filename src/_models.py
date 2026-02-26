"""
Modelli neurali per classificazione EEG
Versione FINALE con massima regolarizzazione anti-overfitting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEEGNet(nn.Module):
    """Versione leggera per test"""

    def __init__(self, n_channels=31, n_samples=2000, n_classes=2):
        super().__init__()

        print(f"⚙️ Inizializzazione SimpleEEGNet con: channels={n_channels}, samples={n_samples}")

        self.n_channels = n_channels
        self.n_samples = n_samples

        # Primo blocco convoluzionale
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.3),  # <-- AGGIUNTO dropout dopo ELU
            nn.AvgPool2d((1, 8))
        )

        # Secondo blocco convoluzionale
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.4),  # <-- AUMENTATO da 0.5? No, è nuovo
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)   # Dropout esistente
        )

        # Calcola dimensione dopo convoluzioni
        self.flatten_size = self._compute_flatten_size()
        print(f"✅ Flatten size calcolata: {self.flatten_size}")

        # Classificatore con più regolarizzazione
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.flatten_size),  # <-- AGGIUNTO BatchNorm
            nn.Dropout(0.3),                    # <-- AGGIUNTO dropout iniziale
            nn.Linear(self.flatten_size, 64),
            nn.ELU(),
            nn.BatchNorm1d(64),                  # <-- AGGIUNTO BatchNorm
            nn.Dropout(0.6),                      # <-- AUMENTATO da 0.5
            nn.Linear(64, 32),                    # <-- AGGIUNTO layer intermedio
            nn.ELU(),
            nn.BatchNorm1d(32),                   # <-- AGGIUNTO BatchNorm
            nn.Dropout(0.6),                      # <-- AGGIUNTO dropout
            nn.Linear(32, n_classes)
        )

    def _compute_flatten_size(self):
        """Calcola dimensione dopo convoluzioni"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.conv1(x)
            x = self.conv2(x)
            flatten_size = x.view(1, -1).size(1)
            return flatten_size

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)
        return self.classifier(x)


class EEGAttentionNet(nn.Module):
    def __init__(self, n_channels=31, n_samples=2000, n_classes=2):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.F1 = 8
        self.D  = 2
        self.F2 = 16

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1),
            nn.ELU(),
        )

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.F1, self.F1 * self.D, kernel_size=(n_channels, 1),
                      groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),   # unico dropout qui
        )

        # Separable convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, 16),
                      padding=(0, 8), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),   # unico dropout qui
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 64))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.F2 * 64, 64),
            nn.ELU(),
            nn.Dropout(0.25),   # unico dropout nel classificatore
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)
