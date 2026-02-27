# vit_models.py
# Wrapper ViT pretrainato (timm) per input EEG multi-canale

import torch
import torch.nn as nn
import timm


class EEGViT(nn.Module):
    """
    ViT pretrainato adattato per input EEG multi-canale.

    Pipeline:
        Input (B, n_channels, target_size, target_size)
            ↓ channel_proj: Conv2d(n_channels, 3, 1x1)  → (B, 3, target_size, target_size)
            ↓ ViT pretrainato (timm)                     → (B, embed_dim)
            ↓ MLP head                                   → (B, n_classes)

    Args:
        n_channels:  numero canali EEG (es. 31)
        n_classes:   numero classi (es. 2)
        model_name:  "vit_small_patch16_224" o "vit_base_patch16_224"
        pretrained:  usa pesi ImageNet (consigliato)
        dropout:     dropout nel classifier head
        freeze_vit:  congela i pesi del ViT (utile con pochissimi dati)
        target_size: dimensione immagine input (deve coincidere con image_size nel config)
    """

    SUPPORTED_MODELS = [
        "vit_small_patch16_224",
        "vit_base_patch16_224",
        "deit_tiny_patch16_224",
    ]

    def __init__(self, n_channels=31, n_classes=2,
                 model_name="vit_small_patch16_224",
                 pretrained=True, dropout=0.3,
                 freeze_vit=False, target_size=64):
        super().__init__()

        assert model_name in self.SUPPORTED_MODELS, \
            f"model_name deve essere uno di: {self.SUPPORTED_MODELS}"

        # 1) Proiezione canali EEG → 3 (RGB-like)
        self.channel_proj = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
        )

        # 2) ViT pretrainato — rimuoviamo la testa originale
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=target_size,   # ← override dimensione input
        )

        # Congela il ViT se richiesto (utile con dataset piccoli)
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT congelato — si allena solo channel_proj + classifier")

        # 3) Dimensione embedding del ViT scelto
        embed_dim = self.vit.embed_dim  # 384 per small, 768 per base

        # 4) Classification head custom
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EEGViT ({model_name}): {total:,} params totali | "
              f"{trainable:,} trainabili | embed_dim={embed_dim} | img_size={target_size}")

    def forward(self, x):
        # x: (B, C, target_size, target_size)
        x = self.channel_proj(x)   # (B, 3, target_size, target_size)
        x = self.vit(x)            # (B, embed_dim)
        return self.classifier(x)  # (B, n_classes)
