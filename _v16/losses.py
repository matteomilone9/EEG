# losses.py — Contrastive Distillation Loss (EEG ↔ GAF)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveDistillLoss(nn.Module):
    """
    Contrastive Distillation tra embedding EEG e embedding GAF.

    Per ogni trial nel batch:
      - Il positivo è l'embedding GAF dello stesso trial
      - I negativi sono tutti gli altri embedding GAF del batch

    Usa InfoNCE (NT-Xent) con temperatura configurabile.
    Funziona sia in modalità simmetrica (EEG→GAF + GAF→EEG)
    che unidirezionale (solo EEG→GAF).

    Riferimento: Chen et al. SimCLR 2020, adattato a cross-modal.
    """

    def __init__(self, temperature: float = 0.07, symmetric: bool = True):
        super().__init__()
        self.temperature = temperature
        self.symmetric   = symmetric

    def forward(self,
                z_eeg: torch.Tensor,
                z_gaf: torch.Tensor) -> torch.Tensor:
        """
        z_eeg: (B, D) — embedding EEG proiettati e normalizzati L2
        z_gaf: (B, D) — embedding GAF proiettati e normalizzati L2
        """
        B = z_eeg.size(0)
        z_eeg = F.normalize(z_eeg, dim=-1)
        z_gaf = F.normalize(z_gaf, dim=-1)

        # Matrice di similarità (B, B)
        sim = z_eeg @ z_gaf.T / self.temperature

        # Target: ogni EEG si abbina al GAF dello stesso trial (diagonale)
        targets = torch.arange(B, device=z_eeg.device)

        loss_eeg2gaf = F.cross_entropy(sim,   targets)  # EEG → GAF
        if self.symmetric:
            loss_gaf2eeg = F.cross_entropy(sim.T, targets)  # GAF → EEG
            return (loss_eeg2gaf + loss_gaf2eeg) * 0.5
        return loss_eeg2gaf