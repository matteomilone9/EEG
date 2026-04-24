# losses_kd.py — Contrastive Distillation Loss (EEG ↔ GAF) + Adaptive Alpha KD
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
        self.symmetric = symmetric

    def forward(self, z_eeg: torch.Tensor, z_gaf: torch.Tensor) -> torch.Tensor:
        """
        z_eeg: (B, D) — embedding EEG proiettati e normalizzati L2
        z_gaf: (B, D) — embedding GAF proiettati e normalizzati L2
        """
        B = z_eeg.size(0)
        z_eeg = F.normalize(z_eeg, dim=-1)
        z_gaf = F.normalize(z_gaf, dim=-1)

        sim = z_eeg @ z_gaf.T / self.temperature
        targets = torch.arange(B, device=z_eeg.device)

        loss_eeg2gaf = F.cross_entropy(sim, targets)
        if self.symmetric:
            loss_gaf2eeg = F.cross_entropy(sim.T, targets)
            return (loss_eeg2gaf + loss_gaf2eeg) * 0.5
        return loss_eeg2gaf


class AdaptiveKDLoss(nn.Module):
    """
    KD loss con alpha adattivo basato sulla confidenza del teacher.

    alpha_eff(x) = alpha_base * clamp(conf(x) / threshold, 0, 1)

    Quando conf >= threshold → alpha_eff = alpha_base (distillazione piena)
    Quando conf < threshold  → alpha_eff si riduce linearmente fino a 0

    Riferimento: "Do Not Blindly Imitate the Teacher" (ICLR 2024),
    adattato a per-sample confidence gating.
    """

    def __init__(self,
                 alpha_base: float = 0.5,
                 temperature: float = 4.0,
                 threshold: float = 0.5,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha_base = alpha_base
        self.temperature = temperature
        self.threshold = threshold
        self.reduction = reduction

    def forward(self,
                logits_student: torch.Tensor,
                logits_teacher: torch.Tensor,
                labels: torch.Tensor) -> tuple:
        """
        Returns: (loss_total, alpha_mean)
        """
        loss_ce = F.cross_entropy(logits_student, labels)

        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=-1)
            conf = probs_teacher.max(dim=-1).values
            alpha_eff = self.alpha_base * (conf / self.threshold).clamp(0.0, 1.0)

        T = self.temperature
        log_p_s = F.log_softmax(logits_student / T, dim=-1)
        p_t = F.softmax(logits_teacher / T, dim=-1).detach()

        kd_per_sample = F.kl_div(log_p_s, p_t, reduction='none').sum(dim=-1) * (T ** 2)
        loss_kd = alpha_eff * kd_per_sample

        if self.reduction == 'mean':
            loss_kd = loss_kd.mean()
        else:
            loss_kd = loss_kd.sum()

        loss_total = (1.0 - self.alpha_base) * loss_ce + loss_kd
        return loss_total, alpha_eff.mean().detach()
