# augmentation_kd.py — S&R (paper-style offline + online), Mixup, EEGAug, MMDataset
# ============================================================
# SR offline (paper): espande il dataset PRIMA del training, raddoppia i trial.
# SR online:          applicata nel trainer a livello di batch (comportamento precedente).
# Mixup:              applicata nel trainer a livello di batch.
# Flags in cfg:
#   use_sr      → True/False  (abilita SR)
#   sr_mode     → "offline" | "online"  (default "offline" = paper-style)
#   sr_prob     → probabilità per SR online (ignorato in offline)
#   n_segments  → numero di segmenti SR (default 8 come nel paper)
#   use_mixup   → True/False  (abilita Mixup nel trainer)
#   mixup_prob  → probabilità per Mixup
#   mixup_alpha → alpha per distribuzione Beta
# ============================================================

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ── SR offline (paper-style) ─────────────────────────────────────────────────
def segment_and_reconstruct_offline(
    X: np.ndarray,
    y: np.ndarray,
    n_segments: int = 8,
    multiplier: int = 1,
) -> tuple:
    """
    SR paper-style: genera `multiplier` trial sintetici PER OGNI trial originale,
    ricombinando segmenti di trial della stessa classe.
    Con multiplier=1 raddoppia il dataset (288 → 576), esattamente come nel paper.

    Args:
        X:           (B, C, T) float32
        y:           (B,)      int64
        n_segments:  numero di segmenti in cui dividere ogni trial (paper usa 8)
        multiplier:  quante copie sintetiche per trial (1 → ×2 totale)

    Returns:
        X_aug, y_aug  — solo i trial SINTETICI (da concatenare con gli originali)
    """
    B, C, T = X.shape
    seg_len = T // n_segments
    classes = np.unique(y)

    # indici per classe
    cls_idx = {c: np.where(y == c)[0] for c in classes}

    synth_X, synth_y = [], []
    for i in range(B):
        label_i = y[i]
        candidates = cls_idx[label_i]
        if len(candidates) < n_segments:
            continue
        for _ in range(multiplier):
            donors = np.random.choice(candidates, size=n_segments, replace=True)
            new_trial = X[i].copy()  # (C, T)
            for seg_idx, donor in enumerate(donors):
                start = seg_idx * seg_len
                end = start + seg_len if seg_idx < n_segments - 1 else T
                new_trial[:, start:end] = X[donor, :, start:end]
            synth_X.append(new_trial)
            synth_y.append(label_i)

    if not synth_X:
        return np.empty((0, C, T), dtype=X.dtype), np.empty((0,), dtype=y.dtype)

    return np.stack(synth_X).astype(np.float32), np.array(synth_y, dtype=y.dtype)


def apply_sr_offline(X: np.ndarray, y: np.ndarray, cfg: dict):
    """
    Wrapper: se use_sr=True e sr_mode="offline" espande X, y con trial sintetici.
    Restituisce (X_expanded, y_expanded).
    """
    if not cfg.get("use_sr", False):
        return X, y
    if cfg.get("sr_mode", "offline") != "offline":
        return X, y

    n_segments = cfg.get("n_segments", 8)
    multiplier = cfg.get("sr_multiplier", 1)
    X_syn, y_syn = segment_and_reconstruct_offline(X, y, n_segments, multiplier)
    if len(X_syn) == 0:
        return X, y
    X_out = np.concatenate([X, X_syn], axis=0)
    y_out = np.concatenate([y, y_syn], axis=0)
    # shuffle
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ── SR online (batch-level, comportamento precedente) ────────────────────────
def segment_and_reconstruct(x_t, y, n_segments=8, sr_prob=0.5):
    """SR online: applicata su un batch torch durante il training."""
    B, C, T = x_t.shape
    x_aug = x_t.clone()
    seg_len = T // n_segments
    for i in range(B):
        if random.random() > sr_prob:
            continue
        label_i = y[i].item()
        same_class = [j for j in range(B) if y[j].item() == label_i and j != i]
        if len(same_class) < n_segments:
            continue
        donors = random.sample(same_class, n_segments)
        for seg_idx in range(n_segments):
            start = seg_idx * seg_len
            end = start + seg_len if seg_idx < n_segments - 1 else T
            x_aug[i, :, start:end] = x_t[donors[seg_idx], :, start:end]
    return x_aug


# ── Mixup (batch-level) ───────────────────────────────────────────────────────
def mixup_batch(x_t, y, n_classes, mixup_prob=0.5, alpha=0.4):
    B = x_t.shape[0]
    y_oh = F.one_hot(y, n_classes).float()
    x_mix = x_t.clone()
    y_soft = y_oh.clone()
    for i in range(B):
        if random.random() > mixup_prob:
            continue
        label_i = y[i].item()
        same_class = [j for j in range(B) if y[j].item() == label_i and j != i]
        if not same_class:
            continue
        j = random.choice(same_class)
        lam = float(np.random.beta(alpha, alpha))
        lam = max(lam, 1 - lam)
        x_mix[i] = lam * x_t[i] + (1 - lam) * x_t[j]
        y_soft[i] = lam * y_oh[i] + (1 - lam) * y_oh[j]
    return x_mix, y_soft


# ── EEGAug (sample-level, usato in MMDataset) ─────────────────────────────────
class EEGAug:
    @staticmethod
    def noise(x):
        return x + torch.randn_like(x) * 0.03 * x.std()

    @staticmethod
    def shift(x):
        return torch.roll(x, torch.randint(-12, 13, (1,)).item(), dims=-1)

    @staticmethod
    def scale(x):
        return x * torch.FloatTensor(1).uniform_(0.85, 1.15)


# ── MMDataset ─────────────────────────────────────────────────────────────────
class MMDataset(Dataset):
    """
    Dataset multimodale EEG + GAF.
    L'augmentation sample-level (noise/shift/scale) è opzionale e controllata da aug_prob.
    SR offline e Mixup vengono gestiti FUORI dal dataset (nel pipeline/trainer).
    """
    def __init__(self, X_t, X_g, y, augment=False, aug_prob=0.5):
        self.X_t = torch.tensor(X_t)
        self.X_g = torch.tensor(X_g)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x_t = self.X_t[i].clone()
        x_g = self.X_g[i].clone()
        if self.augment:
            p = self.aug_prob
            if torch.rand(1) < p:
                x_t = EEGAug.noise(x_t)
            if torch.rand(1) < p:
                x_t = EEGAug.shift(x_t)
            if torch.rand(1) < p:
                x_t = EEGAug.scale(x_t)
        return {"eeg": x_t, "gaf": x_g, "label": self.y[i]}


def make_dummy_gaf(n: int) -> np.ndarray:
    return np.zeros((n, 1, 1, 1), dtype=np.float32)
