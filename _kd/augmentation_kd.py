# augmentation_kd.py — Copia di augmentation.py per la pipeline _kd
# ============================================================

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def segment_and_reconstruct(xt, y, n_segments=10, sr_prob=0.5):
    B, C, T = xt.shape
    x_aug = xt.clone()
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
            x_aug[i, :, start:end] = xt[donors[seg_idx], :, start:end]
    return x_aug


def mixup_batch(xt, y, n_classes, mixup_prob=0.5, alpha=0.4):
    B = xt.shape[0]
    y_oh = F.one_hot(y, n_classes).float()
    x_mix = xt.clone()
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
        x_mix[i] = lam * xt[i] + (1 - lam) * xt[j]
        y_soft[i] = lam * y_oh[i] + (1 - lam) * y_oh[j]
    return x_mix, y_soft


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


class MMDataset(Dataset):
    def __init__(self, Xt, Xg, y, augment=False, aug_prob=0.5):
        self.Xt = torch.tensor(Xt)
        self.Xg = torch.tensor(Xg)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        xt = self.Xt[i].clone()
        xg = self.Xg[i].clone()
        if self.augment:
            p = self.aug_prob
            if torch.rand(1) < p:
                xt = EEGAug.noise(xt)
            if torch.rand(1) < p:
                xt = EEGAug.shift(xt)
            if torch.rand(1) < p:
                xt = EEGAug.scale(xt)
        return {"eeg": xt, "gaf": xg, "label": self.y[i]}


def make_dummy_gaf(n: int) -> np.ndarray:
    return np.zeros((n, 1, 1, 1), dtype=np.float32)
