# pretrain_gaf.py — Pretraining cross-subject del GAFEncoder
# Lancia UNA VOLTA, salva i pesi, poi il run principale li carica.
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from config import CFG, set_seed
from preprocessing import load_subject, preprocess_subject
from model import GAFMiniEncoder


PRETRAIN_SAVE_PATH = "gaf_encoder_pretrained.pt"
PRETRAIN_EPOCHS    = 300
PRETRAIN_LR        = 1e-3
PRETRAIN_BATCH     = 64
PRETRAIN_PATIENCE  = 50
PRETRAIN_SEED      = 42


def collect_all_gaf(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Raccoglie tutte le GAF di tutti i soggetti (train + test)."""
    all_gaf, all_y = [], []
    for s in range(1, cfg['n_subjects'] + 1):
        print(f"  Caricamento S{s:02d}...", end=" ")
        X_tr_raw, y_tr, X_te_raw, y_te = load_subject(s, cfg)

        # ── FIX: forza use_gaf=True per generare GAF reali ──────────
        cfg_gaf = {**cfg, 'use_gaf': True}
        _, _, X_tr_g, X_te_g = preprocess_subject(X_tr_raw, X_te_raw, cfg_gaf)

        all_gaf.extend([X_tr_g, X_te_g])
        all_y.extend([y_tr, y_te])
        print(f"✓ ({len(y_tr)+len(y_te)} trial)")
    return np.concatenate(all_gaf), np.concatenate(all_y)


def pretrain_gaf_encoder(cfg, save_path: str = PRETRAIN_SAVE_PATH):
    set_seed(PRETRAIN_SEED)
    device = cfg['device']

    print("\n" + "="*60)
    print(" GAF Encoder — Pretraining cross-subject")
    print("="*60)

    # ── Raccolta dati ─────────────────────────────────────────
    print("\n[1/3] Raccolta GAF da tutti i soggetti...")
    all_gaf, all_y = collect_all_gaf(cfg)
    print(f"\n  Totale: {len(all_y)} trial | shape GAF: {all_gaf.shape}")

    # ── Split train/val stratificato ─────────────────────────
    idx = np.arange(len(all_y))
    idx_tr, idx_va = train_test_split(idx, test_size=0.15,
                                      stratify=all_y, random_state=PRETRAIN_SEED)
    G_tr = torch.FloatTensor(all_gaf[idx_tr])
    G_va = torch.FloatTensor(all_gaf[idx_va])
    y_tr = torch.LongTensor(all_y[idx_tr])
    y_va = torch.LongTensor(all_y[idx_va])

    tr_ld = DataLoader(TensorDataset(G_tr, y_tr),
                       batch_size=PRETRAIN_BATCH, shuffle=True,  num_workers=0)
    va_ld = DataLoader(TensorDataset(G_va, y_va),
                       batch_size=PRETRAIN_BATCH, shuffle=False, num_workers=0)

    # ── Modello: GAFEncoder + head classificazione ────────────
    print("\n[2/3] Pretraining...")
    gaf_dim = cfg['gaf_aux_hidden']
    n_ch    = cfg['n_channels']
    encoder = GAFMiniEncoder(n_ch, gaf_dim, cfg['gaf_aux_dropout']).to(device)
    head    = nn.Linear(gaf_dim, cfg['n_classes']).to(device)

    opt   = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=PRETRAIN_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=PRETRAIN_EPOCHS, eta_min=1e-5)
    ce    = nn.CrossEntropyLoss()

    best_va_acc  = 0.0
    best_state   = None
    wait         = 0

    for ep in range(PRETRAIN_EPOCHS):
        # Train
        encoder.train(); head.train()
        for G, y in tr_ld:
            G, y = G.to(device), y.to(device)
            loss = ce(head(encoder(G)), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        # Val
        encoder.eval(); head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for G, y in va_ld:
                G, y = G.to(device), y.to(device)
                preds = head(encoder(G)).argmax(-1)
                correct += (preds == y).sum().item()
                total   += len(y)
        va_acc = correct / total * 100

        if va_acc > best_va_acc:
            best_va_acc = va_acc
            best_state  = {k: v.clone() for k, v in encoder.state_dict().items()}
            wait = 0; tag = " ✨ BEST"
        else:
            wait += 1; tag = f" [{wait}/{PRETRAIN_PATIENCE}]"

        if (ep + 1) % 25 == 0 or wait == 0:
            lr = opt.param_groups[0]['lr']
            print(f"  Ep {ep+1:03d}/{PRETRAIN_EPOCHS} | "
                  f"LR {lr:.1e} | Va {va_acc:.1f}%{tag}")

        if wait >= PRETRAIN_PATIENCE:
            print(f"\n  🛑 Early stop ep {ep+1}")
            break

    # ── Salvataggio ───────────────────────────────────────────
    print(f"\n[3/3] Salvataggio → {save_path}")
    print(f"  Best val accuracy: {best_va_acc:.2f}%")
    torch.save(best_state, save_path)
    print("  ✅ Pesi salvati!")
    return save_path


if __name__ == "__main__":
    pretrain_gaf_encoder(CFG)