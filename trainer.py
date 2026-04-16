# trainer.py — DistillationTrainer, FineTuner, evaluate
# ============================================================

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score

from config import _mode_tag
from augmentation import segment_and_reconstruct, mixup_batch


# ── TTA augmentation + evaluate ──────────────────────────────

def _tta_augment(x: torch.Tensor) -> torch.Tensor:
    x = x + torch.randn_like(x) * 0.02 * x.std()
    return x * torch.FloatTensor(1).uniform_(0.9, 1.1).to(x.device)


def evaluate(model: nn.Module, loader, device, cfg):
    model.eval()
    n_tta       = cfg['n_tta']
    use_gaf     = cfg['use_gaf']
    use_gaf_inf = use_gaf and cfg['gaf_inference']
    alpha       = cfg['gaf_inference_alpha']
    preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            eeg = b['eeg'].to(device)
            gaf = b['gaf'].to(device)
            y   = b['label'].numpy()
            if use_gaf_inf:
                feat        = model.tcformer.get_features(eeg)
                gaf_emb     = model.gaf_enc(gaf)
                logits_aux  = model.aux_head(feat.detach(), gaf_emb)
                logits_main = model.tcformer.tcn_head(feat)
                for _ in range(n_tta - 1):
                    f2 = model.tcformer.get_features(_tta_augment(eeg))
                    logits_main = logits_main + model.tcformer.tcn_head(f2)
                logits_main /= max(n_tta, 1)
                logits = alpha * logits_main + (1.0 - alpha) * logits_aux
            elif use_gaf:
                logits = model(eeg, gaf=None)
                for _ in range(n_tta - 1):
                    logits = logits + model(_tta_augment(eeg), gaf=None)
                logits /= max(n_tta, 1)
            else:
                logits = model(eeg)
                for _ in range(n_tta - 1):
                    logits = logits + model(_tta_augment(eeg))
                logits /= max(n_tta, 1)
            preds.extend(logits.argmax(-1).cpu().numpy())
            trues.extend(y)
    return np.array(trues), np.array(preds)


# ── Helper condiviso per il loop di un'epoca ─────────────────

def _run_epoch(model, loader, device, cfg, optimizer, train: bool):
    model.train(train)
    tot_loss, preds, trues = 0.0, [], []
    lam       = cfg['lambda_aux']
    use_gaf   = cfg['use_gaf']
    use_mixup = cfg['use_mixup']
    ce        = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(train):
        for b in loader:
            eeg = b['eeg'].to(device)
            gaf = b['gaf'].to(device)
            y   = b['label'].to(device)

            if train:
                eeg = segment_and_reconstruct(
                    eeg, y,
                    n_segments=cfg['n_segments'],
                    sr_prob=cfg['sr_prob'])
                optimizer.zero_grad()

                if use_mixup:
                    eeg, y_soft = mixup_batch(
                        eeg, y,
                        n_classes=cfg['n_classes'],
                        mixup_prob=cfg['mixup_prob'],
                        alpha=cfg['mixup_alpha'])
                    if use_gaf:
                        logits_m, logits_a = model(eeg, gaf)
                        loss = -(y_soft * F.log_softmax(logits_m, -1)).sum(-1).mean()
                        if logits_a is not None:
                            loss = loss + lam * (-(y_soft * F.log_softmax(logits_a, -1)).sum(-1).mean())
                    else:
                        logits_m = model(eeg)
                        loss = -(y_soft * F.log_softmax(logits_m, -1)).sum(-1).mean()
                else:
                    if use_gaf:
                        logits_m, logits_a = model(eeg, gaf)
                        loss = ce(logits_m, y)
                        if logits_a is not None:
                            loss = loss + lam * ce(logits_a, y)
                    else:
                        logits_m = model(eeg)
                        loss = ce(logits_m, y)

                loss.backward()
                optimizer.step()
            else:
                logits_m = model(eeg, gaf=None) if use_gaf else model(eeg)
                loss     = ce(logits_m, y)

            tot_loss += loss.item()
            preds.extend(logits_m.argmax(-1).cpu().numpy())
            trues.extend(y.cpu().numpy())

    acc   = accuracy_score(trues, preds) * 100
    kappa = cohen_kappa_score(trues, preds)
    return tot_loss / max(len(loader), 1), acc, kappa


# ── DistillationTrainer ──────────────────────────────────────

class DistillationTrainer:
    """Trainer principale con Mixup, S&R, aux loss e early stopping."""

    def __init__(self, model: nn.Module, cfg: dict):
        self.model  = model.to(cfg['device'])
        self.device = cfg['device']
        self.cfg    = cfg
        self.opt    = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        self.sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg['epochs'], eta_min=1e-5)
        self.best_acc   = 0.0
        self.best_state = None
        self.history    = dict(tr_loss=[], va_loss=[], tr_acc=[], va_acc=[], va_kap=[])

    def fit(self, tr_ld, va_ld, seed: int = 42):
        epochs = self.cfg['epochs']
        pat    = self.cfg['patience']
        wait   = 0
        print(f"\n{'='*65}\nMode: {_mode_tag(self.cfg)} | seed={seed} | "
              f"epochs={epochs} | patience={pat}\n{'='*65}")

        for ep in range(epochs):
            tr_loss, tr_acc, _    = _run_epoch(self.model, tr_ld, self.device,
                                               self.cfg, self.opt, train=True)
            va_loss, va_acc, va_k = _run_epoch(self.model, va_ld, self.device,
                                               self.cfg, self.opt, train=False)
            self.sched.step()
            lr = self.opt.param_groups[0]['lr']

            for k, v in zip(['tr_loss', 'va_loss', 'tr_acc', 'va_acc', 'va_kap'],
                            [tr_loss, va_loss, tr_acc, va_acc, va_k]):
                self.history[k].append(v)

            if va_acc > self.best_acc:
                self.best_acc   = va_acc
                self.best_state = copy.deepcopy(self.model.state_dict())
                wait = 0; tag = ' ✨ BEST'
            else:
                wait += 1; tag = f' [{wait}/{pat}]'

            lam_str = f"λ={self.cfg['lambda_aux']:.3f} | " if self.cfg['use_gaf'] else ""
            print(f"Ep {ep+1:03d}/{epochs} | {lam_str}LR {lr:.1e} | "
                  f"Tr {tr_acc:.1f}% {tr_loss:.4f} | "
                  f"Va {va_acc:.1f}% {va_loss:.4f} k={va_k:.3f}{tag}")

            if wait >= pat:
                print(f"\n🛑 Early stop ep {ep+1}")
                break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        print(f"\n🏆 Best val: {self.best_acc:.2f}%")
        return self.history


# ── FineTuner (Fase 2 post-LOSO) ────────────────────────────

class FineTuner:
    """
    Trainer per la Fase 2 del fine-tuning post-LOSO.
    Parte dai pesi LOSO pre-addestrati.
    Usa lr ridotto, meno epoche, freeze opzionale del backbone.

    ft_freeze_backbone=True → congela conv_block, mix, transformer, reduce.
    ft_freeze_backbone=False → fine-tuning completo di tutti i layer.
    """

    def __init__(self, model: nn.Module, cfg: dict):
        self.model  = model.to(cfg['device'])
        self.device = cfg['device']
        self.cfg    = cfg

        if cfg.get('ft_freeze_backbone', False):
            for name, param in model.tcformer.named_parameters():
                if any(p in name for p in ['conv_block', 'mix', 'transformer', 'reduce']):
                    param.requires_grad = False
            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            total  = sum(p.numel() for p in model.parameters())
            print(f" [FT] Backbone CONGELATO — {frozen:,}/{total:,} param frozen "
                  f"({100*frozen/total:.1f}%)")
        else:
            print(f" [FT] Fine-tuning COMPLETO — tutti i parametri liberi")

        trainable  = [p for p in model.parameters() if p.requires_grad]
        self.opt   = torch.optim.Adam(trainable, lr=cfg['ft_lr'])
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg['ft_epochs'], eta_min=cfg['ft_lr'] * 0.01)
        self.best_acc   = 0.0
        self.best_state = None

    def fit(self, tr_ld, va_ld, seed: int = 42):
        epochs     = self.cfg['ft_epochs']
        pat        = self.cfg['ft_patience']
        wait       = 0
        freeze_tag = "backbone frozen" if self.cfg.get('ft_freeze_backbone', False) else "full FT"
        print(f"\n  {'─'*55}")
        print(f"  [FT Fase 2] seed={seed} | lr={self.cfg['ft_lr']:.1e} | "
              f"epochs={epochs} | patience={pat} | {freeze_tag}")
        print(f"  {'─'*55}")

        for ep in range(epochs):
            tr_loss, tr_acc, _    = _run_epoch(self.model, tr_ld, self.device,
                                               self.cfg, self.opt, train=True)
            va_loss, va_acc, va_k = _run_epoch(self.model, va_ld, self.device,
                                               self.cfg, self.opt, train=False)
            self.sched.step()
            lr = self.opt.param_groups[0]['lr']

            if va_acc > self.best_acc:
                self.best_acc   = va_acc
                self.best_state = copy.deepcopy(self.model.state_dict())
                wait = 0; tag = ' ✨ BEST'
            else:
                wait += 1; tag = f' [{wait}/{pat}]'

            print(f"  FT {ep+1:03d}/{epochs} | LR {lr:.1e} | "
                  f"Tr {tr_acc:.1f}% {tr_loss:.4f} | "
                  f"Va {va_acc:.1f}% {va_loss:.4f} k={va_k:.3f}{tag}")

            if wait >= pat:
                print(f"\n  🛑 FT early stop ep {ep+1}")
                break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        print(f"\n  🏆 FT Best val: {self.best_acc:.2f}%")
        return self.best_acc