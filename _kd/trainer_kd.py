# trainer_kd.py — Trainer separati per Teacher e Student KD
# Import aggiornati: augmentation_kd invece di augmentation
# ============================================================

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, cohen_kappa_score

from augmentation_kd import segment_and_reconstruct, mixup_batch
from losses_kd import ContrastiveDistillLoss, AdaptiveKDLoss


# ── TTA augmentation ─────────────────────────────────────────

def _tta_augment(x: torch.Tensor) -> torch.Tensor:
    x = x + torch.randn_like(x) * 0.02 * x.std()
    return x * torch.FloatTensor(1).uniform_(0.9, 1.1).to(x.device)


# ── Evaluation ───────────────────────────────────────────────

def evaluate_student(model: nn.Module, loader, device, cfg: dict):
    model.eval()
    n_tta = cfg["n_tta"]
    preds, trues = [], []

    with torch.no_grad():
        for b in loader:
            eeg = b["eeg"].to(device)
            y = b["label"].cpu().numpy()

            logits = model(eeg)
            for _ in range(n_tta - 1):
                logits = logits + model(_tta_augment(eeg))
            logits = logits / max(n_tta, 1)

            preds.extend(logits.argmax(-1).cpu().numpy())
            trues.extend(y)

    return np.array(trues), np.array(preds)


def evaluate_teacher(model: nn.Module, loader, device, cfg: dict):
    model.eval()
    n_tta = cfg["n_tta"]
    use_gaf = cfg.get("use_gaf", True)
    preds, trues = [], []

    with torch.no_grad():
        for b in loader:
            eeg = b["eeg"].to(device)
            gaf = b["gaf"].to(device) if use_gaf else None
            y = b["label"].cpu().numpy()

            logits, _ = model(eeg, gaf)
            for _ in range(n_tta - 1):
                l, _ = model(_tta_augment(eeg), gaf)
                logits = logits + l
            logits = logits / max(n_tta, 1)

            preds.extend(logits.argmax(-1).cpu().numpy())
            trues.extend(y)

    return np.array(trues), np.array(preds)


# ── KD loss helper ───────────────────────────────────────────

def kd_loss(student_logits, teacher_logits, labels,
            alpha=0.5, temperature=2.0, label_smoothing=0.0):
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_ce = ce(student_logits, labels)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    loss_kd = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)

    loss = alpha * loss_ce + (1.0 - alpha) * loss_kd
    return loss, loss_ce.detach(), loss_kd.detach()


# ── Teacher epoch loop ───────────────────────────────────────

def _run_teacher_epoch(model, loader, device, cfg, optimizer=None, train=True):
    model.train(train)
    use_gaf = cfg.get("use_gaf", True)
    tot_loss = 0.0
    preds, trues = [], []

    ce = nn.CrossEntropyLoss(label_smoothing=cfg.get("teacher_label_smoothing", 0.15))

    with torch.set_grad_enabled(train):
        for b in loader:
            eeg = b["eeg"].to(device)
            gaf = b["gaf"].to(device) if use_gaf else None
            y = b["label"].to(device)

            if train:
                eeg = segment_and_reconstruct(
                    eeg, y,
                    n_segments=cfg["n_segments"],
                    sr_prob=cfg["sr_prob"],
                )
                optimizer.zero_grad()

            if cfg["use_mixup"]:
                eeg, y_soft = mixup_batch(
                    eeg, y,
                    n_classes=cfg["n_classes"],
                    mixup_prob=cfg["mixup_prob"],
                    alpha=cfg["mixup_alpha"],
                )
                logits, _ = model(eeg, gaf)
                loss = -(y_soft * F.log_softmax(logits, -1)).sum(-1).mean()
            else:
                logits, _ = model(eeg, gaf)
                loss = ce(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            tot_loss += loss.item()
            preds.extend(logits.argmax(-1).detach().cpu().numpy())
            trues.extend(y.detach().cpu().numpy())

    acc = accuracy_score(trues, preds) * 100.0
    kappa = cohen_kappa_score(trues, preds)
    return tot_loss / max(len(loader), 1), acc, kappa


# ── Student KD epoch loop ────────────────────────────────────

def _run_student_kd_epoch(student, teacher, loader, device, cfg,
                          optimizer=None, train=True):
    student.train(train)
    teacher.eval()
    use_gaf = cfg.get("use_gaf", True)
    tot_loss = tot_ce = tot_kd = 0.0
    preds, trues = [], []

    with torch.set_grad_enabled(train):
        for b in loader:
            eeg = b["eeg"].to(device)
            gaf = b["gaf"].to(device) if use_gaf else None
            y = b["label"].to(device)

            if train:
                eeg = segment_and_reconstruct(
                    eeg, y,
                    n_segments=cfg["n_segments"],
                    sr_prob=cfg["sr_prob"],
                )
                optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits, _ = teacher(eeg, gaf)

            student_logits = student(eeg)

            loss, loss_ce, loss_kd = kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=y,
                alpha=cfg["kd_alpha"],
                temperature=cfg["kd_temperature"],
                label_smoothing=cfg["student_label_smoothing"],
            )

            if train:
                loss.backward()
                optimizer.step()

            tot_loss += loss.item()
            tot_ce += float(loss_ce)
            tot_kd += float(loss_kd)
            preds.extend(student_logits.argmax(-1).detach().cpu().numpy())
            trues.extend(y.detach().cpu().numpy())

    n = max(len(loader), 1)
    acc = accuracy_score(trues, preds) * 100.0
    kappa = cohen_kappa_score(trues, preds)
    return tot_loss / n, tot_ce / n, tot_kd / n, acc, kappa


# ── Student KD + Align epoch loop ────────────────────────────

def _run_student_kd_align_epoch(
    student, teacher, gaf_encoder, proj_head,
    loader, device, cfg,
    align_loss_fn, adaptive_kd_fn,
    optimizer=None, train=True,
):
    student.train(train)
    proj_head.train(train)
    teacher.eval()
    gaf_encoder.eval()

    use_gaf = cfg.get("use_gaf", True)
    beta = cfg.get("align_beta", 0.3)
    tot_loss = tot_ce = tot_kd = tot_al = tot_alpha = 0.0
    preds, trues = [], []

    with torch.set_grad_enabled(train):
        for b in loader:
            eeg = b["eeg"].to(device)
            gaf = b["gaf"].to(device) if use_gaf else None
            y = b["label"].to(device)

            if train:
                eeg = segment_and_reconstruct(
                    eeg, y,
                    n_segments=cfg["n_segments"],
                    sr_prob=cfg["sr_prob"],
                )
                optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits, _ = teacher(eeg, gaf)

            # Student forward
            feat = student.get_features(eeg)                    # [B, d_tcf, T]
            student_logits = student.student.tcn_head(feat)     # unwrap EEGStudentWrapper

            loss_kd_total, alpha_mean = adaptive_kd_fn(
                logits_student=student_logits,
                logits_teacher=teacher_logits,
                labels=y,
            )

            with torch.no_grad():
                loss_ce_log = F.cross_entropy(student_logits, y)

            loss = loss_kd_total

            gaf_is_real = (gaf is not None) and (gaf.shape[-1] > 1) and (gaf.shape[-2] > 1)
            if gaf_is_real:
                with torch.no_grad():
                    z_gaf = gaf_encoder(gaf).squeeze(1)
                z_eeg = proj_head(feat)
                l_align = align_loss_fn(z_eeg, z_gaf)
                loss = (1.0 - beta) * loss + beta * l_align
                tot_al += float(l_align)

            if train:
                loss.backward()
                optimizer.step()

            tot_loss += loss.item()
            tot_ce += float(loss_ce_log)
            tot_kd += float(loss_kd_total)
            tot_alpha += float(alpha_mean)
            preds.extend(student_logits.argmax(-1).detach().cpu().numpy())
            trues.extend(y.detach().cpu().numpy())

    n = max(len(loader), 1)
    acc = accuracy_score(trues, preds) * 100.0
    kappa = cohen_kappa_score(trues, preds)
    return tot_loss / n, tot_ce / n, tot_kd / n, tot_al / n, tot_alpha / n, acc, kappa


# ── Teacher trainer ──────────────────────────────────────────

class TeacherTrainer:
    def __init__(self, model: nn.Module, cfg: dict):
        self.model = model.to(cfg["device"])
        self.device = cfg["device"]
        self.cfg = cfg

        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["lr_teacher"],
            weight_decay=cfg["weight_decay"],
        )

        warmup_epochs = cfg["warmup_epochs_teacher"]
        total_epochs = cfg["epochs_teacher"]

        def lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / max(warmup_epochs, 1)
            progress = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.best_acc = 0.0
        self.best_state = None
        self.history = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": [], "va_kap": []}

    def fit(self, tr_ld, va_ld, seed: int = 42):
        epochs = self.cfg["epochs_teacher"]
        pat = self.cfg["patience_teacher"]
        wait = 0

        print(f"\n{'='*70}")
        print(f"Teacher training | seed={seed} | epochs={epochs} | patience={pat}")
        print(f"{'='*70}")

        for ep in range(epochs):
            tr_loss, tr_acc, _ = _run_teacher_epoch(
                self.model, tr_ld, self.device, self.cfg, self.opt, train=True)
            va_loss, va_acc, va_k = _run_teacher_epoch(
                self.model, va_ld, self.device, self.cfg, train=False)

            self.sched.step()
            lr = self.opt.param_groups[0]["lr"]

            for k, v in zip(
                ["tr_loss", "va_loss", "tr_acc", "va_acc", "va_kap"],
                [tr_loss, va_loss, tr_acc, va_acc, va_k],
            ):
                self.history[k].append(v)

            if va_acc > self.best_acc:
                self.best_acc = va_acc
                self.best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
                tag = " ✨ BEST"
            else:
                wait += 1
                tag = f" [{wait}/{pat}]"

            print(
                f"Teacher {ep+1:03d}/{epochs} | LR {lr:.1e} | "
                f"Tr {tr_acc:.1f}% {tr_loss:.4f} | "
                f"Va {va_acc:.1f}% {va_loss:.4f} k={va_k:.3f}{tag}"
            )

            if wait >= pat:
                print(f"\n🛑 Teacher early stop ep {ep+1}")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        print(f"\n🏆 Teacher best val: {self.best_acc:.2f}%")
        return self.history


# ── KD trainer ───────────────────────────────────────────────

class KDTrainer:
    def __init__(self, teacher: nn.Module, student: nn.Module, cfg: dict):
        self.teacher = teacher.to(cfg["device"])
        self.student = student.to(cfg["device"])
        self.device = cfg["device"]
        self.cfg = cfg

        if cfg["freeze_teacher"]:
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.opt = torch.optim.Adam(
            self.student.parameters(),
            lr=cfg["lr_student"],
            weight_decay=cfg["weight_decay"],
        )

        warmup_epochs = cfg["warmup_epochs_student"]
        total_epochs = cfg["epochs_student"]

        def lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / max(warmup_epochs, 1)
            progress = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.best_acc = 0.0
        self.best_state = None
        self.history = {
            "tr_loss": [], "va_loss": [],
            "tr_ce": [], "va_ce": [],
            "tr_kd": [], "va_kd": [],
            "tr_acc": [], "va_acc": [],
            "va_kap": [],
        }

    def fit(self, tr_ld, va_ld, seed: int = 42):
        epochs = self.cfg["epochs_student"]
        pat = self.cfg["patience_student"]
        wait = 0

        print(f"\n{'='*70}")
        print(
            f"Student KD training | seed={seed} | "
            f"epochs={epochs} | patience={pat} | "
            f"alpha={self.cfg['kd_alpha']:.2f} | T={self.cfg['kd_temperature']:.2f}"
        )
        print(f"{'='*70}")

        for ep in range(epochs):
            tr_loss, tr_ce, tr_kd, tr_acc, _ = _run_student_kd_epoch(
                self.student, self.teacher, tr_ld, self.device, self.cfg, self.opt, train=True)
            va_loss, va_ce, va_kd, va_acc, va_k = _run_student_kd_epoch(
                self.student, self.teacher, va_ld, self.device, self.cfg, optimizer=None, train=False)

            self.sched.step()
            lr = self.opt.param_groups[0]["lr"]

            for k, v in zip(
                ["tr_loss", "va_loss", "tr_ce", "va_ce", "tr_kd", "va_kd",
                 "tr_acc", "va_acc", "va_kap"],
                [tr_loss, va_loss, tr_ce, va_ce, tr_kd, va_kd, tr_acc, va_acc, va_k],
            ):
                self.history[k].append(v)

            if va_acc > self.best_acc:
                self.best_acc = va_acc
                self.best_state = copy.deepcopy(self.student.state_dict())
                wait = 0
                tag = " ✨ BEST"
            else:
                wait += 1
                tag = f" [{wait}/{pat}]"

            print(
                f"Student {ep+1:03d}/{epochs} | LR {lr:.1e} | "
                f"Tr {tr_acc:.1f}% L={tr_loss:.4f} CE={tr_ce:.4f} KD={tr_kd:.4f} | "
                f"Va {va_acc:.1f}% L={va_loss:.4f} CE={va_ce:.4f} KD={va_kd:.4f} "
                f"k={va_k:.3f}{tag}"
            )

            if wait >= pat:
                print(f"\n🛑 Student early stop ep {ep+1}")
                break

        if self.best_state is not None:
            self.student.load_state_dict(self.best_state)

        print(f"\n🏆 Student KD best val: {self.best_acc:.2f}%")
        return self.history


# ── KD Align trainer ─────────────────────────────────────────

class KDAlignTrainer:
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 gaf_encoder: nn.Module, proj_head: nn.Module,
                 cfg: dict):
        self.teacher = teacher.to(cfg["device"])
        self.student = student.to(cfg["device"])
        self.gaf_encoder = gaf_encoder.to(cfg["device"])
        self.proj_head = proj_head.to(cfg["device"])
        self.device = cfg["device"]
        self.cfg = cfg

        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.gaf_encoder.parameters():
            p.requires_grad = False

        params = list(self.student.parameters()) + list(self.proj_head.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg["lr_student"], weight_decay=cfg["weight_decay"])

        warmup_epochs = cfg["warmup_epochs_student"]
        total_epochs = cfg["epochs_student"]

        def lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / max(warmup_epochs, 1)
            progress = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

        self.align_loss_fn = ContrastiveDistillLoss(
            temperature=cfg.get("align_temperature", 0.07),
            symmetric=True,
        )
        self.adaptive_kd_fn = AdaptiveKDLoss(
            alpha_base=cfg.get("kd_alpha", 0.5),
            temperature=cfg.get("kd_temperature", 4.0),
            threshold=cfg.get("kd_threshold", 0.5),
        )

        self.best_acc = 0.0
        self.best_state = None
        self.history = {
            "tr_loss": [], "va_loss": [],
            "tr_ce": [], "va_ce": [],
            "tr_kd": [], "va_kd": [],
            "tr_al": [], "va_al": [],
            "tr_alpha": [], "va_alpha": [],
            "tr_acc": [], "va_acc": [],
            "va_kap": [],
        }

    def fit(self, tr_ld, va_ld, seed: int = 42):
        epochs = self.cfg["epochs_student"]
        pat = self.cfg["patience_student"]
        beta = self.cfg.get("align_beta", 0.3)
        temp = self.cfg.get("align_temperature", 0.07)
        thr = self.cfg.get("kd_threshold", 0.5)
        wait = 0

        print(f"\n{'='*70}")
        print(
            f"Student KD-Align training | seed={seed} | "
            f"epochs={epochs} | patience={pat} | "
            f"alpha={self.cfg['kd_alpha']:.2f} | T={self.cfg['kd_temperature']:.2f} | "
            f"beta={beta:.2f} | τ={temp:.3f} | kd_thr={thr:.2f}"
        )
        print(f"{'='*70}")

        for ep in range(epochs):
            tr_loss, tr_ce, tr_kd, tr_al, tr_alpha, tr_acc, _ = _run_student_kd_align_epoch(
                self.student, self.teacher, self.gaf_encoder, self.proj_head,
                tr_ld, self.device, self.cfg, self.align_loss_fn,
                self.adaptive_kd_fn, self.opt, train=True,
            )
            va_loss, va_ce, va_kd, va_al, va_alpha, va_acc, va_k = _run_student_kd_align_epoch(
                self.student, self.teacher, self.gaf_encoder, self.proj_head,
                va_ld, self.device, self.cfg, self.align_loss_fn,
                self.adaptive_kd_fn, optimizer=None, train=False,
            )

            self.sched.step()
            lr = self.opt.param_groups[0]["lr"]

            for key, val in zip(
                ["tr_loss", "va_loss", "tr_ce", "va_ce", "tr_kd", "va_kd",
                 "tr_al", "va_al", "tr_alpha", "va_alpha", "tr_acc", "va_acc", "va_kap"],
                [tr_loss, va_loss, tr_ce, va_ce, tr_kd, va_kd,
                 tr_al, va_al, tr_alpha, va_alpha, tr_acc, va_acc, va_k],
            ):
                self.history[key].append(val)

            if va_acc > self.best_acc:
                self.best_acc = va_acc
                self.best_state = copy.deepcopy(self.student.state_dict())
                wait = 0
                tag = " ✨ BEST"
            else:
                wait += 1
                tag = f" [{wait}/{pat}]"

            print(
                f"KD-Align {ep+1:03d}/{epochs} | LR {lr:.1e} | "
                f"Tr {tr_acc:.1f}% L={tr_loss:.4f} CE={tr_ce:.4f} "
                f"KD={tr_kd:.4f} AL={tr_al:.4f} α={tr_alpha:.3f} | "
                f"Va {va_acc:.1f}% L={va_loss:.4f} CE={va_ce:.4f} "
                f"KD={va_kd:.4f} AL={va_al:.4f} α={va_alpha:.3f} "
                f"k={va_k:.3f}{tag}"
            )

            if wait >= pat:
                print(f"\n🛑 KD-Align early stop ep {ep+1}")
                break

        if self.best_state is not None:
            self.student.load_state_dict(self.best_state)

        print(f"\n🏆 Student KD-Align best val: {self.best_acc:.2f}%")
        return self.history
