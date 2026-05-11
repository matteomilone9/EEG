#!/usr/bin/env python3
"""
loso_pipeline_DTKDEG.py  —  TCFormer LOSO + Dual-Teacher KD con Entropy Gate
==============================================================================
Implementazione fedele di:
  Xu & Yu, "Entropy-Based Dual-Teacher Distillation for Efficient Motor
  Imagery EEG Classification", Entropy 2026, 28, 310.
  https://doi.org/10.3390/e28030310

BASE: loso_pipeline.py (loader, augmentation, scaler, CSV, argparse invariati)
AGGIUNTO:
  - build_ensemble_teachers : K teacher TCFormer trainati offline (Algorithm 1)
  - ensemble_logits         : media logits K teacher
  - softmax_tau / kd_loss   : distillation a temperatura tau
  - entropy_gate            : w(x) = clip((h-hlow)/(hhigh-hlow), 0, 1)
  - make_scheduler_2stage   : cosine annealing 2-stage con restart a N
  - fit_kd                  : student training con dual-teacher (Algorithm 2)
  - run_fold_kd / run_loso_kd: drop-in replacement di run_fold/run_loso

Iperparametri paper (Sezione 4.1):
  K=5, ET=750, tau=4, lambda_ens=0.5, lambda_ema=0.4,
  hlow=0.6, hhigh=0.9, alpha=0.995, N=500 (tot 3N=1500 ep student)

FIX1 integrato:
  K=3, lambda_ens=0.2, hlow=0.75
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

# ── import tutto da loso_pipeline.py ──────────────────────────────────────
from loso_pipeline import (
    EEGDataset,
    _append_row,
    _csv_path,
    _load_done,
    _print_summary,
    _print_multiseed,
    _read_csv,
    _div,
    build_loaders,
    build_subject_cache,
    evaluate,
    generate_folds,
    get_default_cfg,
    get_device,
    set_seed,
)
from model import build_tcformer


# ══════════════════════════════════════════════════════════════════════════
# Iperparametri default DTKDEG
# ══════════════════════════════════════════════════════════════════════════

_DTKD_DEFAULTS = dict(
    kd_n_teachers     = 3,      # FIX1: 3      | primo esperimento: 5
    kd_teacher_epochs = 750,    # invariato
    kd_teacher_lr     = 1e-3,   # invariato
    kd_teacher_wd     = 9e-3,   # invariato
    kd_tau            = 4.0,    # invariato
    kd_lambda_ens     = 0.2,    # FIX1: 0.2    | primo esperimento: 0.5
    kd_lambda_ema     = 0.4,    # invariato
    kd_hlow           = 0.75,   # FIX1: 0.75   | primo esperimento: 0.6
    kd_hhigh          = 0.9,    # invariato
    kd_ema_alpha      = 0.995,  # invariato
    kd_N              = 500,    # invariato
    kd_student_lr_max = 1e-3,   # invariato
    kd_student_lr_min = 0.0,    # invariato
    kd_student_wd     = 9e-3,   # invariato
    kd_batch_size     = 64,     # invariato
)


def get_default_cfg_kd() -> dict:
    """Restituisce la cfg base LOSO con overrides DTKDEG."""
    cfg = get_default_cfg()
    cfg.update(_DTKD_DEFAULTS)
    cfg["batch_size"] = cfg["kd_batch_size"]
    return cfg


# ══════════════════════════════════════════════════════════════════════════
# Utility KD
# ══════════════════════════════════════════════════════════════════════════

def softmax_tau(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """sigma_tau(z) = softmax(z / tau)"""
    return F.softmax(logits / tau, dim=-1)


def kd_loss(q_teacher: torch.Tensor, z_student: torch.Tensor, tau: float) -> torch.Tensor:
    """
    tau^2 * KL(q_teacher || sigma_tau(z_student))
    q_teacher : soft target gia a temperatura tau  [B, C]
    z_student : logits raw dello student           [B, C]
    """
    log_ps = F.log_softmax(z_student / tau, dim=-1)
    return (tau ** 2) * F.kl_div(log_ps, q_teacher, reduction="batchmean")


@torch.no_grad()
def entropy_gate(q_ema: torch.Tensor, hlow: float, hhigh: float) -> torch.Tensor:
    """
    w(x) per ogni sample nel batch.
    q_ema : distribuzione EMA teacher a temperatura tau  [B, C]
    return: pesi per-sample in [0, 1]                   [B]
    """
    C = q_ema.shape[-1]
    H = -(q_ema * (q_ema + 1e-12).log()).sum(dim=-1)
    h = H / math.log(C)
    return ((h - hlow) / (hhigh - hlow)).clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# Scheduler 2-stage cosine
# ══════════════════════════════════════════════════════════════════════════

def make_scheduler_2stage(
    optimizer: torch.optim.Optimizer,
    N: int,
    lr_max: float,
    lr_min: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """LambdaLR che implementa il 2-stage cosine del paper."""

    def lr_lambda(epoch: int) -> float:
        e = epoch + 1
        if e <= N:
            t, Ts = e - 1, N
        else:
            t, Ts = e - 1 - N, 2 * N
        cos_val = 0.5 * (1.0 + math.cos(math.pi * t / Ts))
        eta = lr_min + (lr_max - lr_min) * cos_val
        return eta / lr_max if lr_max > 0 else 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════
# Algorithm 1: Training offline dei K teacher
# ══════════════════════════════════════════════════════════════════════════

def _train_teacher(
    model: nn.Module,
    loader: DataLoader,
    n_epochs: int,
    lr: float,
    wd: float,
    device: torch.device,
) -> nn.Module:
    """Addestra un singolo teacher con CE-only e lr fisso (AdamW)."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(n_epochs):
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(X), y).backward()
            opt.step()
    return model


def build_ensemble_teachers(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cfg: dict,
    device: torch.device,
    verbose: bool = True,
) -> List[nn.Module]:
    """
    Algorithm 1: divide Dtr in K fold; teacher k viene trainato
    su Dtr escluso il fold k (bagging con diversita).
    Restituisce K modelli in eval mode su CPU (per risparmiare VRAM).
    """
    K    = cfg["kd_n_teachers"]          # FIX1: 3 | primo esperimento: 5
    ET   = cfg["kd_teacher_epochs"]
    lr_t = cfg["kd_teacher_lr"]
    wd_t = cfg["kd_teacher_wd"]
    bs   = cfg["kd_batch_size"]
    nw   = cfg.get("num_workers", 0)
    pin  = torch.cuda.is_available()

    n         = len(y_tr)
    fold_size = n // K
    indices   = np.arange(n)
    teachers  = []

    for k in range(K):
        if verbose:
            print(f"   [Teacher {k+1}/{K}] training {ET} epoche...", end=" ", flush=True)
        t0 = time.time()

        val_idx   = indices[k * fold_size: (k + 1) * fold_size]
        train_idx = np.concatenate([indices[: k * fold_size],
                                    indices[(k + 1) * fold_size:]])
        loader = DataLoader(
            EEGDataset(X_tr[train_idx], y_tr[train_idx]),
            batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pin,
            persistent_workers=(nw > 0),
        )

        teacher = build_tcformer(cfg).to(device)
        teacher = _train_teacher(teacher, loader, ET, lr_t, wd_t, device)
        teacher.eval().cpu()
        teachers.append(teacher)

        if verbose:
            print(f"OK ({(time.time() - t0) / 60:.1f} min)")

    return teachers


@torch.no_grad()
def ensemble_logits(
    teachers: List[nn.Module],
    X: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    z_ens(x) = (1/K) * sum_k z^(k)(x)   [B, C]
    Ogni teacher viene spostato su device solo per il forward.
    """
    logits_sum = None
    for t in teachers:
        t.to(device).eval()
        z = t(X)
        logits_sum = z if logits_sum is None else logits_sum + z
        t.cpu()
    return logits_sum / len(teachers)


# ══════════════════════════════════════════════════════════════════════════
# Algorithm 2: Student training con dual-teacher KD
# ══════════════════════════════════════════════════════════════════════════

def fit_kd(
    student: nn.Module,
    train_loader: DataLoader,
    teachers: List[nn.Module],
    cfg: dict,
    device: torch.device,
    verbose: bool = False,
) -> nn.Module:
    """
    Algorithm 2:
      Phase I  (ep 1 .. N)    : Lce + lambda_ens * Lens_kd
      Phase II (ep N+1 .. 3N) : Lce + lambda_ens * Lens_kd + lambda_ema * w(x) * Lema_kd
    EMA teacher inizializzato all'inizio di Phase II.
    """
    N        = cfg["kd_N"]
    tau      = cfg["kd_tau"]
    lam_ens  = cfg["kd_lambda_ens"]      # FIX1: 0.2 | primo esperimento: 0.5
    lam_ema  = cfg["kd_lambda_ema"]
    hlow     = cfg["kd_hlow"]            # FIX1: 0.75 | primo esperimento: 0.6
    hhigh    = cfg["kd_hhigh"]
    alpha    = cfg["kd_ema_alpha"]
    lr_max   = cfg["kd_student_lr_max"]
    lr_min   = cfg["kd_student_lr_min"]
    wd_s     = cfg["kd_student_wd"]
    total_ep = 3 * N

    opt = torch.optim.AdamW(student.parameters(), lr=lr_max, weight_decay=wd_s)
    sch = make_scheduler_2stage(opt, N, lr_max, lr_min)

    ema_model: Optional[nn.Module] = None
    student.to(device)

    for epoch in range(1, total_ep + 1):

        if epoch == N + 1:
            ema_model = copy.deepcopy(student)
            ema_model.eval()

        student.train()
        total_loss, total_n = 0.0, 0

        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            z_s  = student(X)
            loss = F.cross_entropy(z_s, y)

            with torch.no_grad():
                z_ens = ensemble_logits(teachers, X, device)
            q_ens = softmax_tau(z_ens, tau)
            loss  = loss + lam_ens * kd_loss(q_ens, z_s, tau)

            if epoch > N and ema_model is not None:
                with torch.no_grad():
                    ema_model.eval()
                    z_ema = ema_model(X)
                q_ema = softmax_tau(z_ema, tau)
                w     = entropy_gate(q_ema, hlow, hhigh)
                log_ps_tau = F.log_softmax(z_s / tau, dim=-1)
                per_sample_kl = (tau ** 2) * (
                    q_ema * (q_ema.clamp(min=1e-12).log() - log_ps_tau)
                ).sum(dim=-1)
                loss = loss + lam_ema * (w * per_sample_kl).mean()

            loss.backward()
            opt.step()

            if epoch > N and ema_model is not None:
                with torch.no_grad():
                    for p_e, p_s in zip(ema_model.parameters(), student.parameters()):
                        p_e.data.mul_(alpha).add_((1.0 - alpha) * p_s.data)

            total_loss += loss.item() * X.shape[0]
            total_n    += X.shape[0]

        sch.step()

        if verbose and (epoch % 100 == 0 or epoch == 1 or epoch == N or epoch == total_ep):
            stage = "I " if epoch <= N else "II"
            print(
                f"   Ep {epoch:4d}/{total_ep} [Ph {stage}] "
                f"loss={total_loss / max(total_n, 1):.4f} "
                f"lr={opt.param_groups[0]['lr']:.6f}"
            )

    return student


# ══════════════════════════════════════════════════════════════════════════
# run_fold_kd / run_loso_kd
# ══════════════════════════════════════════════════════════════════════════

_CSV_FIELDS_KD = ["fold", "test_sub", "seed", "acc", "kappa", "elapsed_s"]


def run_fold_kd(
    fold: dict,
    cfg: dict,
    seed: int,
    cache: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    rng    = np.random.default_rng(seed)
    device = get_device()
    t0     = time.time()

    tr_ld, _, te_ld = build_loaders(
        train_subs=fold["train_subs"],
        test_sub=fold["test_sub"],
        cfg=cfg,
        cache=cache,
        rng=rng,
    )

    if verbose:
        print(
            f"  Fold {fold['fold']:2d} [test=S{fold['test_sub']:02d} | "
            f"train={len(fold['train_subs'])} sub, {len(tr_ld.dataset)} trial]",
            flush=True,
        )
        print(f"  -> Building {cfg['kd_n_teachers']} ensemble teachers...", flush=True)

    X_list, y_list = [], []
    for Xb, yb in tr_ld:
        X_list.append(Xb.numpy())
        y_list.append(yb.numpy())
    X_tr_np = np.concatenate(X_list, axis=0)
    y_tr_np = np.concatenate(y_list, axis=0)

    teachers = build_ensemble_teachers(X_tr_np, y_tr_np, cfg, device, verbose=verbose)

    if verbose:
        print(
            f"  -> Training student "
            f"(3x{cfg['kd_N']}={3 * cfg['kd_N']} epoche)...",
            flush=True,
        )

    student = build_tcformer(cfg).to(device)
    student = fit_kd(student, tr_ld, teachers, cfg, device, verbose=verbose)

    acc, kappa = evaluate(student, te_ld, device)
    elapsed    = time.time() - t0

    if verbose:
        print(f"  Fold {fold['fold']:2d} -> Acc={acc:.2f}% kappa={kappa:.4f} ({elapsed / 60:.1f} min)")

    return {
        "fold":      fold["fold"],
        "test_sub":  fold["test_sub"],
        "seed":      seed,
        "acc":       acc,
        "kappa":     kappa,
        "elapsed_s": elapsed,
    }


def run_loso_kd(
    cfg: dict,
    seeds: list = (42,),
    cache: Optional[dict] = None,
    out_dir: str = "./loso_dtkdeg_fix1",
    verbose: bool = True,
    subjects: Optional[list] = None,
) -> list:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    all_folds = generate_folds(cfg.get("n_subjects", 9))
    folds     = (
        [f for f in all_folds if f["test_sub"] in subjects]
        if subjects else all_folds
    )

    if subjects and not folds:
        raise ValueError(f"Nessun fold trovato per subjects={subjects}.")
    if subjects:
        print(f"[LOSO-DTKDEG] Soggetti: {subjects} -> {len(folds)} fold")

    all_results: list = []

    for seed in seeds:
        csv_file = _csv_path(out_dir, seed)
        done     = _load_done(csv_file, seed)
        if done:
            print(f"[Resume] Seed={seed}: {len(done)} fold gia completati.")
        _div()
        print(
            f"LOSO-DTKDEG | seed={seed} | device={get_device()} | "
            f"K={cfg['kd_n_teachers']} teachers (primo exp: 5) | "
            f"lambda_ens={cfg['kd_lambda_ens']} (primo exp: 0.5) | "
            f"hlow={cfg['kd_hlow']} (primo exp: 0.6) | "
            f"3N={3 * cfg['kd_N']} ep student"
        )
        _div()

        for fold in folds:
            key = (fold["fold"], seed)
            if key in done:
                print(f"  Fold {fold['fold']:2d} -> gia completato, skip.")
                continue
            result = run_fold_kd(fold, cfg, seed, cache=cache, verbose=verbose)
            all_results.append(result)
            _append_row(csv_file, result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        seed_results = _read_csv(csv_file, seed)
        _print_summary(seed_results, title=f"LOSO-DTKDEG Summary -- seed={seed}")

    if len(list(seeds)) > 1:
        _print_multiseed(out_dir, list(seeds))

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# __main__
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TCFormer LOSO + Dual-Teacher KD con Entropy Gate (FIX1 integrato)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42])
    parser.add_argument("--subjects",        type=int, nargs="+", default=None)
    parser.add_argument("--fold",            type=int, default=None)
    parser.add_argument("--out_dir",         default="./loso_dtkdeg_fix1")
    parser.add_argument("--cache",           action="store_true")
    parser.add_argument("--n_teachers",      type=int, default=3,    help="FIX1: 3 | primo esperimento: 5")
    parser.add_argument("--teacher_epochs",  type=int, default=750)
    parser.add_argument("--kd_N",            type=int, default=500,  help="Phase I length N (student total = 3N)")
    parser.add_argument("--tau",             type=float, default=4.0)
    parser.add_argument("--lambda_ens",      type=float, default=0.2, help="FIX1: 0.2 | primo esperimento: 0.5")
    parser.add_argument("--lambda_ema",      type=float, default=0.4)
    parser.add_argument("--ema_alpha",       type=float, default=0.995)
    parser.add_argument("--hlow",            type=float, default=0.75, help="FIX1: 0.75 | primo esperimento: 0.6")
    parser.add_argument("--hhigh",           type=float, default=0.9)
    parser.add_argument("--batch_size",      type=int, default=64)
    parser.add_argument("--trans_depth",     type=int, default=5)
    parser.add_argument("--num_workers",     type=int, default=0)
    parser.add_argument("--no_sr",           action="store_true")
    parser.add_argument("--no_interaug",     action="store_true")
    parser.add_argument("--verbose",         action="store_true")
    args = parser.parse_args()

    cfg = get_default_cfg_kd()
    cfg["kd_n_teachers"]     = args.n_teachers
    cfg["kd_teacher_epochs"] = args.teacher_epochs
    cfg["kd_N"]              = args.kd_N
    cfg["kd_tau"]            = args.tau
    cfg["kd_lambda_ens"]     = args.lambda_ens
    cfg["kd_lambda_ema"]     = args.lambda_ema
    cfg["kd_ema_alpha"]      = args.ema_alpha
    cfg["kd_hlow"]           = args.hlow
    cfg["kd_hhigh"]          = args.hhigh
    cfg["kd_batch_size"]     = args.batch_size
    cfg["batch_size"]        = args.batch_size
    cfg["trans_depth"]       = args.trans_depth
    cfg["num_workers"]       = args.num_workers
    cfg["use_sr"]            = not args.no_sr
    cfg["interaug"]          = not args.no_interaug

    subject_cache = build_subject_cache(cfg) if args.cache else None

    if args.fold is not None:
        all_folds = generate_folds(cfg["n_subjects"])
        fi = next((f for f in all_folds if f["fold"] == args.fold), None)
        if fi is None:
            raise ValueError(f"Fold {args.fold} non valido.")
        for seed in args.seeds:
            print(run_fold_kd(fi, cfg, seed, cache=subject_cache, verbose=True))
    else:
        run_loso_kd(
            cfg,
            seeds=args.seeds,
            cache=subject_cache,
            out_dir=args.out_dir,
            verbose=args.verbose,
            subjects=args.subjects,
        )