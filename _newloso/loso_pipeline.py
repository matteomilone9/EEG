"""
loso_pipeline.py — Pipeline LOSO per TCFormer EEG-Only
=======================================================
Protocollo esatto da:
  [TCFormer]  Altaheri et al., Scientific Reports, 2025
  [Wimpff]    Wimpff et al., J. Neural Eng., 2024

REGOLE DI TRAINING (direttamente dai paper):
  - Cross-subject BCIC: 125 epoche, Adam lr=9e-4
  - Warmup lineare 3 epoche + cosine decay  [Wimpff sec.2.4, TCFormer sec."Experimental setup"]
  - FINAL checkpoint (no early stopping, no validation split)  [TCFormer sec."Evaluation"]
  - Normalizzazione per canale: mu/sigma su axis=(trial, time) del solo training set
  - S&R augmentation: N_s=8, m_A=m (raddoppia training)  [TCFormer sec."Data augmentation"]
  - 5 seed per soggetto per BCIC  [TCFormer sec."Evaluation"]
  - LOSO: train = sessione 1 di tutti tranne target; test = sessione 2 del target
"""

from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from model import TCFormer, build_tcformer


# ════════════════════════════════════════════════════════════
# 0. Utilità
# ════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _div(w: int = 70) -> None:
    print("─" * w)


def _metrics(y_true, y_pred) -> tuple[float, float]:
    acc   = accuracy_score(y_true, y_pred) * 100.0
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, kappa


# ════════════════════════════════════════════════════════════
# 1. Caricamento dati
# ════════════════════════════════════════════════════════════

def _load_bcic2a_moabb(subject_id: int, cfg: dict) -> dict:
    """
    Carica BCIC IV-2a via MOABB.
    Bandpass 4-40 Hz, resample 250 Hz, finestra [0, 4] s.
    Sessione 0 = train, Sessione 1 = test (protocollo ufficiale).
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as e:
        raise ImportError("Installa moabb: pip install moabb") from e

    paradigm = MotorImagery(
        n_classes = cfg.get("n_classes", 4),
        fmin      = cfg.get("fmin", 4.0),
        fmax      = cfg.get("fmax", 40.0),
        tmin      = cfg.get("tmin", 0.0),
        tmax      = cfg.get("tmax", 4.0),
        resample  = cfg.get("sfreq", 250),
    )
    X, y, meta = paradigm.get_data(
        BNCI2014_001(), subjects=[subject_id], return_epochs=False
    )
    label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    y_int = np.array(
        [label_map[str(l)] if str(l) in label_map else int(l) for l in y],
        dtype=np.int64,
    )
    m_tr = meta["session"] == "0train"
    m_te = meta["session"] == "1test"
    return {
        "X_train": X[m_tr].astype(np.float32),
        "y_train": y_int[m_tr],
        "X_test":  X[m_te].astype(np.float32),
        "y_test":  y_int[m_te],
    }


def _load_bcic2a_braindecode(subject_id: int, cfg: dict) -> dict:
    try:
        from braindecode.datasets import BNCI2014001
        from braindecode.preprocessing import (
            Preprocessor, create_windows_from_events, preprocess,
        )
    except ImportError as e:
        raise ImportError("Installa braindecode: pip install braindecode") from e

    ds = BNCI2014001(subject_ids=[subject_id])
    preprocess(ds, [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor("resample", sfreq=cfg.get("sfreq", 250)),
        Preprocessor("filter",
                     l_freq=cfg.get("fmin", 4.0),
                     h_freq=cfg.get("fmax", 40.0)),
    ])
    T_samp = int(cfg.get("tmax", 4.0) * cfg.get("sfreq", 250))
    wins   = create_windows_from_events(
        ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=T_samp,
        preload=True,
    )
    X_list, y_list, s_list = [], [], []
    for i in range(len(wins)):
        xi, yi, md = wins[i]
        X_list.append(xi)
        y_list.append(yi)
        s_list.append(md["session"])
    X  = np.stack(X_list).astype(np.float32)
    y  = np.array(y_list, dtype=np.int64)
    ss = np.array(s_list)
    return {
        "X_train": X[ss == 0], "y_train": y[ss == 0],
        "X_test":  X[ss == 1], "y_test":  y[ss == 1],
    }


def _load_bcic2a_numpy(subject_id: int, cfg: dict) -> dict:
    d = Path(cfg.get("data_dir", "./data"))
    files = {
        "train": d / f"S{subject_id:02d}_train.npy",
        "test":  d / f"S{subject_id:02d}_test.npy",
    }
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(
                f"File mancante: {p}. Usa data_backend='moabb' o 'braindecode'."
            )
    tr = np.load(files["train"], allow_pickle=True).item()
    te = np.load(files["test"],  allow_pickle=True).item()
    return {
        "X_train": tr["X"].astype(np.float32), "y_train": tr["y"].astype(np.int64),
        "X_test":  te["X"].astype(np.float32), "y_test":  te["y"].astype(np.int64),
    }


def load_subject(subject_id: int, cfg: dict) -> dict:
    backend = cfg.get("data_backend", "moabb")
    if backend == "moabb":
        return _load_bcic2a_moabb(subject_id, cfg)
    if backend == "braindecode":
        return _load_bcic2a_braindecode(subject_id, cfg)
    if backend == "numpy":
        return _load_bcic2a_numpy(subject_id, cfg)
    raise ValueError(f"data_backend non riconosciuto: '{backend}'")


def build_subject_cache(cfg: dict) -> dict:
    """Pre-carica tutti i soggetti in RAM (consigliato con seed multipli)."""
    n = cfg.get("n_subjects", 9)
    _div()
    print(f"[Cache] Carico {n} soggetti ({cfg.get('data_backend', 'moabb')})...")
    _div()
    cache = {}
    for s in range(1, n + 1):
        print(f"  Soggetto {s}/{n}...", end=" ", flush=True)
        cache[s] = load_subject(s, cfg)
        print("OK")
    _div()
    return cache


# ════════════════════════════════════════════════════════════
# 2. Normalizzazione per canale
#
# TCFormer paper, sezione "Input representation":
#   x'_i = (x_i - mu_i) / sigma_i
#   mu_i, sigma_i calcolati su TUTTI i training sample e time point
#   del canale i-esimo  → axis=(0, 2) su [N, C, T]
#
# Wimpff 2024, sezione "2.4 Training":
#   "normalize each channel to zero mean and unit deviation"
#   Statistiche SOLO dal training set, applicate identicamente a test.
# ════════════════════════════════════════════════════════════

def channel_normalize(
    X_tr: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu  = X_tr.mean(axis=(0, 2), keepdims=True).astype(np.float32)  # [1, C, 1]
    std = X_tr.std(axis=(0, 2),  keepdims=True).astype(np.float32) + 1e-8
    return (X_tr - mu) / std, (X_te - mu) / std


# ════════════════════════════════════════════════════════════
# 3. S&R Augmentation
#
# TCFormer paper, sezione "Data augmentation":
#   N_s=8 segmenti non sovrapposti per trial.
#   Ricostruzione: N_s frammenti della STESSA classe,
#   ordine temporale originale preservato.
#   m_A = m → raddoppia il training set.
# ════════════════════════════════════════════════════════════

def augment_sr(
    X:          np.ndarray,
    y:          np.ndarray,
    n_segments: int = 8,
    multiplier: int = 1,
    rng:        Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    N, C, T = X.shape
    seg_len  = T // n_segments
    cls_idx  = {c: np.where(y == c)[0] for c in np.unique(y)}

    synth_X, synth_y = [], []
    for i in range(N):
        cls = y[i]
        cands = cls_idx[cls]
        if len(cands) < 2:
            continue
        for _ in range(multiplier):
            donors    = rng.choice(cands, size=n_segments, replace=True)
            new_trial = X[i].copy()
            for s, d in enumerate(donors):
                start = s * seg_len
                end   = (start + seg_len) if s < n_segments - 1 else T
                new_trial[:, start:end] = X[d, :, start:end]
            synth_X.append(new_trial)
            synth_y.append(cls)

    if not synth_X:
        return X, y
    X_aug = np.concatenate([X, np.stack(synth_X).astype(np.float32)], axis=0)
    y_aug = np.concatenate([y, np.array(synth_y, dtype=y.dtype)],      axis=0)
    perm  = rng.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]


# ════════════════════════════════════════════════════════════
# 4. Dataset PyTorch
# ════════════════════════════════════════════════════════════

class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self) -> int:         return len(self.y)
    def __getitem__(self, i: int):    return self.X[i], self.y[i]


# ════════════════════════════════════════════════════════════
# 5. LR Scheduler: warmup lineare + cosine decay
#
# Wimpff 2024, sezione "2.4 Training":
#   "linear warmup of 20 epochs, followed by cosine decay"
#   "for cross-subject experiments, warmup reduced to 3 epochs"
#
# TCFormer paper, sezione "Experimental setup":
#   "linear warm-up over the first 20 epochs, followed by
#    cosine decay; warm-up adjusted to 3 epochs" (cross-subject)
# ════════════════════════════════════════════════════════════

def make_scheduler(
    optimizer:     torch.optim.Optimizer,
    n_epochs:      int,
    warmup_epochs: int,
) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
            max(1, n_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════
# 6. DataLoaders per un fold LOSO
#
# Protocollo:
#   train  = sessione 1 di train_subs (8 soggetti)
#   test   = sessione 2 di test_sub
#   norm   = mu/sigma calcolati SOLO sul training pool
#   aug    = S&R offline (m_A = m, raddoppia il train set)
# ════════════════════════════════════════════════════════════

def build_loaders(
    train_subs: list[int],
    test_sub:   int,
    cfg:        dict,
    cache:      Optional[dict] = None,
    rng:        Optional[np.random.Generator] = None,
) -> tuple[DataLoader, DataLoader]:
    if rng is None:
        rng = np.random.default_rng()

    def _get(s: int) -> dict:
        return cache[s] if (cache and s in cache) else load_subject(s, cfg)

    # Costruzione training pool
    X_tr = np.concatenate([_get(s)["X_train"] for s in train_subs], axis=0)
    y_tr = np.concatenate([_get(s)["y_train"] for s in train_subs], axis=0)
    perm = rng.permutation(len(y_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]

    # Test set
    d_te = _get(test_sub)
    X_te, y_te = d_te["X_test"], d_te["y_test"]

    # Normalizzazione per canale (statistiche dal training pool)
    X_tr, X_te = channel_normalize(X_tr, X_te)

    # S&R augmentation
    if cfg.get("use_sr", True):
        X_tr, y_tr = augment_sr(
            X_tr, y_tr,
            n_segments = cfg.get("n_segments", 8),
            multiplier = cfg.get("sr_multiplier", 1),
            rng        = rng,
        )

    bs = cfg.get("batch_size", 64)
    tr_ld = DataLoader(EEGDataset(X_tr, y_tr), batch_size=bs,
                       shuffle=True,  num_workers=0, pin_memory=True)
    te_ld = DataLoader(EEGDataset(X_te, y_te), batch_size=bs,
                       shuffle=False, num_workers=0, pin_memory=True)
    return tr_ld, te_ld


# ════════════════════════════════════════════════════════════
# 7. Training loop
#
# TCFormer paper, sezione "Evaluation":
#   "Test accuracy reported using the FINAL MODEL CHECKPOINT"
# Wimpff 2024, sezione "2.4 Training":
#   "train for a FIXED NUMBER OF epochs (no early stopping)"
# ════════════════════════════════════════════════════════════

def _train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        F.cross_entropy(model(X), y).backward()
        optimizer.step()
        total += X.shape[0]
    return total


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    true_all, pred_all = [], []
    for X, y in loader:
        preds = model(X.to(device)).argmax(dim=-1).cpu().numpy()
        true_all.extend(y.numpy())
        pred_all.extend(preds)
    return _metrics(np.array(true_all), np.array(pred_all))


def fit(
    model:   nn.Module,
    loader:  DataLoader,
    cfg:     dict,
    device:  torch.device,
    verbose: bool = False,
) -> nn.Module:
    """
    Addestramento con final checkpoint.
    - 125 epoche (cross-subject BCIC)
    - Adam lr=9e-4
    - Warmup lineare 3 epoche + cosine decay
    """
    n_epochs  = cfg.get("n_epochs",      125)
    lr        = cfg.get("lr",            9e-4)
    warmup    = cfg.get("warmup_epochs", 3)

    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    sch  = make_scheduler(opt, n_epochs, warmup)

    for epoch in range(1, n_epochs + 1):
        _train_epoch(model, loader, opt, device)
        sch.step()
        if verbose and (epoch % 25 == 0 or epoch == 1 or epoch == n_epochs):
            print(f"   Epoch {epoch:4d}/{n_epochs} | lr={opt.param_groups[0]['lr']:.6f}")
    return model


# ════════════════════════════════════════════════════════════
# 8. Folds LOSO
# ════════════════════════════════════════════════════════════

def generate_folds(n_subjects: int = 9) -> list[dict]:
    all_subs = list(range(1, n_subjects + 1))
    return [
        {
            "fold":       s,
            "test_sub":   s,
            "train_subs": [x for x in all_subs if x != s],
        }
        for s in all_subs
    ]


# ════════════════════════════════════════════════════════════
# 9. CSV (salvataggio incrementale + resume)
# ════════════════════════════════════════════════════════════

_CSV_FIELDS = ["fold", "test_sub", "seed", "acc", "kappa", "elapsed_s"]


def _csv_path(out_dir: str, seed: int) -> Path:
    return Path(out_dir) / f"loso_seed{seed}.csv"


def _load_done(csv_file: Path, seed: int) -> set:
    done = set()
    if not csv_file.exists():
        return done
    with open(csv_file, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(row["seed"]) == seed:
                    done.add((int(row["fold"]), seed))
            except (KeyError, ValueError):
                pass
    return done


def _append_row(csv_file: Path, row: dict) -> None:
    write_header = not csv_file.exists() or csv_file.stat().st_size == 0
    with open(csv_file, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ════════════════════════════════════════════════════════════
# 10. Singolo fold
# ════════════════════════════════════════════════════════════

def run_fold(
    fold:    dict,
    cfg:     dict,
    seed:    int,
    cache:   Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    rng    = np.random.default_rng(seed)
    device = get_device()
    t0     = time.time()

    tr_ld, te_ld = build_loaders(
        train_subs = fold["train_subs"],
        test_sub   = fold["test_sub"],
        cfg        = cfg,
        cache      = cache,
        rng        = rng,
    )
    model = build_tcformer(cfg).to(device)
    if verbose:
        print(
            f"  Fold {fold['fold']:2d} [test=S{fold['test_sub']:02d} | "
            f"train={len(fold['train_subs'])} sub, {len(tr_ld.dataset)} trial]...",
            flush=True,
        )

    model = fit(model, tr_ld, cfg, device, verbose=verbose)
    acc, kappa = evaluate(model, te_ld, device)
    elapsed    = time.time() - t0

    if verbose:
        print(
            f"  Fold {fold['fold']:2d} → Acc={acc:.2f}%  κ={kappa:.4f} "
            f"({elapsed / 60:.1f} min)"
        )
    return {
        "fold":      fold["fold"],
        "test_sub":  fold["test_sub"],
        "seed":      seed,
        "acc":       acc,
        "kappa":     kappa,
        "elapsed_s": elapsed,
    }


# ════════════════════════════════════════════════════════════
# 11. Pipeline LOSO completa
# ════════════════════════════════════════════════════════════

def run_loso(
    cfg:      dict,
    seeds:    list[int]        = (42,),
    cache:    Optional[dict]   = None,
    out_dir:  str              = "./loso_results",
    verbose:  bool             = True,
    subjects: Optional[list[int]] = None,
) -> list[dict]:
    """
    Esegue la pipeline LOSO completa con supporto a:
    - Multi-seed (paper usa 5 seed: [42, 0, 1, 2, 3])
    - Resume automatico (salta fold già completati)
    - Salvataggio incrementale CSV per seed

    Args:
        cfg:      Configurazione (vedi get_default_cfg()).
        seeds:    Seed multipli. Default [42].
        cache:    Cache soggetti pre-caricati.
        out_dir:  Directory output.
        verbose:  Mostra progresso per epoch ogni 25.
        subjects: Lista soggetti (None = tutti e 9).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    all_folds = generate_folds(cfg.get("n_subjects", 9))

    if subjects:
        folds = [f for f in all_folds if f["test_sub"] in subjects]
        if not folds:
            raise ValueError(f"Nessun fold trovato per subjects={subjects}.")
        print(f"[LOSO] Soggetti selezionati: {subjects} → {len(folds)} fold")
    else:
        folds = all_folds

    all_results: list[dict] = []

    for seed in seeds:
        csv_file = _csv_path(out_dir, seed)
        done     = _load_done(csv_file, seed)
        if done:
            print(f"[Resume] Seed={seed}: {len(done)} fold già completati.")

        _div()
        print(f"LOSO | seed={seed} | device={get_device()} | epochs={cfg.get('n_epochs',125)}")
        _div()

        for fold in folds:
            key = (fold["fold"], seed)
            if key in done:
                print(f"  Fold {fold['fold']:2d} → già completato, skip.")
                continue
            result = run_fold(fold, cfg, seed, cache=cache, verbose=verbose)
            all_results.append(result)
            _append_row(csv_file, result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Stampa summary del seed corrente
        seed_results = _read_csv(csv_file, seed)
        _print_summary(seed_results, title=f"LOSO Summary — seed={seed}")

    if len(seeds) > 1:
        _print_multiseed(all_results, list(seeds), out_dir)

    return all_results


# ════════════════════════════════════════════════════════════
# 12. Report
# ════════════════════════════════════════════════════════════

def _read_csv(csv_file: Path, seed: int) -> list[dict]:
    results = []
    if not csv_file.exists():
        return results
    with open(csv_file, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(row["seed"]) == seed:
                    results.append({
                        "fold":     int(row["fold"]),
                        "test_sub": int(row["test_sub"]),
                        "seed":     int(row["seed"]),
                        "acc":      float(row["acc"]),
                        "kappa":    float(row["kappa"]),
                    })
            except (KeyError, ValueError):
                pass
    return sorted(results, key=lambda r: r["fold"])


def _print_summary(results: list[dict], title: str = "LOSO Summary") -> None:
    if not results:
        return
    _div()
    print(title)
    _div()
    header = f"{'Fold':>5}  {'TestS':>5}  {'Acc':>8}  {'Kappa':>7}"
    print(header)
    print("─" * len(header))
    accs, kappas = [], []
    for r in results:
        print(f"{r['fold']:>5}  S{r['test_sub']:02d}   {r['acc']:>7.2f}%  {r['kappa']:>7.4f}")
        accs.append(r["acc"])
        kappas.append(r["kappa"])
    print("─" * len(header))
    print(f"{'MEAN':>5}  {'':>5}  {np.mean(accs):>7.2f}%  {np.mean(kappas):>7.4f}")
    print(f"{'STD':>5}  {'':>5}  {np.std(accs):>7.2f}   {np.std(kappas):>7.4f}")
    _div()


def _print_multiseed(
    all_results: list[dict],
    seeds:       list[int],
    out_dir:     str,
) -> None:
    """
    Aggrega i risultati su N seed per soggetto, poi media globale.
    Protocollo TCFormer: "for each subject, compute average accuracy
    across multiple runs; final results averaged across all subjects".
    """
    _div()
    print(f"Multi-Seed Summary ({len(seeds)} seed: {seeds})")
    _div()

    by_sub: dict[int, list[float]] = defaultdict(list)
    for r in all_results:
        by_sub[r["test_sub"]].append(r["acc"])

    sub_means: list[float] = []
    header = f"{'SubID':>6}  {'Acc mean':>9}  {'±std':>6}  {'N':>4}"
    print(header)
    print("─" * len(header))
    for sid in sorted(by_sub):
        accs = by_sub[sid]
        m, s = float(np.mean(accs)), float(np.std(accs))
        print(f"  S{sid:02d}   {m:>8.2f}%  ±{s:>5.2f}  {len(accs):>4}")
        sub_means.append(m)
    print("─" * len(header))
    print(f"{'GRAND MEAN':>10}  {np.mean(sub_means):>8.2f}%")
    _div()

    agg = Path(out_dir) / "loso_multiseed_summary.csv"
    with open(agg, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "acc_mean", "acc_std", "n_runs"])
        for sid in sorted(by_sub):
            accs = by_sub[sid]
            w.writerow([sid, round(float(np.mean(accs)), 4),
                        round(float(np.std(accs)), 4), len(accs)])
        w.writerow(["ALL", round(float(np.mean(sub_means)), 4),
                    round(float(np.std(sub_means)), 4), len(sub_means)])
    print(f"[CSV] Summary multi-seed salvato → {agg}")


# ════════════════════════════════════════════════════════════
# 13. Configurazione di default
#
# Valori da:
#   - TCFormer paper, Tabella 1 (architettura)
#   - TCFormer paper, sezione "Experimental setup" (training)
#   - Wimpff 2024, sezione "2.4 Training" (warmup, norm)
# ════════════════════════════════════════════════════════════

def get_default_cfg() -> dict:
    return {
        # Dataset
        "n_subjects":   9,
        "n_classes":    4,
        "n_channels":   22,
        "sfreq":        250,
        "fmin":         4.0,
        "fmax":         40.0,
        "tmin":         0.0,
        "tmax":         4.0,
        "data_backend": "moabb",
        "data_dir":     "./data",

        # S&R Augmentation (TCFormer paper, N_s=8, m_A=m)
        "use_sr":        True,
        "n_segments":    8,
        "sr_multiplier": 1,

        # Training
        "n_epochs":      125,   # cross-subject BCIC (TCFormer + Wimpff)
        "lr":            9e-4,  # Adam lr=0.0009 (TCFormer paper)
        "warmup_epochs": 3,     # cross-subject (Wimpff 2024, sezione 2.4)
        "batch_size":    64,

        # Architettura TCFormer (Tabella 1)
        "F1":                  32,
        "temp_kernel_lengths": (20, 32, 64),
        "D":                   2,
        "pool_length_1":       8,
        "pool_length_2":       7,
        "dropout_conv":        0.4,
        "d_group":             16,
        "trans_depth":         2,
        "q_heads":             4,
        "kv_heads":            2,
        "trans_dropout":       0.4,
        "drop_path_max":       0.25,
        "ffn_expansion":       2,
        "tcn_depth":           2,
        "tcn_kernel":          4,
        "tcn_dropout":         0.3,
        "classifier_max_norm": 0.25,
    }


# ════════════════════════════════════════════════════════════
# 14. CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TCFormer LOSO Pipeline (paper-faithful)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="Seed da usare. Paper usa 5: --seeds 42 0 1 2 3",
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="Soggetti da testare (default tutti e 9).",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Esegui solo il fold N (1-9).",
    )
    parser.add_argument(
        "--backend", default="moabb",
        choices=["moabb", "braindecode", "numpy"],
    )
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--out_dir",  default="./loso_results")
    parser.add_argument(
        "--no_aug", action="store_true",
        help="Disabilita S&R augmentation.",
    )
    parser.add_argument(
        "--cache", action="store_true",
        help="Pre-carica tutti i soggetti in RAM (consigliato con seed multipli).",
    )
    parser.add_argument("--epochs",      type=int,   default=125)
    parser.add_argument("--trans_depth", type=int,   default=2,
                        help="N=5 per migliori risultati cross-subject su HGD.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg                 = get_default_cfg()
    cfg["data_backend"] = args.backend
    cfg["data_dir"]     = args.data_dir
    cfg["use_sr"]       = not args.no_aug
    cfg["n_epochs"]     = args.epochs
    cfg["trans_depth"]  = args.trans_depth

    subject_cache = build_subject_cache(cfg) if args.cache else None

    if args.fold is not None:
        all_folds = generate_folds(cfg["n_subjects"])
        fi = next((f for f in all_folds if f["fold"] == args.fold), None)
        if fi is None:
            raise ValueError(f"Fold {args.fold} non valido (1-{cfg['n_subjects']}).")
        for seed in args.seeds:
            print(run_fold(fi, cfg, seed, cache=subject_cache, verbose=True))
    else:
        run_loso(
            cfg,
            seeds    = args.seeds,
            cache    = subject_cache,
            out_dir  = args.out_dir,
            verbose  = args.verbose,
            subjects = args.subjects,
        )
