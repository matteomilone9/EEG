"""
loso_pipeline.py — Pipeline LOSO per TCFormer EEG-Only
=======================================================
Replica il più fedelmente possibile il repository ufficiale TCFormer per BCIC IV-2a LOSO.

Modifiche integrate rispetto alla versione precedente:
  - Loader dati allineato al repo: Braindecode + MOABBDataset + create_windows_from_events
  - Nessun band-pass per BCIC IV-2a (low_cut=None, high_cut=None)
  - Scaling x1e6 prima del resample
  - StandardScaler identico al repo ufficiale (_z_scale_tvt)
  - interaug online nel collate_fn
  - Adam con betas=(0.5, 0.999) e weight_decay=1e-3
  - batch_size=48
  - trans_depth default=5 per LOSO come nel config ufficiale tcformer.yaml
  - warmup 3 epoche + cosine decay
  - final checkpoint, no early stopping
"""

from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

from model import build_tcformer


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
    acc = accuracy_score(y_true, y_pred) * 100.0
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, kappa


def interaug(batch):
    x, y = batch
    new_samples = torch.zeros_like(x)
    new_labels = torch.zeros_like(y)
    current = 0
    n_chunks = 9 if new_samples.shape[-1] == 1125 else (8 if new_samples.shape[-1] % 8 == 0 else 7)
    for cls in torch.unique(y):
        x_cls = x[y == cls]
        if len(x_cls) == 0:
            continue
        chunks = torch.cat(torch.chunk(x_cls, chunks=n_chunks, dim=-1))
        indices = torch.randint(0, len(x_cls), size=(len(x_cls), n_chunks), device=x_cls.device)
        for idx in indices:
            idx = idx + torch.arange(0, chunks.shape[0], len(x_cls), device=x_cls.device)
            new_sample = chunks[idx]
            new_sample = new_sample.permute(1, 0, 2).reshape(1, x_cls.shape[1], x_cls.shape[2])
            new_samples[current] = new_sample.squeeze(0)
            new_labels[current] = cls
            current += 1
    combined_x = torch.cat((x, new_samples), dim=0)
    combined_y = torch.cat((y, new_labels), dim=0)
    perm = torch.randperm(len(combined_x), device=combined_x.device)
    return combined_x[perm], combined_y[perm]


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def make_collate_fn(cfg: dict):
    def collate(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs)
        y = torch.tensor(ys, dtype=torch.long)
        if cfg.get("interaug", True):
            x, y = interaug((x, y))
        return x, y
    return collate


class _Scale:
    """Callable class per scalare i dati raw MNE.
    Usare una classe callable (invece di una funzione o lambda) evita
    ambiguità nel meccanismo interno di braindecode che distingue tra
    stringhe (metodi MNE) e callable (funzioni custom).
    """
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, raw):
        raw._data *= self.factor
        return raw


def load_bcic4(subject_ids: list[int], preprocessing_dict: Dict, verbose: str = "WARNING"):
    try:
        from braindecode.datasets import MOABBDataset
        from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess
    except ImportError as e:
        raise ImportError("Installa braindecode e moabb: pip install braindecode moabb") from e

    dataset = MOABBDataset("BNCI2014_001", subject_ids=subject_ids)

    # Step 1: preprocessori MNE puri (niente scaling qui)
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose),
    ]
    l_freq = preprocessing_dict.get("low_cut", None)
    h_freq = preprocessing_dict.get("high_cut", None)
    if l_freq is not None or h_freq is not None:
        preprocessors.append(Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, verbose=verbose))
    preprocess(dataset, preprocessors)

    # Step 2: scaling x1e6 direttamente su raw._data, fuori da Preprocessor
    for ds in dataset.datasets:
        ds.raw._data *= 1e6

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        preload=False,
    )

def load_subject(subject_id: int, cfg: dict) -> dict:
    preproc = {
        "sfreq": cfg.get("sfreq", 250),
        "low_cut": cfg.get("low_cut", None),
        "high_cut": cfg.get("high_cut", None),
        "start": cfg.get("start", 0.0),
        "stop": cfg.get("stop", 0.0),
    }
    dataset = load_bcic4([subject_id], preprocessing_dict=preproc)
    splitted_ds = dataset.split("session")


    train_dataset = splitted_ds["0train"]
    test_dataset  = splitted_ds["1test"]

    def _extract(concat_ds):
        X_list, y_list = [], []
        for ds in concat_ds.datasets:
            data   = np.array([ds[i][0] for i in range(len(ds))])
            labels = np.array([ds[i][1] for i in range(len(ds))])
            X_list.append(data)
            y_list.append(labels)
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
    
    X,      y      = _extract(train_dataset)
    X_test, y_test = _extract(test_dataset)

    return {
        "X_train": X.astype(np.float32),
        "y_train": y.astype(np.int64),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
    }


def build_subject_cache(cfg: dict) -> dict:
    n = cfg.get("n_subjects", 9)
    _div()
    print(f"[Cache] Carico {n} soggetti (braindecode+MOABB)...")
    _div()
    cache = {}
    for s in range(1, n + 1):
        print(f"  Soggetto {s}/{n}...", end=" ", flush=True)
        cache[s] = load_subject(s, cfg)
        print("OK")
    _div()
    return cache


def channel_normalize_tvt(X: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    s, c, t = X.shape
    X_2d = X.transpose(1, 0, 2).reshape(c, -1).T
    X_val_2d = X_val.transpose(1, 0, 2).reshape(c, -1).T
    X_test_2d = X_test.transpose(1, 0, 2).reshape(c, -1).T
    sc = StandardScaler().fit(X_2d)
    X = sc.transform(X_2d).T.reshape(c, s, t).transpose(1, 0, 2)
    X_val = sc.transform(X_val_2d).T.reshape(c, X_val.shape[0], t).transpose(1, 0, 2)
    X_test = sc.transform(X_test_2d).T.reshape(c, X_test.shape[0], t).transpose(1, 0, 2)
    return X.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)


def augment_sr(X: np.ndarray, y: np.ndarray, n_segments: int = 8, multiplier: int = 1, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    N, C, T = X.shape
    seg_len = T // n_segments
    cls_idx = {c: np.where(y == c)[0] for c in np.unique(y)}
    synth_X, synth_y = [], []
    for i in range(N):
        cls = y[i]
        cands = cls_idx[cls]
        if len(cands) < 2:
            continue
        for _ in range(multiplier):
            donors = rng.choice(cands, size=n_segments, replace=True)
            new_trial = X[i].copy()
            for s, d in enumerate(donors):
                start = s * seg_len
                end = (start + seg_len) if s < n_segments - 1 else T
                new_trial[:, start:end] = X[d, :, start:end]
            synth_X.append(new_trial)
            synth_y.append(cls)
    if not synth_X:
        return X, y
    X_aug = np.concatenate([X, np.stack(synth_X).astype(np.float32)], axis=0)
    y_aug = np.concatenate([y, np.array(synth_y, dtype=y.dtype)], axis=0)
    perm = rng.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]


def make_scheduler(optimizer: torch.optim.Optimizer, n_epochs: int, warmup_epochs: int) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, n_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def build_loaders(train_subs: list[int], test_sub: int, cfg: dict, cache: Optional[dict] = None, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    def _get(s: int) -> dict:
        return cache[s] if (cache and s in cache) else load_subject(s, cfg)

    X_tr = np.concatenate([_get(s)["X_train"] for s in train_subs], axis=0)
    y_tr = np.concatenate([_get(s)["y_train"] for s in train_subs], axis=0)
    X_val = np.concatenate([_get(s)["X_test"] for s in train_subs], axis=0)
    y_val = np.concatenate([_get(s)["y_test"] for s in train_subs], axis=0)
    d_te = _get(test_sub)
    X_te, y_te = d_te["X_test"], d_te["y_test"]

    X_tr, X_val, X_te = channel_normalize_tvt(X_tr, X_val, X_te)
    if cfg.get("use_sr", True):
        X_tr, y_tr = augment_sr(X_tr, y_tr, n_segments=cfg.get("n_segments", 8), multiplier=cfg.get("sr_multiplier", 1), rng=rng)
    perm = rng.permutation(len(y_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]

    bs = cfg.get("batch_size", 48)
    nw = cfg.get("num_workers", 0)
    pin = torch.cuda.is_available()
    tr_ld = DataLoader(EEGDataset(X_tr, y_tr), batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, collate_fn=make_collate_fn(cfg), persistent_workers=(nw > 0))
    val_ld = DataLoader(EEGDataset(X_val, y_val), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    te_ld = DataLoader(EEGDataset(X_te, y_te), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    return tr_ld, val_ld, te_ld


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.shape[0]
        total_n += X.shape[0]
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    true_all, pred_all = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        preds = model(X).argmax(dim=-1).cpu().numpy()
        true_all.extend(y.numpy())
        pred_all.extend(preds)
    return _metrics(np.array(true_all), np.array(pred_all))


def fit(model: nn.Module, train_loader: DataLoader, cfg: dict, device: torch.device, verbose: bool = False) -> nn.Module:
    n_epochs = cfg.get("n_epochs", 125)
    lr = cfg.get("lr", 9e-4)
    warmup = cfg.get("warmup_epochs", 3)
    beta_1 = cfg.get("beta_1", 0.5)
    beta_2 = cfg.get("beta_2", 0.999)
    weight_decay = cfg.get("weight_decay", 1e-3)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta_1, beta_2), weight_decay=weight_decay)
    sch = make_scheduler(opt, n_epochs, warmup)
    for epoch in range(1, n_epochs + 1):
        loss = _train_epoch(model, train_loader, opt, device)
        sch.step()
        if verbose and (epoch % 25 == 0 or epoch == 1 or epoch == n_epochs):
            print(f"   Epoch {epoch:4d}/{n_epochs} | loss={loss:.4f} | lr={opt.param_groups[0]['lr']:.6f}")
    return model


def generate_folds(n_subjects: int = 9) -> list[dict]:
    all_subs = list(range(1, n_subjects + 1))
    return [{"fold": s, "test_sub": s, "train_subs": [x for x in all_subs if x != s]} for s in all_subs]


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


def run_fold(fold: dict, cfg: dict, seed: int, cache: Optional[dict] = None, verbose: bool = True) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()
    t0 = time.time()
    tr_ld, _, te_ld = build_loaders(train_subs=fold["train_subs"], test_sub=fold["test_sub"], cfg=cfg, cache=cache, rng=rng)
    model = build_tcformer(cfg).to(device)
    if verbose:
        print(f"  Fold {fold['fold']:2d} [test=S{fold['test_sub']:02d} | train={len(fold['train_subs'])} sub, {len(tr_ld.dataset)} trial]...", flush=True)
    model = fit(model, tr_ld, cfg, device, verbose=verbose)
    acc, kappa = evaluate(model, te_ld, device)
    elapsed = time.time() - t0
    if verbose:
        print(f"  Fold {fold['fold']:2d} → Acc={acc:.2f}%  κ={kappa:.4f} ({elapsed / 60:.1f} min)")
    return {"fold": fold["fold"], "test_sub": fold["test_sub"], "seed": seed, "acc": acc, "kappa": kappa, "elapsed_s": elapsed}


def run_loso(cfg: dict, seeds: list[int] = (42,), cache: Optional[dict] = None, out_dir: str = "./loso_results", verbose: bool = True, subjects: Optional[list[int]] = None) -> list[dict]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    all_folds = generate_folds(cfg.get("n_subjects", 9))
    folds = [f for f in all_folds if f["test_sub"] in subjects] if subjects else all_folds
    if subjects and not folds:
        raise ValueError(f"Nessun fold trovato per subjects={subjects}.")
    if subjects:
        print(f"[LOSO] Soggetti selezionati: {subjects} → {len(folds)} fold")

    all_results: list[dict] = []
    for seed in seeds:
        csv_file = _csv_path(out_dir, seed)
        done = _load_done(csv_file, seed)
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
        seed_results = _read_csv(csv_file, seed)
        _print_summary(seed_results, title=f"LOSO Summary — seed={seed}")
    if len(seeds) > 1:
        _print_multiseed(out_dir, list(seeds))
    return all_results


def _read_csv(csv_file: Path, seed: int) -> list[dict]:
    results = []
    if not csv_file.exists():
        return results
    with open(csv_file, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(row["seed"]) == seed:
                    results.append({"fold": int(row["fold"]), "test_sub": int(row["test_sub"]), "seed": int(row["seed"]), "acc": float(row["acc"]), "kappa": float(row["kappa"])})
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


def _print_multiseed(out_dir: str, seeds: list[int]) -> None:
    _div()
    print(f"Multi-Seed Summary ({len(seeds)} seed: {list(seeds)})")
    _div()
    by_sub: dict[int, list[float]] = defaultdict(list)
    for seed in seeds:
        csv_file = _csv_path(out_dir, seed)
        for r in _read_csv(csv_file, seed):
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
            w.writerow([sid, round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4), len(accs)])
        w.writerow(["ALL", round(float(np.mean(sub_means)), 4), round(float(np.std(sub_means)), 4), len(sub_means)])
    print(f"[CSV] Summary multi-seed salvato → {agg}")


def get_default_cfg() -> dict:
    return {
        "n_subjects": 9,
        "n_classes": 4,
        "n_channels": 22,
        "sfreq": 250,
        "low_cut": None,
        "high_cut": None,
        "start": 0.0,
        "stop": 0.0,
        "use_sr": True,
        "n_segments": 8,
        "sr_multiplier": 1,
        "interaug": True,
        "n_epochs": 125,
        "lr": 9e-4,
        "warmup_epochs": 3,
        "batch_size": 48,
        "beta_1": 0.5,
        "beta_2": 0.999,
        "weight_decay": 1e-3,
        "num_workers": 0,
        "F1": 32,
        "temp_kernel_lengths": (20, 32, 64),
        "D": 2,
        "pool_length_1": 8,
        "pool_length_2": 7,
        "dropout_conv": 0.4,
        "d_group": 16,
        "trans_depth": 5,
        "q_heads": 4,
        "kv_heads": 2,
        "trans_dropout": 0.4,
        "drop_path_max": 0.25,
        "ffn_expansion": 2,
        "tcn_depth": 2,
        "tcn_kernel": 4,
        "tcn_dropout": 0.3,
        "classifier_max_norm": 0.25,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TCFormer LOSO Pipeline (repo-aligned)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 64, 128, 256, 512])
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--out_dir", default="./loso_results")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--trans_depth", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_sr", action="store_true")
    parser.add_argument("--no_interaug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = get_default_cfg()
    cfg["n_epochs"] = args.epochs
    cfg["trans_depth"] = args.trans_depth
    cfg["batch_size"] = args.batch_size
    cfg["num_workers"] = args.num_workers
    cfg["use_sr"] = not args.no_sr
    cfg["interaug"] = not args.no_interaug

    subject_cache = build_subject_cache(cfg) if args.cache else None
    if args.fold is not None:
        all_folds = generate_folds(cfg["n_subjects"])
        fi = next((f for f in all_folds if f["fold"] == args.fold), None)
        if fi is None:
            raise ValueError(f"Fold {args.fold} non valido (1-{cfg['n_subjects']}).")
        for seed in args.seeds:
            print(run_fold(fi, cfg, seed, cache=subject_cache, verbose=True))
    else:
        run_loso(cfg, seeds=args.seeds, cache=subject_cache, out_dir=args.out_dir, verbose=args.verbose, subjects=args.subjects)
