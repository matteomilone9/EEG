# pipelines.py — Pipeline WS, LOSO, LOSO+FT (singolo seed e multi-seed)
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score

from config import get_subject_cfg, set_seed, HARD_SUBJECTS
from preprocessing import load_subject, preprocess_subject, build_loso_cache
from augmentation import MMDataset, make_dummy_gaf
from model import build_model
from trainer import DistillationTrainer, FineTuner, evaluate


# ── Within-Subject ───────────────────────────────────────────

def run_subject(sub_id, seed=42, verbose=True):
    cfg = get_subject_cfg(sub_id)
    if verbose:
        print(f"\n{'─'*55}\nSoggetto {sub_id} | seed={seed}\n{'─'*55}")
    X_tr_raw, y_tr, X_te_raw, y_te = load_subject(sub_id, cfg)
    X_tr_t, X_te_t, X_tr_g, X_te_g = preprocess_subject(X_tr_raw, X_te_raw, cfg)
    tr_ds = MMDataset(X_tr_t, X_tr_g, y_tr, augment=True, aug_prob=cfg['aug_prob'])
    te_ds = MMDataset(X_te_t, X_te_g, y_te, augment=False, aug_prob=0.0)
    tr_ld = DataLoader(tr_ds, cfg['batch_size'], shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, cfg['batch_size'], shuffle=False, num_workers=0)
    set_seed(seed)
    model   = build_model(cfg)
    trainer = DistillationTrainer(model, cfg)
    trainer.fit(tr_ld, te_ld, seed=seed)
    y_true, y_pred = evaluate(trainer.model, te_ld, cfg['device'], cfg)
    acc   = accuracy_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred)
    if verbose:
        tta_tag = f" (TTA×{cfg['n_tta']})" if cfg['n_tta'] > 1 else ""
        print(f"\nS{sub_id:02d} seed={seed} → Acc: {acc:.2f}%{tta_tag} | Kappa: {kappa:.4f}")
    return acc, kappa


def run_subject_multiseed(sub_id):
    cfg = get_subject_cfg(sub_id)
    seeds = cfg['seeds']
    print(f"\n{'═'*65}\n[WS] Soggetto {sub_id} | Multi-seed ({len(seeds)} run): {seeds}\n{'═'*65}")
    X_tr_raw, y_tr, X_te_raw, y_te = load_subject(sub_id, cfg)
    X_tr_t, X_te_t, X_tr_g, X_te_g = preprocess_subject(X_tr_raw, X_te_raw, cfg)
    tr_ds = MMDataset(X_tr_t, X_tr_g, y_tr, augment=True, aug_prob=cfg['aug_prob'])
    te_ds = MMDataset(X_te_t, X_te_g, y_te, augment=False, aug_prob=0.0)
    tr_ld = DataLoader(tr_ds, cfg['batch_size'], shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, cfg['batch_size'], shuffle=False, num_workers=0)
    seed_accs, seed_kappas = [], []
    for i, seed in enumerate(seeds):
        print(f"\n ── Seed {i+1}/{len(seeds)}: {seed} ──")
        set_seed(seed)
        model   = build_model(cfg)
        trainer = DistillationTrainer(model, cfg)
        trainer.fit(tr_ld, te_ld, seed=seed)
        y_true, y_pred = evaluate(trainer.model, te_ld, cfg['device'], cfg)
        acc   = accuracy_score(y_true, y_pred) * 100
        kappa = cohen_kappa_score(y_true, y_pred)
        seed_accs.append(acc); seed_kappas.append(kappa)
        print(f" ✓ seed={seed} → Acc: {acc:.2f}% | Kappa: {kappa:.4f}")
        del model, trainer; torch.cuda.empty_cache()
    mean_acc, std_acc = np.mean(seed_accs), np.std(seed_accs)
    mean_k,   std_k   = np.mean(seed_kappas), np.std(seed_kappas)
    print(f"\n 📊 S{sub_id:02d} [WS] → {mean_acc:.2f} ± {std_acc:.2f}% | κ={mean_k:.4f} ± {std_k:.4f}")
    return mean_acc, std_acc, mean_k, std_k, seed_accs


# ── LOSO puro ────────────────────────────────────────────────

def run_loso_fold(target_sub, cache, seed=42, verbose=True):
    cfg = get_subject_cfg(target_sub)
    if verbose:
        print(f"\n{'─'*55}\n[LOSO] Fold S{target_sub:02d} (test) | seed={seed}\n{'─'*55}")
    tr_X_list, tr_y_list = [], []
    for s, data in cache.items():
        if s == target_sub: continue
        X_t, y_t = data['T']; X_e, y_e = data['E']
        tr_X_list += [X_t, X_e]; tr_y_list += [y_t, y_e]
    X_tr = np.concatenate(tr_X_list); y_tr = np.concatenate(tr_y_list)
    X_te, y_te = cache[target_sub]['E']
    tr_ds = MMDataset(X_tr, make_dummy_gaf(len(X_tr)), y_tr, augment=True, aug_prob=cfg['aug_prob'])
    te_ds = MMDataset(X_te, make_dummy_gaf(len(X_te)), y_te, augment=False, aug_prob=0.0)
    tr_ld = DataLoader(tr_ds, cfg['batch_size'], shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, cfg['batch_size'], shuffle=False, num_workers=0)
    set_seed(seed)
    model   = build_model(cfg)
    trainer = DistillationTrainer(model, cfg)
    trainer.fit(tr_ld, te_ld, seed=seed)
    y_true, y_pred = evaluate(trainer.model, te_ld, cfg['device'], cfg)
    acc   = accuracy_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred)
    if verbose:
        print(f"\n[LOSO] S{target_sub:02d} seed={seed} → Acc: {acc:.2f}% | Kappa: {kappa:.4f}")
    return acc, kappa


def run_loso_fold_multiseed(target_sub, cache):
    cfg = get_subject_cfg(target_sub); seeds = cfg['seeds']
    print(f"\n{'═'*65}\n[LOSO] Fold S{target_sub:02d} | Multi-seed ({len(seeds)} run): {seeds}\n{'═'*65}")
    seed_accs, seed_kappas = [], []
    for i, seed in enumerate(seeds):
        print(f"\n ── Seed {i+1}/{len(seeds)}: {seed} ──")
        acc, kappa = run_loso_fold(target_sub, cache, seed=seed, verbose=False)
        seed_accs.append(acc); seed_kappas.append(kappa)
        print(f" ✓ seed={seed} → Acc: {acc:.2f}% | Kappa: {kappa:.4f}")
        torch.cuda.empty_cache()
    mean_acc, std_acc = np.mean(seed_accs), np.std(seed_accs)
    mean_k,   std_k   = np.mean(seed_kappas), np.std(seed_kappas)
    print(f"\n 📊 S{target_sub:02d} [LOSO] → {mean_acc:.2f} ± {std_acc:.2f}% | κ={mean_k:.4f} ± {std_k:.4f}")
    return mean_acc, std_acc, mean_k, std_k, seed_accs


# ── LOSO + Fine-Tuning ───────────────────────────────────────

def run_loso_ft_fold(target_sub, cache, seed=42, verbose=True):
    cfg = get_subject_cfg(target_sub)
    if verbose:
        print(f"\n{'─'*55}\n[LOSO+FT] S{target_sub:02d} | seed={seed}\n{'─'*55}")

    # Fase 1 — Pre-training LOSO
    if verbose: print(f"\n [Fase 1] Pre-training LOSO (tutti tranne S{target_sub:02d})...")
    tr_X_list, tr_y_list = [], []
    for s, data in cache.items():
        if s == target_sub: continue
        X_t, y_t = data['T']; X_e, y_e = data['E']
        tr_X_list += [X_t, X_e]; tr_y_list += [y_t, y_e]
    X_tr_loso = np.concatenate(tr_X_list); y_tr_loso = np.concatenate(tr_y_list)
    X_te, y_te = cache[target_sub]['E']
    loso_tr_ds = MMDataset(X_tr_loso, make_dummy_gaf(len(X_tr_loso)),
                           y_tr_loso, augment=True, aug_prob=cfg['aug_prob'])
    loso_te_ds = MMDataset(X_te, make_dummy_gaf(len(X_te)), y_te, augment=False, aug_prob=0.0)
    loso_tr_ld = DataLoader(loso_tr_ds, cfg['batch_size'], shuffle=True,  num_workers=0)
    loso_te_ld = DataLoader(loso_te_ds, cfg['batch_size'], shuffle=False, num_workers=0)
    set_seed(seed)
    model   = build_model(cfg)
    trainer = DistillationTrainer(model, cfg)
    trainer.fit(loso_tr_ld, loso_te_ld, seed=seed)
    y_true, y_pred = evaluate(trainer.model, loso_te_ld, cfg['device'], cfg)
    acc_loso = accuracy_score(y_true, y_pred) * 100
    if verbose: print(f"\n [Fase 1] LOSO acc S{target_sub:02d}: {acc_loso:.2f}%")

    # Fase 2 — Fine-tuning Within-Subject
    if verbose: print(f"\n [Fase 2] Fine-tuning su S{target_sub:02d} (sessione T → val E)...")
    X_tr_raw, y_tr_ws, X_te_raw, y_te_ws = load_subject(target_sub, cfg)
    X_tr_ws_t, X_te_ws_t, X_tr_ws_g, X_te_ws_g = preprocess_subject(X_tr_raw, X_te_raw, cfg)
    ft_tr_ds = MMDataset(X_tr_ws_t, X_tr_ws_g, y_tr_ws, augment=True, aug_prob=cfg['aug_prob'])
    ft_te_ds = MMDataset(X_te_ws_t, X_te_ws_g, y_te_ws, augment=False, aug_prob=0.0)
    ft_tr_ld = DataLoader(ft_tr_ds, cfg['batch_size'], shuffle=True,  num_workers=0)
    ft_te_ld = DataLoader(ft_te_ds, cfg['batch_size'], shuffle=False, num_workers=0)
    for p in model.parameters(): p.requires_grad = True
    fine_tuner = FineTuner(model, cfg)
    fine_tuner.fit(ft_tr_ld, ft_te_ld, seed=seed)
    y_true, y_pred = evaluate(model, ft_te_ld, cfg['device'], cfg)
    acc_ft = accuracy_score(y_true, y_pred) * 100
    kappa  = cohen_kappa_score(y_true, y_pred)
    if verbose:
        print(f"\n [LOSO+FT] S{target_sub:02d} seed={seed}")
        print(f"  LOSO puro: {acc_loso:.2f}%")
        print(f"  Dopo FT:   {acc_ft:.2f}% (Δ={acc_ft-acc_loso:+.2f}%)")
        print(f"  Kappa:     {kappa:.4f}")
    return acc_ft, kappa, acc_loso


def run_loso_ft_fold_multiseed(target_sub, cache):
    cfg = get_subject_cfg(target_sub); seeds = cfg['seeds']
    print(f"\n{'═'*65}\n[LOSO+FT] S{target_sub:02d} | Multi-seed ({len(seeds)} run): {seeds}\n{'═'*65}")
    seed_accs, seed_kappas, seed_loso_accs = [], [], []
    for i, seed in enumerate(seeds):
        print(f"\n ── Seed {i+1}/{len(seeds)}: {seed} ──")
        acc_ft, kappa, acc_loso = run_loso_ft_fold(target_sub, cache, seed=seed, verbose=False)
        seed_accs.append(acc_ft); seed_kappas.append(kappa); seed_loso_accs.append(acc_loso)
        print(f" ✓ seed={seed} → LOSO: {acc_loso:.2f}% | FT: {acc_ft:.2f}% (Δ={acc_ft-acc_loso:+.2f}%) | κ={kappa:.4f}")
        torch.cuda.empty_cache()
    mean_acc,  std_acc  = np.mean(seed_accs),      np.std(seed_accs)
    mean_k,    std_k    = np.mean(seed_kappas),     np.std(seed_kappas)
    mean_loso, std_loso = np.mean(seed_loso_accs),  np.std(seed_loso_accs)
    print(f"\n 📊 S{target_sub:02d} [LOSO+FT]")
    print(f"  LOSO puro: {mean_loso:.2f} ± {std_loso:.2f}%")
    print(f"  Dopo FT:   {mean_acc:.2f} ± {std_acc:.2f}% (Δ={mean_acc-mean_loso:+.2f}%)")
    print(f"  Kappa:     {mean_k:.4f} ± {std_k:.4f}")
    return mean_acc, std_acc, mean_k, std_k, seed_accs, mean_loso