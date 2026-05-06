"""
pipelines_loso_kd.py
====================
Pipeline LOSO (Leave-One-Subject-Out) per KD e KD-Align sul dataset BNCI2014001.

Schema fold (standard):
  - 9 soggetti totali (1..9)
  - Ogni fold: 1 soggetto come test, 1 come validation, 7 come training
  - 9 fold totali: test_sub = 1..9, val_sub = (test_sub % 9) + 1

Schema fold (multi-val, robusto):
  - Per ogni test_sub s, si usano tutti gli altri 8 soggetti come val a turno
  - Per ogni (test_sub, val_sub): train = {1..9} \ {test_sub, val_sub}
  - Accuracy finale per soggetto = media delle 8 run
  - Totale: 9 × 8 = 72 run

Normalizzazione:
  - Se use_global_norm=True (default per LOSO): μ/σ calcolati sul training pool
    (7 soggetti concatenati) e applicati identicamente a val e test.
    I dati vengono caricati RAW (raw=True in load_subject_both) senza z-score
    per-soggetto.
  - Se use_global_norm=False: usa la normalizzazione per-soggetto già applicata
    da load_subject_both (compatibilità con pipeline sub-dep).
"""

import os
import copy
import csv
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score

from config_kd import get_kd_subject_cfg, set_seed
from augmentation_kd import MMDataset
from preprocessing_kd import load_subject_both, make_gaf, zscore_per_trial
from model_kd import build_teacher_model, build_student_model, build_gaf_proj_head
from trainer_kd import (
    TeacherTrainer, KDTrainer, KDAlignTrainer,
    evaluate_teacher, evaluate_student,
)

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _printf(msg="", width=70):
    print(msg if msg else "─" * width)


def _compute_metrics(y_true, y_pred):
    acc   = accuracy_score(y_true, y_pred) * 100.0
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, kappa

# ─────────────────────────────────────────────────────────────────────────────
# CSV checkpoint: salvataggio incrementale dei risultati
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDNAMES_STANDARD = [
    "fold", "test_sub", "val_sub", "seed",
    "teacher_acc", "teacher_kappa",
    "student_acc", "student_kappa",
]

_CSV_FIELDNAMES_MULTIVAL = [
    "run_id", "test_sub", "val_sub", "seed",
    "teacher_acc", "teacher_kappa",
    "student_acc", "student_kappa",
]


def _csv_path(mode: str, scheme: str, seed: int, out_dir: str = ".") -> Path:
    """Restituisce il path del CSV per la combinazione mode/scheme/seed."""
    return Path(out_dir) / f"loso_{mode}_{scheme}_seed{seed}.csv"


def _load_done_keys_standard(csv_file: Path) -> set:
    """
    Legge il CSV standard e restituisce l'insieme delle chiavi già completate.
    Chiave = (fold, test_sub, val_sub, seed)
    """
    done = set()
    if not csv_file.exists():
        return done
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (int(row["fold"]), int(row["test_sub"]),
                       int(row["val_sub"]), int(row["seed"]))
                done.add(key)
            except (KeyError, ValueError):
                pass
    return done


def _load_done_keys_multival(csv_file: Path) -> set:
    """
    Legge il CSV multival e restituisce l'insieme delle chiavi già completate.
    Chiave = (test_sub, val_sub, seed)
    """
    done = set()
    if not csv_file.exists():
        return done
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (int(row["test_sub"]), int(row["val_sub"]), int(row["seed"]))
                done.add(key)
            except (KeyError, ValueError):
                pass
    return done


def _append_row(csv_file: Path, fieldnames: list, row: dict):
    """
    Aggiunge una riga al CSV. Crea l'header se il file non esiste ancora.
    Thread-safe per scritture sequenziali (non parallele).
    """
    write_header = not csv_file.exists() or csv_file.stat().st_size == 0
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _get_loso_cfg(cfg_override: dict | None = None) -> dict:
    cfg = get_kd_subject_cfg(1)
    cfg["split_train"]      = False
    cfg["use_global_norm"]  = False   # MODIFICA: da True a False
    if cfg_override:
        cfg.update(cfg_override)
    return cfg


def _generate_loso_folds(n_subjects: int = 9) -> list[dict]:
    """
    Genera i 9 fold LOSO standard con schema circolare:
      test_sub  = s  (s = 1..9)
      val_sub   = (s % n_subjects) + 1
      train_subs = tutti gli altri
    """
    folds = []
    all_subs = list(range(1, n_subjects + 1))
    for s in all_subs:
        test_sub   = s
        val_sub    = (s % n_subjects) + 1
        train_subs = [x for x in all_subs if x != test_sub and x != val_sub]
        folds.append({
            "fold":       s,
            "test_sub":   test_sub,
            "val_sub":    val_sub,
            "train_subs": train_subs,
        })
    return folds


def _generate_loso_multival_runs(n_subjects: int = 9) -> list[dict]:
    """
    Genera 72 run per lo schema multi-val:
      per ogni test_sub, tutti gli altri 8 soggetti vengono usati come val a turno.
      train_subs = {1..9} \ {test_sub, val_sub}

    Returns list di dict:
      {"test_sub": int, "val_sub": int, "train_subs": list, "run_id": int}
    """
    all_subs = list(range(1, n_subjects + 1))
    runs = []
    run_id = 1
    for test_sub in all_subs:
        others = [s for s in all_subs if s != test_sub]
        for val_sub in others:
            train_subs = [s for s in others if s != val_sub]
            runs.append({
                "run_id":     run_id,
                "test_sub":   test_sub,
                "val_sub":    val_sub,
                "train_subs": train_subs,
            })
            run_id += 1
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Cache (caricamento raw opzionale)
# ─────────────────────────────────────────────────────────────────────────────

def build_loso_cache(cfg: dict) -> dict:
    """
    Precarica tutti i soggetti in memoria.

    Se cfg["use_global_norm"] è True, i dati vengono caricati RAW
    (senza z-score per-soggetto) perché la normalizzazione verrà
    applicata sul training pool in build_loso_loaders_kd.

    Se cfg["use_global_norm"] è False, usa load_subject_both standard
    (z-score per-soggetto già applicato).

    cache[sub_id] = {
        "Xt": np.ndarray (T-session, raw o z-scored),
        "yt": np.ndarray,
        "Xe": np.ndarray (E-session, raw o z-scored),
        "ye": np.ndarray,
        "Xtrg": np.ndarray (GAF T-session, calcolato sui raw),
        "Xteg": np.ndarray (GAF E-session, calcolato sui raw),
    }
    """
    use_gaf        = cfg.get("use_gaf", False)
    use_global_norm = cfg.get("use_global_norm", True)
    n_subs         = cfg["n_subjects"]
    cache          = {}

    _printf()
    norm_tag = "RAW (global norm verrà applicata per fold)" if use_global_norm \
               else "z-score per soggetto"
    print(f"[LOSO cache] Caricamento {n_subs} soggetti — normalizzazione: {norm_tag}")
    _printf()

    for s in range(1, n_subs + 1):
        print(f"  Soggetto {s}/{n_subs}...", end=" ", flush=True)

        # load_subject_both con raw=True salta lo z-score interno
        data = load_subject_both(s, cfg, raw=use_global_norm)

        Xt = data["T"][0]
        yt = data["T"][1]
        Xe = data["E"][0]
        ye = data["E"][1]

        if use_gaf:
            # GAF si calcola sempre sui segnali raw (prima della norm globale)
            # Se use_global_norm=False i segnali sono già z-scored per-soggetto,
            # ma il GAF in sub-dep è calcolato sugli stessi → coerente.
            Xtrg = make_gaf(Xt, cfg)
            Xteg = make_gaf(Xe, cfg)
        else:
            Xtrg = np.zeros((len(yt), 1, 1, 1), dtype=np.float32)
            Xteg = np.zeros((len(ye), 1, 1, 1), dtype=np.float32)

        cache[s] = {
            "Xt": Xt, "yt": yt,
            "Xe": Xe, "ye": ye,
            "Xtrg": Xtrg, "Xteg": Xteg,
        }
        print("OK")

    _printf()
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Normalizzazione globale sul training pool
# ─────────────────────────────────────────────────────────────────────────────

def _apply_global_norm(Xt_tr, Xt_va, Xt_te):
    """
    Calcola μ/σ per canale sul training pool e applica a train, val, test.

    Xt_tr shape: (N_train, C, T)
    Statistiche calcolate su axes (0, 2) → shape (1, C, 1)

    Restituisce (Xt_tr_norm, Xt_va_norm, Xt_te_norm, mu, sigma)
    """
    mu  = Xt_tr.mean(axis=(0, 2), keepdims=True).astype(np.float32)   # (1, C, 1)
    std = Xt_tr.std(axis=(0, 2),  keepdims=True).astype(np.float32) + 1e-8
    Xt_tr_n = (Xt_tr - mu) / std
    Xt_va_n = (Xt_va - mu) / std
    Xt_te_n = (Xt_te - mu) / std
    return Xt_tr_n, Xt_va_n, Xt_te_n, mu, std


# ─────────────────────────────────────────────────────────────────────────────
# SR offline multi-soggetto (coordinato EEG + GAF)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_sr_offline_multimodal(Xt, Xg, y, cfg):
    if not cfg.get("use_sr", False) or cfg.get("sr_mode", "offline") != "offline":
        return Xt, Xg, y

    n_segments = cfg.get("n_segments", 8)
    multiplier = cfg.get("sr_multiplier", 1)
    B, C, T    = Xt.shape
    classes    = np.unique(y)
    cls_idx    = {c: np.where(y == c)[0] for c in classes}
    seg_len    = T // n_segments
    synth_t, synth_g, synth_y = [], [], []

    for i in range(B):
        label_i    = y[i]
        candidates = cls_idx[label_i]
        if len(candidates) < n_segments:
            continue
        for _ in range(multiplier):
            donors = np.random.choice(candidates, size=n_segments, replace=True)
            new_t  = Xt[i].copy()
            for seg_idx, donor in enumerate(donors):
                start = seg_idx * seg_len
                end   = start + seg_len if seg_idx < n_segments - 1 else T
                new_t[:, start:end] = Xt[donor, :, start:end]
            synth_t.append(new_t)
            synth_g.append(Xg[donors[n_segments // 2]].copy())
            synth_y.append(label_i)

    if not synth_t:
        return Xt, Xg, y

    Xt_out = np.concatenate([Xt, np.stack(synth_t).astype(np.float32)], axis=0)
    Xg_out = np.concatenate([Xg, np.stack(synth_g).astype(np.float32)], axis=0)
    y_out  = np.concatenate([y,  np.array(synth_y, dtype=y.dtype)],     axis=0)
    perm   = np.random.permutation(len(y_out))
    return Xt_out[perm], Xg_out[perm], y_out[perm]


# ─────────────────────────────────────────────────────────────────────────────
# Costruzione loaders LOSO
# ─────────────────────────────────────────────────────────────────────────────

def build_loso_loaders_kd(
    cfg: dict,
    train_subs: list[int],
    val_sub: int,
    test_sub: int,
    cache: dict | None = None,
) -> tuple:
    """
    Costruisce i tre DataLoader per un singolo run LOSO.

    train  = T-session concatenata di train_subs (7 soggetti)
    val    = T-session di val_sub
    test   = E-session di test_sub

    Normalizzazione:
      - use_global_norm=True  → μ/σ calcolati sul training pool e applicati
                                 a train, val e test (come da paper TCFormer).
                                 Richiede che i dati in cache siano RAW.
      - use_global_norm=False → dati già normalizzati per-soggetto in cache.
    """
    use_global_norm = cfg.get("use_global_norm", True)

    def _load(sub_id):
        if cache is not None and sub_id in cache:
            return cache[sub_id]
        # carico al volo con raw=use_global_norm
        data = load_subject_both(sub_id, cfg, raw=use_global_norm)
        Xt, yt = data["T"]
        Xe, ye = data["E"]
        use_gaf = cfg.get("use_gaf", False)
        if use_gaf:
            Xtrg = make_gaf(Xt, cfg)
            Xteg = make_gaf(Xe, cfg)
        else:
            Xtrg = np.zeros((len(yt), 1, 1, 1), dtype=np.float32)
            Xteg = np.zeros((len(ye),  1, 1, 1), dtype=np.float32)
        return {"Xt": Xt, "yt": yt, "Xe": Xe, "ye": ye,
                "Xtrg": Xtrg, "Xteg": Xteg}

    # ── Concatena training pool ───────────────────────────────────────────────
    Xt_list, Xg_list, y_list = [], [], []
    for s in train_subs:
        d = _load(s)
        Xt_list.append(d["Xt"])
        Xg_list.append(d["Xtrg"])
        y_list.append(d["yt"])
    Xt_tr = np.concatenate(Xt_list, axis=0)
    Xg_tr = np.concatenate(Xg_list, axis=0)
    y_tr  = np.concatenate(y_list,  axis=0)

    # shuffle iniziale
    perm  = np.random.permutation(len(y_tr))
    Xt_tr = Xt_tr[perm];  Xg_tr = Xg_tr[perm];  y_tr = y_tr[perm]

    # ── Val e test ────────────────────────────────────────────────────────────
    dv    = _load(val_sub)
    Xt_va = dv["Xt"];   Xg_va = dv["Xtrg"];  y_va = dv["yt"]

    dt    = _load(test_sub)
    Xt_te = dt["Xe"];   Xg_te = dt["Xteg"];  y_te = dt["ye"]

    # ── Normalizzazione globale sul training pool ─────────────────────────────
    norm_tag = ""
    if use_global_norm:
        Xt_tr, Xt_va, Xt_te, mu, std = _apply_global_norm(Xt_tr, Xt_va, Xt_te)
        norm_tag = f"global-norm (μ/σ su {len(y_tr)} trial train)"

    # ── SR offline (dopo normalizzazione, sui dati già normalizzati) ──────────
    sr_tag = ""
    if cfg.get("use_sr", False) and cfg.get("sr_mode", "offline") == "offline":
        n_orig = len(y_tr)
        Xt_tr, Xg_tr, y_tr = _apply_sr_offline_multimodal(Xt_tr, Xg_tr, y_tr, cfg)
        sr_tag = f"SR offline {n_orig}→{len(y_tr)} trial"

    if norm_tag: _printf(f"  [{norm_tag}]")
    if sr_tag:   _printf(f"  [{sr_tag}]")

    trds = MMDataset(Xt_tr, Xg_tr, y_tr, augment=True,  aug_prob=cfg["aug_prob"])
    vads = MMDataset(Xt_va, Xg_va, y_va, augment=False, aug_prob=0.0)
    teds = MMDataset(Xt_te, Xg_te, y_te, augment=False, aug_prob=0.0)

    bs = cfg["batch_size"]
    trld = DataLoader(trds, batch_size=bs, shuffle=True,  num_workers=0)
    vald = DataLoader(vads, batch_size=bs, shuffle=False, num_workers=0)
    teld = DataLoader(teds, batch_size=bs, shuffle=False, num_workers=0)
    return trld, vald, teld


# ─────────────────────────────────────────────────────────────────────────────
# Singolo run (teacher + student) — condiviso da tutti gli schemi
# ─────────────────────────────────────────────────────────────────────────────

def _run_single(
    cfg: dict,
    trld, vald, teld,
    seed: int,
    verbose: bool,
    label: str,
    mode: str = "kd",      # "kd" | "kd_align"
) -> dict:
    """
    Addestra teacher + student su loader già pronti.
    mode: "kd" = KD standard, "kd_align" = KD con GAF alignment.
    Restituisce dict con teacher_acc/kappa e student_acc/kappa.
    """
    set_seed(seed)

    # ── Teacher ───────────────────────────────────────────────────────────────
    teacher         = build_teacher_model(cfg)
    teacher_trainer = TeacherTrainer(teacher, cfg)

    if cfg.get("train_teacher", True):
        if verbose: print(f"  STEP 1 [{label}] Training teacher...")
        teacher_trainer.fit(trld, vald, seed=seed)
        ckpt = cfg.get("teacher_ckpt_path", None)
        if ckpt:
            torch.save(teacher_trainer.model.state_dict(), ckpt)
    else:
        ckpt = cfg.get("teacher_ckpt_path", None)
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError("train_teacher=False ma teacher_ckpt_path non trovato.")
        teacher.load_state_dict(torch.load(ckpt, map_location=cfg["device"]))
        teacher = teacher.to(cfg["device"])

    active_teacher = teacher_trainer.model if cfg.get("train_teacher", True) else teacher

    y_true_t, y_pred_t = evaluate_teacher(active_teacher, teld, cfg["device"], cfg)
    teacher_acc, teacher_kappa = _compute_metrics(y_true_t, y_pred_t)
    if verbose:
        print(f"  Teacher [{label}]  Acc={teacher_acc:.2f}%  κ={teacher_kappa:.4f}")

    if not cfg.get("train_student", True):
        return dict(teacher_acc=teacher_acc, teacher_kappa=teacher_kappa,
                    student_acc=None, student_kappa=None)

    # ── Student ───────────────────────────────────────────────────────────────
    set_seed(seed)
    student = build_student_model(cfg)

    if mode == "kd_align":
        if not hasattr(active_teacher, "gaf_encoder"):
            raise RuntimeError("Il teacher non ha gaf_encoder. Assicurati che use_gaf=True.")
        gaf_encoder = active_teacher.gaf_encoder
        proj_head   = build_gaf_proj_head(cfg)
        kd_trainer  = KDAlignTrainer(
            teacher=active_teacher, student=student,
            gaf_encoder=gaf_encoder, proj_head=proj_head, cfg=cfg,
        )
        if verbose: print(f"  STEP 2 [{label}] Training student con KD-Align...")
        kd_trainer.fit(trld, vald, seed=seed)
        final_student = kd_trainer.student
        ckpt = cfg.get("student_ckpt_path", None)
        if ckpt:
            torch.save(final_student.state_dict(), ckpt)
    else:
        kd_trainer = KDTrainer(active_teacher, student, cfg)
        if verbose: print(f"  STEP 2 [{label}] Training student con KD...")
        kd_trainer.fit(trld, vald, seed=seed)
        final_student = kd_trainer.student
        ckpt = cfg.get("student_ckpt_path", None)
        if ckpt:
            torch.save(final_student.state_dict(), ckpt)

    y_true_s, y_pred_s = evaluate_student(final_student, teld, cfg["device"], cfg)
    student_acc, student_kappa = _compute_metrics(y_true_s, y_pred_s)
    if verbose:
        print(f"  Student [{label}]  Acc={student_acc:.2f}%  κ={student_kappa:.4f}"
              f"  Δ={student_acc - teacher_acc:+.2f}%")

    return dict(teacher_acc=teacher_acc, teacher_kappa=teacher_kappa,
                student_acc=student_acc, student_kappa=student_kappa)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline LOSO standard (9 fold, 1 val per test)
# ─────────────────────────────────────────────────────────────────────────────

def run_loso_kd_fold(
    fold_info: dict,
    seed: int = 42,
    verbose: bool = True,
    cache: dict | None = None,
    cfg_override: dict | None = None,
    mode: str = "kd",
) -> dict:
    """
    Esegue un singolo fold LOSO (schema standard circolare).
    mode: "kd" | "kd_align"
    """
    cfg = _get_loso_cfg(cfg_override)
    if mode == "kd_align":
        cfg["use_gaf"] = True

    fold_id    = fold_info["fold"]
    test_sub   = fold_info["test_sub"]
    val_sub    = fold_info["val_sub"]
    train_subs = fold_info["train_subs"]

    norm_tag = "global-norm" if cfg.get("use_global_norm", True) else "per-subject-norm"
    aug_parts = []
    if cfg.get("use_sr",    False): aug_parts.append(f"SR({cfg.get('sr_mode','offline')})")
    if cfg.get("use_mixup", False): aug_parts.append("Mixup")
    aug_str = " ".join(aug_parts) if aug_parts else "no-aug"

    if verbose:
        _printf()
        print(f"LOSO-{mode.upper()}  Fold {fold_id}/9 | "
              f"test=S{test_sub:02d}  val=S{val_sub:02d}  "
              f"train={train_subs}  seed={seed}  {norm_tag}  {aug_str}")
        _printf()

    set_seed(seed)
    trld, vald, teld = build_loso_loaders_kd(
        cfg, train_subs, val_sub, test_sub, cache=cache
    )
    label = f"Fold {fold_id} S{test_sub:02d}"
    out   = _run_single(cfg, trld, vald, teld, seed, verbose, label, mode=mode)

    if verbose: _printf()
    return dict(fold=fold_id, test_sub=test_sub, val_sub=val_sub, seed=seed, **out)


def run_loso_kd_all_folds(
    seed: int = 42,
    verbose: bool = True,
    cache: dict | None = None,
    cfg_override: dict | None = None,
    mode: str = "kd",
    out_dir: str = ".",
) -> list[dict]:
    """
    Esegue tutti i 9 fold LOSO standard.
    Salva ogni risultato su CSV appena completato.
    Se il CSV esiste già, salta i fold già presenti (resume automatico).
    """
    folds    = _generate_loso_folds(9)
    csv_file = _csv_path(mode, "standard", seed, out_dir)
    done     = _load_done_keys_standard(csv_file)

    if done:
        print(f"[CSV resume] Trovati {len(done)} fold già completati in '{csv_file}'")

    results = []
    for fold_info in folds:
        key = (fold_info["fold"], fold_info["test_sub"],
               fold_info["val_sub"], seed)
        if key in done:
            print(f"  → Fold {fold_info['fold']} già completato, skip.")
            continue

        out = run_loso_kd_fold(fold_info, seed=seed, verbose=verbose,
                               cache=cache, cfg_override=cfg_override, mode=mode)
        results.append(out)

        # Salva subito su CSV
        row = {**out, "seed": seed}
        _append_row(csv_file, _CSV_FIELDNAMES_STANDARD, row)
        print(f"  [CSV] Fold {fold_info['fold']} salvato → '{csv_file}'")

        torch.cuda.empty_cache()

    # Ricarica tutto il CSV per il summary (include fold già presenti al resume)
    all_results = _reload_standard_results(csv_file, seed)
    _print_summary(all_results, title=f"LOSO-{mode.upper()} Summary (seed={seed})")
    return all_results

def _reload_standard_results(csv_file: Path, seed: int) -> list[dict]:
    """Rilegge il CSV standard e ricostruisce la lista di result dict."""
    results = []
    if not csv_file.exists():
        return results
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["seed"]) != seed:
                continue
            results.append({
                "fold":          int(row["fold"]),
                "test_sub":      int(row["test_sub"]),
                "val_sub":       int(row["val_sub"]),
                "seed":          int(row["seed"]),
                "teacher_acc":   float(row["teacher_acc"]),
                "teacher_kappa": float(row["teacher_kappa"]),
                "student_acc":   float(row["student_acc"]) if row["student_acc"] not in ("", "None") else None,
                "student_kappa": float(row["student_kappa"]) if row["student_kappa"] not in ("", "None") else None,
            })
    return sorted(results, key=lambda r: r["fold"])


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline LOSO multi-val (72 run, accuracy media per soggetto)
# ─────────────────────────────────────────────────────────────────────────────
def run_loso_kd_multival_all(
    seed: int = 42,
    verbose: bool = True,
    cache: dict | None = None,
    cfg_override: dict | None = None,
    mode: str = "kd",
    out_dir: str = ".",
) -> list[dict]:
    """
    Schema multi-val: 72 run (9 test × 8 val).
    Salva ogni run su CSV appena completata.
    Resume automatico: salta le run già presenti nel CSV.
    """
    cfg = _get_loso_cfg(cfg_override)
    if mode == "kd_align":
        cfg["use_gaf"] = True

    runs     = _generate_loso_multival_runs(9)
    n_tot    = len(runs)
    csv_file = _csv_path(mode, "multival", seed, out_dir)
    done     = _load_done_keys_multival(csv_file)

    norm_tag  = "global-norm" if cfg.get("use_global_norm", True) else "per-subject-norm"
    aug_parts = []
    if cfg.get("use_sr",    False): aug_parts.append(f"SR({cfg.get('sr_mode','offline')})")
    if cfg.get("use_mixup", False): aug_parts.append("Mixup")
    aug_str = " ".join(aug_parts) if aug_parts else "no-aug"

    _printf()
    print(f"LOSO-{mode.upper()}-MultiVal | {n_tot} run totali (9 test × 8 val)"
          f"  {norm_tag}  {aug_str}  seed={seed}")
    if done:
        print(f"[CSV resume] Trovate {len(done)} run già completate in '{csv_file}'")
    _printf()

    from collections import defaultdict
    per_sub: dict[int, list[dict]] = defaultdict(list)

    for run_info in runs:
        test_sub   = run_info["test_sub"]
        val_sub    = run_info["val_sub"]
        train_subs = run_info["train_subs"]
        run_id     = run_info["run_id"]
        key        = (test_sub, val_sub, seed)

        if key in done:
            print(f"  → Run {run_id:02d} (test=S{test_sub:02d} val=S{val_sub:02d}) già completata, skip.")
            # Recupera i dati dal CSV per l'aggregazione finale
            row_data = _find_multival_row(csv_file, test_sub, val_sub, seed)
            if row_data:
                per_sub[test_sub].append(row_data)
            continue

        print(f"\n[Run {run_id:02d}/{n_tot}] test=S{test_sub:02d}  val=S{val_sub:02d}"
              f"  train={train_subs}")

        set_seed(seed)
        trld, vald, teld = build_loso_loaders_kd(
            cfg, train_subs, val_sub, test_sub, cache=cache
        )
        label = f"Run {run_id} S{test_sub:02d}/val=S{val_sub:02d}"
        out   = _run_single(cfg, trld, vald, teld, seed, verbose, label, mode=mode)
        torch.cuda.empty_cache()

        row = {
            "run_id":        run_id,
            "test_sub":      test_sub,
            "val_sub":       val_sub,
            "seed":          seed,
            "teacher_acc":   out["teacher_acc"],
            "teacher_kappa": out["teacher_kappa"],
            "student_acc":   out["student_acc"],
            "student_kappa": out["student_kappa"],
        }
        _append_row(csv_file, _CSV_FIELDNAMES_MULTIVAL, row)
        print(f"  [CSV] Run {run_id:02d} salvata → '{csv_file}'")

        per_sub[test_sub].append({
            "val_sub":       val_sub,
            "teacher_acc":   out["teacher_acc"],
            "teacher_kappa": out["teacher_kappa"],
            "student_acc":   out["student_acc"],
            "student_kappa": out["student_kappa"],
        })

    # Aggrega per soggetto
    results = []
    for test_sub in sorted(per_sub.keys()):
        sub_runs = per_sub[test_sub]
        t_accs   = [r["teacher_acc"]   for r in sub_runs]
        t_kaps   = [r["teacher_kappa"] for r in sub_runs]
        s_accs   = [r["student_acc"]   for r in sub_runs if r["student_acc"] is not None]
        s_kaps   = [r["student_kappa"] for r in sub_runs if r["student_kappa"] is not None]
        results.append(dict(
            test_sub           = test_sub,
            n_val_runs         = len(sub_runs),
            teacher_acc_mean   = float(np.mean(t_accs)),
            teacher_acc_std    = float(np.std(t_accs)),
            teacher_kappa_mean = float(np.mean(t_kaps)),
            teacher_kappa_std  = float(np.std(t_kaps)),
            teacher_accs       = t_accs,
            student_acc_mean   = float(np.mean(s_accs)) if s_accs else None,
            student_acc_std    = float(np.std(s_accs))  if s_accs else None,
            student_kappa_mean = float(np.mean(s_kaps)) if s_kaps else None,
            student_kappa_std  = float(np.std(s_kaps))  if s_kaps else None,
            student_accs       = s_accs,
        ))

    _print_summary_multival(results,
                            title=f"LOSO-{mode.upper()}-MultiVal Summary (seed={seed})")
    return results

def _find_multival_row(csv_file: Path, test_sub: int, val_sub: int, seed: int) -> dict | None:
    """Recupera una singola run dal CSV multival per l'aggregazione in caso di resume."""
    if not csv_file.exists():
        return None
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row["test_sub"]) == test_sub
                    and int(row["val_sub"]) == val_sub
                    and int(row["seed"]) == seed):
                return {
                    "val_sub":       val_sub,
                    "teacher_acc":   float(row["teacher_acc"]),
                    "teacher_kappa": float(row["teacher_kappa"]),
                    "student_acc":   float(row["student_acc"]) if row["student_acc"] not in ("", "None") else None,
                    "student_kappa": float(row["student_kappa"]) if row["student_kappa"] not in ("", "None") else None,
                }
    return None
# ─────────────────────────────────────────────────────────────────────────────
# Stampa summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict], title: str = "LOSO Summary"):
    _printf()
    print(title)
    _printf()
    t_accs = [r["teacher_acc"]   for r in results]
    t_kaps = [r["teacher_kappa"] for r in results]
    s_accs = [r["student_acc"]   for r in results if r.get("student_acc") is not None]
    s_kaps = [r["student_kappa"] for r in results if r.get("student_kappa") is not None]

    header = f"{'Fold':>5}  {'TestS':>6}  {'ValS':>5}  {'TeachAcc':>9}  {'TeachK':>7}"
    if s_accs:
        header += f"  {'StudAcc':>8}  {'StudK':>7}  {'Δ':>6}"
    print(header)
    print("─" * len(header))
    for r in results:
        line = (f"{r['fold']:>5}  S{r['test_sub']:02d}    S{r['val_sub']:02d}"
                f"    {r['teacher_acc']:>8.2f}%  {r['teacher_kappa']:>7.4f}")
        if r.get("student_acc") is not None:
            line += (f"  {r['student_acc']:>7.2f}%  {r['student_kappa']:>7.4f}"
                     f"  {r['student_acc'] - r['teacher_acc']:>+6.2f}")
        print(line)
    print("─" * len(header))
    print(f"{'MEAN':>5}  {'':>6}  {'':>5}"
          f"    {np.mean(t_accs):>8.2f}%  {np.mean(t_kaps):>7.4f}", end="")
    if s_accs:
        print(f"  {np.mean(s_accs):>7.2f}%  {np.mean(s_kaps):>7.4f}"
              f"  {np.mean(s_accs) - np.mean(t_accs):>+6.2f}")
    else:
        print()
    print(f"{'STD':>5}  {'':>6}  {'':>5}"
          f"    {np.std(t_accs):>8.2f}   {np.std(t_kaps):>7.4f}", end="")
    if s_accs:
        print(f"  {np.std(s_accs):>7.2f}   {np.std(s_kaps):>7.4f}")
    else:
        print()
    _printf()


def _print_summary_multival(results: list[dict], title: str = "LOSO MultiVal Summary"):
    _printf()
    print(title)
    _printf()
    t_means = [r["teacher_acc_mean"] for r in results]
    t_kaps  = [r["teacher_kappa_mean"] for r in results]
    s_means = [r["student_acc_mean"] for r in results if r["student_acc_mean"] is not None]
    s_kaps  = [r["student_kappa_mean"] for r in results if r["student_kappa_mean"] is not None]

    header = (f"{'TestS':>6}  {'N':>3}  {'TeachAcc±std':>16}  {'TeachK':>7}")
    if s_means:
        header += f"  {'StudAcc±std':>15}  {'StudK':>7}  {'Δ':>6}"
    print(header)
    print("─" * len(header))
    for r in results:
        line = (f"S{r['test_sub']:02d}    {r['n_val_runs']:>3}"
                f"  {r['teacher_acc_mean']:>6.2f}%±{r['teacher_acc_std']:>5.2f}"
                f"  {r['teacher_kappa_mean']:>7.4f}")
        if r["student_acc_mean"] is not None:
            delta = r["student_acc_mean"] - r["teacher_acc_mean"]
            line += (f"  {r['student_acc_mean']:>5.2f}%±{r['student_acc_std']:>5.2f}"
                     f"  {r['student_kappa_mean']:>7.4f}"
                     f"  {delta:>+6.2f}")
        print(line)
    print("─" * len(header))
    print(f"{'MEAN':>6}  {'':>3}  {np.mean(t_means):>6.2f}%{'':>9}  {np.mean(t_kaps):>7.4f}", end="")
    if s_means:
        delta_mean = np.mean(s_means) - np.mean(t_means)
        print(f"  {np.mean(s_means):>5.2f}%{'':>9}  {np.mean(s_kaps):>7.4f}  {delta_mean:>+6.2f}")
    else:
        print()
    print(f"{'STD':>6}  {'':>3}  {np.std(t_means):>6.2f} {'':>9}  {np.std(t_kaps):>7.4f}", end="")
    if s_means:
        print(f"  {np.std(s_means):>5.2f} {'':>9}  {np.std(s_kaps):>7.4f}")
    else:
        print()
    _printf()


# ─────────────────────────────────────────────────────────────────────────────
# Shortcut KD-Align (mantiene compatibilità con chiamate precedenti)
# ─────────────────────────────────────────────────────────────────────────────

def run_loso_kd_align_fold(*args, **kwargs):
    kwargs["mode"] = "kd_align"
    return run_loso_kd_fold(*args, **kwargs)

def run_loso_kd_align_all_folds(*args, **kwargs):
    kwargs["mode"] = "kd_align"
    return run_loso_kd_all_folds(*args, **kwargs)

def run_loso_kd_align_multival_all(*args, **kwargs):
    kwargs["mode"] = "kd_align"
    return run_loso_kd_multival_all(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LOSO KD pipeline")
    parser.add_argument("--mode",           choices=["kd", "kd_align"], default="kd")
    parser.add_argument("--scheme",         choices=["standard", "multival"], default="multival",
                        help="standard=9 fold, multival=72 run (9×8)")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--fold",           type=int, default=None,
                        help="Solo per --scheme standard: esegui solo il fold N (1-9).")
    parser.add_argument("--cache",          action="store_true",
                        help="Pre-carica tutti i soggetti in RAM (consigliato).")
    parser.add_argument("--no_global_norm", action="store_true",
                        help="Disabilita normalizzazione globale (usa z-score per-soggetto).")
    parser.add_argument("--no_teacher",     action="store_true")
    parser.add_argument("--no_student",     action="store_true")
    parser.add_argument("--out_dir",        type=str, default=".",
                        help="Directory dove salvare i CSV dei risultati.")
    args = parser.parse_args()

    override = {}
    if args.no_teacher:     override["train_teacher"]   = False
    if args.no_student:     override["train_student"]   = False
    if args.no_global_norm: override["use_global_norm"] = False

    cfg   = _get_loso_cfg(override)
    cache = build_loso_cache(cfg) if args.cache else None

    if args.scheme == "multival":
        run_loso_kd_multival_all(seed=args.seed, verbose=True,
                                 cache=cache, cfg_override=override,
                                 mode=args.mode, out_dir=args.out_dir)
    else:
        if args.fold is not None:
            folds     = _generate_loso_folds(9)
            fold_info = next(f for f in folds if f["fold"] == args.fold)
            run_loso_kd_fold(fold_info, seed=args.seed, verbose=True,
                             cache=cache, cfg_override=override, mode=args.mode)
        else:
            run_loso_kd_all_folds(seed=args.seed, verbose=True,
                                  cache=cache, cfg_override=override,
                                  mode=args.mode, out_dir=args.out_dir)
