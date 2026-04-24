# pipelines_kd.py — Pipeline separata Teacher-Student KD
# Import aggiornati: augmentation_kd e preprocessing_kd
# ============================================================

import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score

from config_kd import get_kd_subject_cfg, set_seed
from augmentation_kd import MMDataset
from preprocessing_kd import load_subject, preprocess_subject, split_train_val

from model_kd import build_teacher_model, build_student_model, build_gaf_proj_head
from trainer_kd import (
    TeacherTrainer,
    KDTrainer,
    KDAlignTrainer,
    evaluate_teacher,
    evaluate_student,
)


# ── Helper dataloaders ───────────────────────────────────────

def _build_ws_loaders_kd(cfg: dict, sub_id: int):
    X_tr_raw, y_tr, X_te_raw, y_te = load_subject(sub_id, cfg)
    X_tr_t, X_te_t, X_tr_g, X_te_g = preprocess_subject(X_tr_raw, X_te_raw, cfg)

    split_train = cfg.get("split_train", True)

    if split_train:
        X_tr_t2, X_tr_g2, y_tr2, X_va_t, X_va_g, y_va = split_train_val(
            X_tr_t, X_tr_g, y_tr, cfg["split_train_ratio"]
        )
        tr_ds = MMDataset(X_tr_t2, X_tr_g2, y_tr2, augment=True, aug_prob=cfg["aug_prob"])
        va_ds = MMDataset(X_va_t, X_va_g, y_va, augment=False, aug_prob=0.0)
        te_ds = MMDataset(X_te_t, X_te_g, y_te, augment=False, aug_prob=0.0)
    else:
        tr_ds = MMDataset(X_tr_t, X_tr_g, y_tr, augment=True, aug_prob=cfg["aug_prob"])
        va_ds = MMDataset(X_te_t, X_te_g, y_te, augment=False, aug_prob=0.0)
        te_ds = va_ds

    tr_ld = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    return tr_ld, va_ld, te_ld


# ── Helper metriche ──────────────────────────────────────────

def _compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100.0
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, kappa


# ── Pipeline KD single-seed ──────────────────────────────────

def run_subject_kd(sub_id: int, seed: int = 42, verbose: bool = True):
    cfg = get_kd_subject_cfg(sub_id)
    split_train = cfg.get("split_train", True)

    if verbose:
        split_tag = "split T 80/20" if split_train else "E=val+test"
        print(f"\n{'═'*70}")
        print(f"[KD] Soggetto {sub_id} | seed={seed} | {split_tag}")
        print(f"{'═'*70}")

    tr_ld, va_ld, te_ld = _build_ws_loaders_kd(cfg, sub_id)

    # ── Step 1: Teacher ──────────────────────────────────────
    set_seed(seed)
    teacher = build_teacher_model(cfg)
    teacher_trainer = TeacherTrainer(teacher, cfg)

    if cfg.get("train_teacher", True):
        if verbose:
            print("\n[STEP 1] Training teacher cross-modal...")
        teacher_trainer.fit(tr_ld, va_ld, seed=seed)

        ckpt = cfg.get("teacher_ckpt_path", None)
        if ckpt:
            torch.save(teacher_trainer.model.state_dict(), ckpt)
            if verbose:
                print(f"  [SAVE] Teacher checkpoint salvato in: {ckpt}")
    else:
        ckpt = cfg.get("teacher_ckpt_path", None)
        if ckpt is None or not os.path.exists(ckpt):
            raise FileNotFoundError("train_teacher=False ma teacher_ckpt_path non esiste.")
        teacher.load_state_dict(torch.load(ckpt, map_location=cfg["device"]))
        teacher = teacher.to(cfg["device"])
        if verbose:
            print(f"\n[STEP 1] Teacher caricato da checkpoint: {ckpt}")

    active_teacher = teacher_trainer.model if cfg.get("train_teacher", True) else teacher
    y_true_t, y_pred_t = evaluate_teacher(active_teacher, te_ld, cfg["device"], cfg)
    teacher_acc, teacher_kappa = _compute_metrics(y_true_t, y_pred_t)

    if verbose:
        print(f"\n[Teacher] S{sub_id:02d} seed={seed} → Acc: {teacher_acc:.2f}% | Kappa: {teacher_kappa:.4f}")

    # ── Step 2: Student KD ───────────────────────────────────
    set_seed(seed)
    student = build_student_model(cfg)
    kd_trainer = KDTrainer(active_teacher, student, cfg)

    if cfg.get("train_student", True):
        if verbose:
            print("\n[STEP 2] Training student with KD...")
        kd_trainer.fit(tr_ld, va_ld, seed=seed)

        ckpt = cfg.get("student_ckpt_path", None)
        if ckpt:
            torch.save(kd_trainer.student.state_dict(), ckpt)
            if verbose:
                print(f"  [SAVE] Student checkpoint salvato in: {ckpt}")
    else:
        ckpt = cfg.get("student_ckpt_path", None)
        if ckpt is None or not os.path.exists(ckpt):
            raise FileNotFoundError("train_student=False ma student_ckpt_path non esiste.")
        student.load_state_dict(torch.load(ckpt, map_location=cfg["device"]))
        student = student.to(cfg["device"])
        if verbose:
            print(f"\n[STEP 2] Student caricato da checkpoint: {ckpt}")

    final_student = kd_trainer.student if cfg.get("train_student", True) else student
    y_true_s, y_pred_s = evaluate_student(final_student, te_ld, cfg["device"], cfg)
    student_acc, student_kappa = _compute_metrics(y_true_s, y_pred_s)

    if verbose:
        print(f"\n[Student KD] S{sub_id:02d} seed={seed} → Acc: {student_acc:.2f}% | Kappa: {student_kappa:.4f}")
        print(f"  Δ Student - Teacher: {student_acc - teacher_acc:+.2f}%")
        print(f"{'─'*70}")

    return {
        "subject": sub_id,
        "seed": seed,
        "teacher_acc": teacher_acc,
        "teacher_kappa": teacher_kappa,
        "student_acc": student_acc,
        "student_kappa": student_kappa,
    }


# ── Pipeline KD multi-seed ───────────────────────────────────

def run_subject_kd_multiseed(sub_id: int):
    cfg = get_kd_subject_cfg(sub_id)
    seeds = cfg["seeds"]

    print(f"\n{'═'*70}")
    print(f"[KD] S{sub_id:02d} | Multi-seed ({len(seeds)} run)")
    print(f"{'═'*70}")

    teacher_accs, teacher_kappas = [], []
    student_accs, student_kappas = [], []

    for i, seed in enumerate(seeds):
        print(f"\n── Seed {i+1}/{len(seeds)}: {seed} ──")
        out = run_subject_kd(sub_id, seed=seed, verbose=False)

        teacher_accs.append(out["teacher_acc"])
        teacher_kappas.append(out["teacher_kappa"])
        student_accs.append(out["student_acc"])
        student_kappas.append(out["student_kappa"])

        print(
            f"  Teacher: {out['teacher_acc']:.2f}% | κ={out['teacher_kappa']:.4f} || "
            f"Student KD: {out['student_acc']:.2f}% | κ={out['student_kappa']:.4f}"
        )
        torch.cuda.empty_cache()

    t_acc_m, t_acc_s = np.mean(teacher_accs), np.std(teacher_accs)
    t_k_m,   t_k_s   = np.mean(teacher_kappas), np.std(teacher_kappas)
    s_acc_m, s_acc_s = np.mean(student_accs), np.std(student_accs)
    s_k_m,   s_k_s   = np.mean(student_kappas), np.std(student_kappas)

    print(f"\n{'─'*70}")
    print(f"[KD] S{sub_id:02d} risultati finali")
    print(f"  Teacher    → {t_acc_m:.2f} ± {t_acc_s:.2f}% | κ={t_k_m:.4f} ± {t_k_s:.4f}")
    print(f"  Student KD → {s_acc_m:.2f} ± {s_acc_s:.2f}% | κ={s_k_m:.4f} ± {s_k_s:.4f}")
    print(f"  Δ mean(Student - Teacher) = {s_acc_m - t_acc_m:+.2f}%")
    print(f"{'─'*70}")

    return {
        "subject": sub_id,
        "teacher_acc_mean": t_acc_m, "teacher_acc_std": t_acc_s,
        "teacher_kappa_mean": t_k_m,  "teacher_kappa_std": t_k_s,
        "student_acc_mean":  s_acc_m, "student_acc_std":  s_acc_s,
        "student_kappa_mean": s_k_m,  "student_kappa_std": s_k_s,
        "teacher_accs": teacher_accs,
        "student_accs":  student_accs,
    }


# ── Pipeline KD-Align single-seed ────────────────────────────

def run_subject_kd_align(sub_id: int, seed: int = 42, verbose: bool = True):
    """
    Flusso:
      Step 1 → TeacherTrainer (use_gaf=True, cross-modal)
      Step 2 → estrai gaf_encoder dal teacher addestrato
      Step 3 → KDAlignTrainer (teacher frozen + gaf_encoder frozen + proj_head nuovo)
    Al termine lo student è EEG-only puro: proj_head scartato.
    """
    cfg = get_kd_subject_cfg(sub_id)
    cfg["use_gaf"] = True   # KD-Align richiede use_gaf=True per il teacher

    split_train = cfg.get("split_train", True)

    if verbose:
        split_tag = "split T 80/20" if split_train else "E=val+test"
        beta = cfg.get("align_beta", 0.3)
        temp = cfg.get("align_temperature", 0.07)
        print(f"\n{'═'*70}")
        print(f"[KD-Align] Soggetto {sub_id} | seed={seed} | {split_tag}")
        print(f"  align_beta={beta} | align_temperature={temp}")
        print(f"{'═'*70}")

    tr_ld, va_ld, te_ld = _build_ws_loaders_kd(cfg, sub_id)

    # ── Step 1: Teacher cross-modal ──────────────────────────
    set_seed(seed)
    teacher = build_teacher_model(cfg)
    teacher_trainer = TeacherTrainer(teacher, cfg)

    if cfg.get("train_teacher", True):
        if verbose:
            print("\n[STEP 1] Training teacher cross-modal (EEG + GAF)...")
        teacher_trainer.fit(tr_ld, va_ld, seed=seed)
    else:
        ckpt = cfg.get("teacher_ckpt_path", None)
        if ckpt is None or not os.path.exists(ckpt):
            raise FileNotFoundError("train_teacher=False ma teacher_ckpt_path non esiste.")
        teacher.load_state_dict(torch.load(ckpt, map_location=cfg["device"]))
        teacher = teacher.to(cfg["device"])
        if verbose:
            print(f"\n[STEP 1] Teacher caricato da checkpoint: {ckpt}")

    active_teacher = teacher_trainer.model if cfg.get("train_teacher", True) else teacher
    y_true_t, y_pred_t = evaluate_teacher(active_teacher, te_ld, cfg["device"], cfg)
    teacher_acc, teacher_kappa = _compute_metrics(y_true_t, y_pred_t)

    if verbose:
        print(f"\n[Teacher] S{sub_id:02d} seed={seed} → Acc: {teacher_acc:.2f}% | Kappa: {teacher_kappa:.4f}")

    # ── Step 2: Estrai GAF encoder dal teacher ───────────────
    if not hasattr(active_teacher, "gaf_encoder"):
        raise RuntimeError(
            "Il teacher non ha gaf_encoder. "
            "Assicurati che use_gaf=True durante il training del teacher."
        )
    gaf_encoder = active_teacher.gaf_encoder

    # ── Step 3: Student con KD + alignment GAF ───────────────
    set_seed(seed)
    student   = build_student_model(cfg)
    proj_head = build_gaf_proj_head(cfg)

    align_trainer = KDAlignTrainer(
        teacher=active_teacher,
        student=student,
        gaf_encoder=gaf_encoder,
        proj_head=proj_head,
        cfg=cfg,
    )

    if cfg.get("train_student", True):
        if verbose:
            print("\n[STEP 3] Training student con KD + GAF alignment...")
        align_trainer.fit(tr_ld, va_ld, seed=seed)

        ckpt = cfg.get("student_ckpt_path", None)
        if ckpt:
            torch.save(align_trainer.student.state_dict(), ckpt)
            if verbose:
                print(f"  [SAVE] Student checkpoint salvato in: {ckpt}")
    else:
        ckpt = cfg.get("student_ckpt_path", None)
        if ckpt is None or not os.path.exists(ckpt):
            raise FileNotFoundError("train_student=False ma student_ckpt_path non esiste.")
        student.load_state_dict(torch.load(ckpt, map_location=cfg["device"]))
        student = student.to(cfg["device"])
        if verbose:
            print(f"\n[STEP 3] Student caricato da checkpoint: {ckpt}")

    final_student = align_trainer.student if cfg.get("train_student", True) else student
    y_true_s, y_pred_s = evaluate_student(final_student, te_ld, cfg["device"], cfg)
    student_acc, student_kappa = _compute_metrics(y_true_s, y_pred_s)

    if verbose:
        print(f"\n[Student KD-Align] S{sub_id:02d} seed={seed} → Acc: {student_acc:.2f}% | Kappa: {student_kappa:.4f}")
        print(f"  Δ Student - Teacher: {student_acc - teacher_acc:+.2f}%")
        print(f"{'─'*70}")

    return {
        "subject": sub_id,
        "seed": seed,
        "teacher_acc": teacher_acc,
        "teacher_kappa": teacher_kappa,
        "student_acc": student_acc,
        "student_kappa": student_kappa,
    }


# ── Pipeline KD-Align multi-seed ─────────────────────────────

def run_subject_kd_align_multiseed(sub_id: int):
    cfg = get_kd_subject_cfg(sub_id)
    seeds = cfg["seeds"]

    print(f"\n{'═'*70}")
    print(f"[KD-Align] S{sub_id:02d} | Multi-seed ({len(seeds)} run)")
    print(f"{'═'*70}")

    teacher_accs, teacher_kappas = [], []
    student_accs, student_kappas = [], []

    for i, seed in enumerate(seeds):
        print(f"\n── Seed {i+1}/{len(seeds)}: {seed} ──")
        out = run_subject_kd_align(sub_id, seed=seed, verbose=False)

        teacher_accs.append(out["teacher_acc"])
        teacher_kappas.append(out["teacher_kappa"])
        student_accs.append(out["student_acc"])
        student_kappas.append(out["student_kappa"])

        print(
            f"  Teacher: {out['teacher_acc']:.2f}% | κ={out['teacher_kappa']:.4f} || "
            f"Student KD-Align: {out['student_acc']:.2f}% | κ={out['student_kappa']:.4f}"
        )
        torch.cuda.empty_cache()

    t_acc_m, t_acc_s = np.mean(teacher_accs), np.std(teacher_accs)
    t_k_m,   t_k_s   = np.mean(teacher_kappas), np.std(teacher_kappas)
    s_acc_m, s_acc_s = np.mean(student_accs), np.std(student_accs)
    s_k_m,   s_k_s   = np.mean(student_kappas), np.std(student_kappas)

    print(f"\n{'─'*70}")
    print(f"[KD-Align] S{sub_id:02d} risultati finali")
    print(f"  Teacher        → {t_acc_m:.2f} ± {t_acc_s:.2f}% | κ={t_k_m:.4f} ± {t_k_s:.4f}")
    print(f"  Student Align  → {s_acc_m:.2f} ± {s_acc_s:.2f}% | κ={s_k_m:.4f} ± {s_k_s:.4f}")
    print(f"  Δ mean(Student - Teacher) = {s_acc_m - t_acc_m:+.2f}%")
    print(f"{'─'*70}")

    return {
        "subject": sub_id,
        "teacher_acc_mean": t_acc_m, "teacher_acc_std": t_acc_s,
        "teacher_kappa_mean": t_k_m,  "teacher_kappa_std": t_k_s,
        "student_acc_mean":  s_acc_m, "student_acc_std":  s_acc_s,
        "student_kappa_mean": s_k_m,  "student_kappa_std": s_k_s,
        "teacher_accs": teacher_accs,
        "student_accs":  student_accs,
    }
