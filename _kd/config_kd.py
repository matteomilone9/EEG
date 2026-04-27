# config_kd.py — Configurazione separata per pipeline Teacher-Student KD
# ============================================================

import warnings
import random
import copy
import numpy as np
import torch

warnings.filterwarnings('ignore')

HARD_SUBJECTS_KD = {2, 6}

KD_CFG = {
    # ============================================================
    # Dataset
    # ============================================================
    "n_subjects": 9,
    "sfreq": 250,
    "lowcut": 4.0,
    "highcut": 38.0,
    "tmin": 0.0,
    "tmax": 4.0,
    "n_channels": 22,
    "n_classes": 4,

    # ============================================================
    # GAF
    # ============================================================
    "image_size": 64,
    "gaf_method": "summation",
    "downsample_to": 128,

    # Backbone GAF: "cnn" oppure "vit"
    "gaf_backbone_type": "vit",
    "gaf_image_size": 224,
    "vit_model_name": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_pretrained": True,
    "vit_freeze": True,

    # ============================================================
    # Run control
    # ============================================================
    "run_all_subjects": True,
    "subject_ids": list(range(1, 10)),
    "single_subject": 3,
    "default_seed": 42,
    "default_multi_seed": False,

    # ============================================================
    # Shared training
    # ============================================================
    "batch_size": 32,
    "aug_prob": 0.5,
    "n_tta": 5,
    "split_train": False,
    "split_train_ratio": 0.8,
    "seeds": [42, 123, 456, 789, 1234],
    "use_gaf": True,
    "use_tgasf": True,
    "kd_threshold": 0.65, #0.5

    # ============================================================
    # TEACHER training
    # ============================================================
    "epochs_teacher": 1000,
    "lr_teacher": 9e-4,
    "patience_teacher": 500,
    "warmup_epochs_teacher": 50,
    "weight_decay": 1e-4,
    "teacher_label_smoothing": 0.15,

    "use_mixup": True,
    "mixup_prob": 0.5,
    "mixup_alpha": 0.4,
    "sr_prob": 0.5,
    "n_segments": 10,

    # ============================================================
    # TEACHER architettura
    # ============================================================
    "teacher_F1": 16,
    "teacher_temp_kernel_lengths": (20, 32, 64),
    "teacher_D": 2,
    "teacher_pool_length_1": 8,
    "teacher_pool_length_2": 7,
    "teacher_dropout_conv": 0.3,
    "teacher_d_group": 16,
    "teacher_use_group_attn": True,
    "teacher_q_heads": 4,
    "teacher_kv_heads": 2,
    "teacher_trans_depth": 2,
    "teacher_trans_dropout": 0.4,
    "teacher_drop_path_max": 0.1,
    "teacher_tcn_depth": 2,
    "teacher_kernel_length_tcn": 4,
    "teacher_dropout_tcn": 0.3,

    "teacher_gaf_token_dim": 32,  #64
    "teacher_gaf_base_channels": 16,
    "teacher_gaf_dropout": 0.4,
    "teacher_cross_attn_depth": 1, #2
    "teacher_cross_attn_heads": 4,
    "teacher_cross_attn_dropout": 0.3,
    "teacher_ff_mult": 1,  #2
    "teacher_cls_hidden": 32,  #64

    # ============================================================
    # STUDENT training
    # ============================================================
    "epochs_student": 1000,
    "lr_student": 9e-4,
    "patience_student": 500,
    "warmup_epochs_student": 50,
    "student_label_smoothing": 0.15,
    "freeze_teacher": True,

    # ============================================================
    # STUDENT architettura
    # ============================================================
    "student_F1": 16,
    "student_temp_kernel_lengths": (20, 32, 64),
    "student_D": 2,
    "student_pool_length_1": 8,
    "student_pool_length_2": 7,
    "student_dropout_conv": 0.3,
    "student_d_group": 16,
    "student_use_group_attn": True,
    "student_q_heads": 4,
    "student_kv_heads": 2,
    "student_trans_depth": 2,
    "student_trans_dropout": 0.4,
    "student_drop_path_max": 0.1,
    "student_tcn_depth": 2,
    "student_kernel_length_tcn": 4,
    "student_dropout_tcn": 0.3,

    # ============================================================
    # KD loss
    # ============================================================
    "kd_alpha": 0.35,  #0.5
    "kd_temperature": 3.0, #2

    # ============================================================
    # KD-Align
    # ============================================================
    "use_kd_align": True,
    "align_beta": 0.1,  #0.2
    "align_temperature": 0.10, #.07

    # ============================================================
    # Checkpoint paths
    # ============================================================
    "train_teacher": True,
    "train_student": True,
    "teacher_ckpt_path": None,
    "student_ckpt_path": None,

    # ============================================================
    # GAF FREQ
    # ============================================================
    "mu_band": (8, 12),
    "beta_band": (13, 30),

    # ============================================================
    # Device
    # ============================================================
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

KD_CFG["timepoints"] = int(KD_CFG["sfreq"] * (KD_CFG["tmax"] - KD_CFG["tmin"]))

HARD_KD_OVERRIDES = {
    "aug_prob": 0.7,
    "sr_prob": 0.7,
    "mixup_prob": 0.7,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_kd_subject_cfg(sub_id: int) -> dict:
    cfg = copy.copy(KD_CFG)
    if sub_id in HARD_SUBJECTS_KD:
        cfg.update(HARD_KD_OVERRIDES)
        print(f"  [KD Subject {sub_id}] HARD — aug_prob={cfg['aug_prob']}, "
              f"sr_prob={cfg['sr_prob']}, mixup_prob={cfg['mixup_prob']}")
    return cfg
