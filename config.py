# config.py — Configurazione globale, seed, hyperparametri
# ============================================================

import warnings, random, copy
import numpy as np
import torch

warnings.filterwarnings('ignore')

# ── Hard subjects ────────────────────────────────────────────
HARD_SUBJECTS = {2, 6}

# ── Configurazione globale ───────────────────────────────────
CFG = {
    # Dataset
    'n_subjects': 9,
    'sfreq': 250,
    'lowcut': 4.0,
    'highcut': 38.0,
    'tmin': 0.0,
    'tmax': 4.0,
    'n_channels': 22,
    'n_classes': 4,
    # GAF
    'image_size': 64,
    'gaf_method': 'summation',
    'downsample_to': 128,
    # Addestramento
    'batch_size': 32,
    'epochs': 1000,
    'lr': 9e-4,
    'patience': 500,
    'aug_prob': 0.5,
    # S&R Augmentation
    'sr_prob': 0.5,
    'n_segments': 10,
    # GAF aux loss
    'lambda_aux': 0.3,
    # TTA
    'n_tta': 5,
    # Mixup Within-class
    'mixup_prob': 0.5,
    'mixup_alpha': 0.4,
    'use_mixup': True,
    # LOSO flag (v14)
    'loso_approach': False,
    # Ablation GAF
    'use_gaf': True,
    # GAF@INFERENCE
    'gaf_inference': False,
    'gaf_inference_alpha': 0.75,
    # Cross-Attention (v15) — default False (AuxHead additiva)
    'cross_attention': False,
    'cross_attn_dk': 32,
    'cross_attn_dropout': 0.3,
    # ── [v16] FINE-TUNING POST-LOSO ──────────────────────────
    'loso_ft_approach': True,
    'ft_lr': 1e-4,
    'ft_epochs': 300,
    'ft_patience': 100,
    'ft_freeze_backbone': False,
    # ─────────────────────────────────────────────────────────
    # Multi-seed
    'multi_seed': False,
    'seeds': [42, 123, 456, 789, 1234],
    # TCFormer
    'F1': 16,
    'temp_kernel_lengths': (20, 32, 64),
    'D': 2,
    'pool_length_1': 8,
    'pool_length_2': 7,
    'dropout_conv': 0.3,
    'd_group': 16,
    'use_group_attn': True,
    'q_heads': 4,
    'kv_heads': 2,
    'trans_depth': 2,
    'trans_dropout': 0.4,
    'drop_path_max': 0.1,
    'tcn_depth': 2,
    'kernel_length_tcn': 4,
    'dropout_tcn': 0.3,
    # GAF AuxHead
    'gaf_aux_hidden': 64,
    'gaf_aux_dropout': 0.4,
    # Device
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

CFG['timepoints'] = int(CFG['sfreq'] * (CFG['tmax'] - CFG['tmin']))

HARD_CFG_OVERRIDES = {
    'lambda_aux': 0.5,
    'aug_prob': 0.7,
    'sr_prob': 0.7,
    'mixup_prob': 0.7,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_subject_cfg(sub_id: int) -> dict:
    cfg = copy.copy(CFG)
    if sub_id in HARD_SUBJECTS:
        cfg.update(HARD_CFG_OVERRIDES)
        print(f" [Subject {sub_id}] HARD — aug_prob={cfg['aug_prob']}, "
              f"sr_prob={cfg['sr_prob']}, mixup_prob={cfg['mixup_prob']}")
    return cfg


def _mode_tag(cfg: dict) -> str:
    base  = "EEG-only" if not cfg['use_gaf'] else "TCFormer+GAF"
    mixup = "Mixup=ON" if cfg['use_mixup'] else "Mixup=OFF"
    if cfg.get('loso_ft_approach', False):
        proto = "LOSO+FT"
    elif cfg['loso_approach']:
        proto = "LOSO"
    else:
        proto = "Within-Subject"
    xattn = "XAttn=ON" if (cfg['use_gaf'] and cfg.get('cross_attention', False)) else "XAttn=OFF"
    return f"{base} | {mixup} | {proto} | {xattn}"


# ── Inizializzazione al primo import ─────────────────────────
set_seed(42)
n_seeds = len(CFG['seeds']) if CFG['multi_seed'] else 1
print(f"Device: {CFG['device']}")
print(f"Timepoints: {CFG['timepoints']}")
print(f"Mode: {_mode_tag(CFG)}")
print(f"Multi-seed: {CFG['multi_seed']} → {n_seeds} seed(s): "
      f"{CFG['seeds'] if CFG['multi_seed'] else [42]}")