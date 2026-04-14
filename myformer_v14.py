# ============================================================
# myformer_v14.py — TCFormer + ECA + S&R + Mixup + Multi-Seed
#                   + LOSO / Within-Subject selezionabile
# ============================================================
# CAMBIAMENTI v13 → v14
#
# [LOSO FLAG] Aggiunto 'loso_approach' (bool) in CFG.
#
#   loso_approach=False → Within-Subject (comportamento v13)
#     Train: sessione T del soggetto target
#     Test:  sessione E del soggetto target
#
#   loso_approach=True  → Leave-One-Subject-Out (cross-subject)
#     Train: sessioni T+E di tutti i soggetti TRANNE il target
#     Test:  sessione E del soggetto target
#     → nessun dato del soggetto target visto durante il training
#     → valuta la generalizzazione cross-subject del modello
#
# Strategia LOSO efficiente:
#   - I dati di tutti i 9 soggetti vengono caricati UNA SOLA
#     VOLTA all'inizio (cache globale), poi riusati per ogni fold.
#   - Per ogni fold s: train = concat(tutti tranne s), test = E(s)
#   - Multi-seed: per ogni fold si eseguono N seed indipendenti
#
# Modifiche applicate (tutto il resto invariato da v13):
#   1. CFG: aggiunto 'loso_approach': False
#   2. _mode_tag: mostra LOSO/Within-Subject
#   3. load_subject_both: carica T+E separati (usato da LOSO)
#   4. build_loso_cache: carica tutti i soggetti una sola volta
#   5. run_loso_fold: training LOSO per un singolo fold (soggetto)
#   6. run_loso_fold_multiseed: wrapper multi-seed per LOSO
#   7. Entry point: biforcato su loso_approach
#   8. Tabella finale: stessa struttura, con tag [LOSO] / [WS]
# ============================================================

import warnings, random, copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm.auto import tqdm

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from pyts.image import GramianAngularField

warnings.filterwarnings('ignore')

# ============================================================
# Seed & Config
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

HARD_SUBJECTS = {2, 6}

CFG = {
    # Dataset
    'n_subjects':  9,
    'sfreq':       250,
    'lowcut':      4.0,
    'highcut':     38.0,
    'tmin':        0.0,
    'tmax':        4.0,
    'n_channels':  22,
    'n_classes':   4,
    # GAF
    'image_size':     64,
    'gaf_method':     'summation',
    'downsample_to':  128,
    # Addestramento
    'batch_size':  32,
    'epochs':      1000,
    'lr':          9e-4,
    'patience':    500,
    'aug_prob':    0.5,
    # S&R Augmentation
    'sr_prob':       0.5,
    'n_segments':    10,
    # GAF aux loss
    'lambda_aux':    0.3,
    # TTA
    'n_tta': 5,
    # Mixup Within-class
    'mixup_prob':  0.5,
    'mixup_alpha': 0.4,
    # Mixup flag (v13)
    'use_mixup': True,
    # ── [v14] LOSO FLAG ──────────────────────────────────────
    'loso_approach': True,   # False = Within-Subject (v13)
                              # True  = Leave-One-Subject-Out
    # ─────────────────────────────────────────────────────────
    # Ablation GAF
    'use_gaf': False,
    # GAF@INFERENCE
    'gaf_inference':       False,
    'gaf_inference_alpha': 0.75,
    # Multi-seed
    'multi_seed': True,
    'seeds':      [42, 123, 456, 789, 1234],
    # TCFormer (paper-exact)
    'F1':                  16,
    'temp_kernel_lengths': (20, 32, 64),
    'D':                   2,
    'pool_length_1':       8,
    'pool_length_2':       7,
    'dropout_conv':        0.3,
    'd_group':             16,
    'use_group_attn':      True,
    'q_heads':             4,
    'kv_heads':            2,
    'trans_depth':         2,
    'trans_dropout':       0.4,
    'drop_path_max':       0.1,
    'tcn_depth':           2,
    'kernel_length_tcn':   4,
    'dropout_tcn':         0.3,
    # GAF AuxHead
    'gaf_aux_hidden':  64,
    'gaf_aux_dropout': 0.4,
    # Device
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}
CFG['timepoints'] = int(CFG['sfreq'] * (CFG['tmax'] - CFG['tmin']))

HARD_CFG_OVERRIDES = {
    'lambda_aux':  0.5,
    'aug_prob':    0.7,
    'sr_prob':     0.7,
    'mixup_prob':  0.7,
}

def get_subject_cfg(sub_id: int) -> dict:
    cfg = copy.copy(CFG)
    if sub_id in HARD_SUBJECTS:
        cfg.update(HARD_CFG_OVERRIDES)
        print(f"  [Subject {sub_id}] HARD — aug_prob={cfg['aug_prob']}, "
              f"sr_prob={cfg['sr_prob']}, mixup_prob={cfg['mixup_prob']}")
    return cfg

def _mode_tag(cfg):
    base   = "EEG-only" if not cfg['use_gaf'] else "TCFormer+GAF"
    mixup  = "Mixup=ON" if cfg['use_mixup'] else "Mixup=OFF"
    proto  = "LOSO" if cfg['loso_approach'] else "Within-Subject"
    return f"{base} | {mixup} | {proto}"

n_seeds = len(CFG['seeds']) if CFG['multi_seed'] else 1
print(f"Device:     {CFG['device']}")
print(f"Timepoints: {CFG['timepoints']}")
print(f"Mode:       {_mode_tag(CFG)}")
print(f"Multi-seed: {CFG['multi_seed']} → {n_seeds} seed(s): "
      f"{CFG['seeds'] if CFG['multi_seed'] else [42]}")

# ============================================================
# Preprocessing (invariato da v13)
# ============================================================

def zscore(X):
    m = X.mean(axis=-1, keepdims=True)
    s = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - m) / s).astype(np.float32)

def minmax(X):
    lo = X.min(axis=-1, keepdims=True)
    hi = X.max(axis=-1, keepdims=True)
    return ((X - lo) / (hi - lo + 1e-8)).astype(np.float32)

def downsample(X, n):
    from scipy.signal import resample
    return resample(X, n, axis=-1).astype(np.float32) if X.shape[-1] != n else X.astype(np.float32)

def windows_to_numpy(ds):
    Xl, yl = [], []
    for X, y, _ in ds:
        Xl.append(X); yl.append(y)
    return np.stack(Xl).astype(np.float32), np.array(yl, dtype=np.int64)

def load_subject(sub_id, cfg):
    """Within-subject: ritorna (X_tr, y_tr, X_te, y_te) — sessioni T e E."""
    ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sub_id])
    preprocess(ds, [
        Preprocessor('pick', picks='eeg'),
        Preprocessor(lambda x: x * 1e6),
        Preprocessor('filter', l_freq=cfg['lowcut'], h_freq=cfg['highcut']),
        Preprocessor('resample', sfreq=cfg['sfreq']),
    ])
    wins = create_windows_from_events(ds, trial_start_offset_samples=0,
                                      trial_stop_offset_samples=0, preload=True)
    sp = wins.split('session')
    tk = [k for k in sp if 'train' in k.lower() or k == 'T'][0]
    ek = [k for k in sp if 'test'  in k.lower() or k == 'E'][0]
    return windows_to_numpy(sp[tk]) + windows_to_numpy(sp[ek])

# ── [v14] Caricamento separato T / E per LOSO ───────────────

def load_subject_both(sub_id, cfg):
    """
    LOSO: carica sessioni T ed E separatamente.
    Ritorna dict con chiavi 'T' e 'E', ognuna (X, y) preprocessata.
    """
    ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sub_id])
    preprocess(ds, [
        Preprocessor('pick', picks='eeg'),
        Preprocessor(lambda x: x * 1e6),
        Preprocessor('filter', l_freq=cfg['lowcut'], h_freq=cfg['highcut']),
        Preprocessor('resample', sfreq=cfg['sfreq']),
    ])
    wins = create_windows_from_events(ds, trial_start_offset_samples=0,
                                      trial_stop_offset_samples=0, preload=True)
    sp = wins.split('session')
    tk = [k for k in sp if 'train' in k.lower() or k == 'T'][0]
    ek = [k for k in sp if 'test'  in k.lower() or k == 'E'][0]
    X_t, y_t = windows_to_numpy(sp[tk])
    X_e, y_e = windows_to_numpy(sp[ek])
    # zscore indipendente per ogni sessione
    return {
        'T': (zscore(X_t), y_t),
        'E': (zscore(X_e), y_e),
    }

def make_gaf(X_raw, cfg):
    gaf = GramianAngularField(image_size=cfg['image_size'], method=cfg['gaf_method'])
    X_ds = downsample(X_raw, cfg['downsample_to'])
    X_ds = minmax(X_ds)
    B, C, _ = X_ds.shape
    out = np.zeros((B, C, cfg['image_size'], cfg['image_size']), dtype=np.float32)
    for i in tqdm(range(B), desc='GAF', leave=False):
        out[i] = gaf.fit_transform(X_ds[i])
    return out

def preprocess_subject(X_tr_raw, X_te_raw, cfg):
    X_tr_t = zscore(X_tr_raw)
    X_te_t = zscore(X_te_raw)
    if cfg['use_gaf']:
        X_tr_g = make_gaf(X_tr_raw, cfg)
        X_te_g = make_gaf(X_te_raw, cfg)
    else:
        X_tr_g = np.zeros((len(X_tr_t), 1, 1, 1), dtype=np.float32)
        X_te_g = np.zeros((len(X_te_t), 1, 1, 1), dtype=np.float32)
    return X_tr_t, X_te_t, X_tr_g, X_te_g

# ── [v14] Cache globale per LOSO ────────────────────────────

def build_loso_cache(cfg):
    """
    Carica tutti i 9 soggetti UNA SOLA VOLTA.
    Ritorna dict: {sub_id: {'T': (X,y), 'E': (X,y)}}
    Il preprocessing (zscore) è già applicato internamente
    da load_subject_both.
    """
    cache = {}
    for s in range(1, cfg['n_subjects'] + 1):
        print(f"  [LOSO cache] Caricamento soggetto {s}/{cfg['n_subjects']}...")
        cache[s] = load_subject_both(s, cfg)
    print(f"  [LOSO cache] ✅ Tutti i {cfg['n_subjects']} soggetti in memoria.\n")
    return cache

# ============================================================
# S&R Augmentation (invariata da v13)
# ============================================================

def segment_and_reconstruct(x_t, y, n_segments=10, sr_prob=0.5):
    B, C, T = x_t.shape
    x_aug   = x_t.clone()
    seg_len = T // n_segments
    for i in range(B):
        if random.random() > sr_prob:
            continue
        label_i    = y[i].item()
        same_class = [j for j in range(B) if y[j].item() == label_i and j != i]
        if len(same_class) < n_segments:
            continue
        donors = random.sample(same_class, n_segments)
        for seg_idx in range(n_segments):
            start = seg_idx * seg_len
            end   = start + seg_len if seg_idx < n_segments - 1 else T
            x_aug[i, :, start:end] = x_t[donors[seg_idx], :, start:end]
    return x_aug

# ============================================================
# Mixup Within-class (invariato da v13)
# ============================================================

def mixup_batch(x_t, y, n_classes, mixup_prob=0.5, alpha=0.4):
    B = x_t.shape[0]
    y_oh   = F.one_hot(y, n_classes).float()
    x_mix  = x_t.clone()
    y_soft = y_oh.clone()
    for i in range(B):
        if random.random() > mixup_prob:
            continue
        label_i    = y[i].item()
        same_class = [j for j in range(B) if y[j].item() == label_i and j != i]
        if not same_class:
            continue
        j   = random.choice(same_class)
        lam = float(np.random.beta(alpha, alpha))
        lam = max(lam, 1 - lam)
        x_mix[i]  = lam * x_t[i]  + (1 - lam) * x_t[j]
        y_soft[i] = lam * y_oh[i] + (1 - lam) * y_oh[j]
    return x_mix, y_soft

# ============================================================
# Dataset (invariato da v13)
# ============================================================

class EEGAug:
    @staticmethod
    def noise(x): return x + torch.randn_like(x) * 0.03 * x.std()
    @staticmethod
    def shift(x): return torch.roll(x, torch.randint(-12, 13, (1,)).item(), dims=-1)
    @staticmethod
    def scale(x): return x * torch.FloatTensor(1).uniform_(0.85, 1.15)

class MMDataset(Dataset):
    def __init__(self, X_t, X_g, y, augment=False, aug_prob=0.5):
        self.X_t      = torch.tensor(X_t)
        self.X_g      = torch.tensor(X_g)
        self.y        = torch.tensor(y, dtype=torch.long)
        self.augment  = augment
        self.aug_prob = aug_prob

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x_t = self.X_t[i].clone()
        x_g = self.X_g[i].clone()
        if self.augment:
            p = self.aug_prob
            if torch.rand(1) < p: x_t = EEGAug.noise(x_t)
            if torch.rand(1) < p: x_t = EEGAug.shift(x_t)
            if torch.rand(1) < p: x_t = EEGAug.scale(x_t)
        return {'eeg': x_t, 'gaf': x_g, 'label': self.y[i]}

# ============================================================
# Utility (invariato da v13)
# ============================================================

def glorot_zero(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, **kw):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation, **kw)
        self._causal_pad = pad
    def forward(self, x):
        out = super().forward(x)
        return out[..., :-self._causal_pad] if self._causal_pad > 0 else out

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm=1.0, **kw):
        self.max_norm = max_norm
        super().__init__(*args, **kw)
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

# ============================================================
# ECA, ChannelGroupAttention, MultiKernelConvBlock (invariati)
# ============================================================

class ECABlock2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        k = int(abs(math.log2(channels) + 1))
        k = k if k % 2 == 1 else k + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.sigmoid(self.conv(y))
        return x * y.transpose(-1, -2).unsqueeze(-1).expand_as(x)

class ChannelGroupAttention(nn.Module):
    def __init__(self, in_channels, num_groups):
        super().__init__()
        assert in_channels % num_groups == 0
        self.num_groups = num_groups
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(in_channels, num_groups), nn.Softmax(dim=-1),
        )
        glorot_zero(self)

    def forward(self, x):
        B, C, T = x.shape
        w   = self.attn(x).view(B, self.num_groups, 1, 1)
        g   = C // self.num_groups
        x_g = x.view(B, self.num_groups, g, T)
        return (x_g * w).view(B, C, T)

class MultiKernelConvBlock(nn.Module):
    def __init__(self, n_channels, temp_kernel_lengths=(20, 32, 64),
                 F1=16, D=2, pool_length_1=8, pool_length_2=7,
                 dropout=0.3, d_group=16, use_group_attn=True):
        super().__init__()
        from einops.layers.torch import Rearrange
        self.rearrange = Rearrange("b c seq -> b 1 c seq")
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad2d(
                    (k // 2 - 1, k // 2, 0, 0) if k % 2 == 0
                    else (k // 2, k // 2, 0, 0), 0),
                nn.Conv2d(1, F1, (1, k), bias=False),
                nn.BatchNorm2d(F1),
            ) for k in temp_kernel_lengths
        ])
        n_groups     = len(temp_kernel_lengths)
        self.d_model = d_group * n_groups
        F2           = F1 * n_groups * D
        self.channel_DW_conv = nn.Sequential(
            nn.Conv2d(F1 * n_groups, F2, (n_channels, 1),
                      bias=False, groups=F1 * n_groups),
            nn.BatchNorm2d(F2), nn.ELU(),
        )
        self.pool1   = nn.AvgPool2d((1, pool_length_1))
        self.drop1   = nn.Dropout(dropout)
        self.eca     = ECABlock2d(F2)
        self.use_cr2 = (self.d_model != F2)
        if self.use_cr2:
            self.channel_reduction_2 = nn.Sequential(
                nn.Conv2d(F2, self.d_model, (1, 1), bias=False, groups=n_groups),
                nn.BatchNorm2d(self.d_model),
            )
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, (1, 16),
                      padding='same', bias=False, groups=n_groups),
            nn.BatchNorm2d(self.d_model), nn.ELU(),
        )
        self.use_group_attn = use_group_attn and n_groups > 1
        if self.use_group_attn:
            self.group_attn = ChannelGroupAttention(self.d_model, n_groups)
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(dropout)
        glorot_zero(self)

    def forward(self, x):
        x = self.rearrange(x)
        x = torch.cat([c(x) for c in self.temporal_convs], dim=1)
        x = self.drop1(self.pool1(self.channel_DW_conv(x)))
        x = self.eca(x)
        if self.use_cr2:
            x = self.channel_reduction_2(x)
        x = self.temporal_conv_2(x)
        if self.use_group_attn:
            xs = x.squeeze(2)
            xs = xs + self.group_attn(xs)
            x  = xs.unsqueeze(2)
        return self.drop2(self.pool2(x)).squeeze(2)

# ============================================================
# RoPE + GQA + DropPath + TransformerBlock + TCN (invariati)
# ============================================================

def _build_rope_cache(head_dim, seq_len, device):
    theta = 1.0 / (10000 ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    idx   = torch.arange(seq_len, device=device).float()
    emb   = torch.cat((torch.outer(idx, theta),) * 2, dim=-1)
    return emb.cos(), emb.sin()

def _rope_rotate(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def _apply_rope(q, k, cos, sin):
    return (q * cos) + (_rope_rotate(q) * sin), \
           (k * cos) + (_rope_rotate(k) * sin)

class GQAttention(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads, dropout=0.3):
        super().__init__()
        assert d_model % q_heads == 0 and q_heads % kv_heads == 0
        self.qh, self.kvh = q_heads, kv_heads
        self.hd    = d_model // q_heads
        self.scale = self.hd ** -0.5
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * kv_heads * self.hd, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)
        glorot_zero(self)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q  = self.q_proj(x).view(B, T, self.qh, self.hd).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, self.kvh, 2, self.hd)
        k  = kv[..., 0, :].transpose(1, 2).repeat_interleave(self.qh // self.kvh, dim=1)
        v  = kv[..., 1, :].transpose(1, 2).repeat_interleave(self.qh // self.kvh, dim=1)
        q, k = _apply_rope(q, k, cos[:T], sin[:T])
        attn = self.drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.o_proj((attn @ v).transpose(1, 2).contiguous().view(B, T, C))

class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p = p
    def forward(self, x):
        if self.p == 0 or not self.training: return x
        kp    = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x / kp * (kp + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()

class TransformerBlock(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads, dropout=0.4, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = GQAttention(d_model, q_heads, kv_heads, dropout)
        self.dp    = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout),
        )
    def forward(self, x, cos, sin):
        x = x + self.dp(self.attn(self.norm1(x), cos, sin))
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x

class TCNBlock(nn.Module):
    def __init__(self, nf, ks=4, dil=1, ng=1, dropout=0.3):
        super().__init__()
        self.c1 = CausalConv1d(nf, nf, ks, dil, groups=ng)
        self.b1 = nn.BatchNorm1d(nf); self.e1 = nn.ELU(); self.d1 = nn.Dropout(dropout)
        self.c2 = CausalConv1d(nf, nf, ks, dil, groups=ng)
        self.b2 = nn.BatchNorm1d(nf); self.e2 = nn.ELU(); self.d2 = nn.Dropout(dropout)
        self.act = nn.ELU()
        nn.init.constant_(self.c1.bias, 0)
        nn.init.constant_(self.c2.bias, 0)

    def forward(self, x):
        h = self.d1(self.e1(self.b1(self.c1(x))))
        h = self.d2(self.e2(self.b2(self.c2(h))))
        return self.act(x + h)

class TCNHead(nn.Module):
    def __init__(self, d, n_groups, n_classes, depth=2, ks=4, dropout=0.3):
        super().__init__()
        self.ng = n_groups; self.nc = n_classes
        self.tcn = nn.Sequential(
            *[TCNBlock(d, ks, 2 ** i, n_groups, dropout) for i in range(depth)])
        self.cls = Conv1dWithConstraint(
            d, n_classes * n_groups, 1, groups=n_groups, max_norm=0.25)

    def forward(self, x):
        x = self.tcn(x)[..., -1:]
        x = self.cls(x).squeeze(-1)
        return x.view(x.size(0), self.ng, self.nc).mean(1)

# ============================================================
# TCFormerModule (invariato da v13)
# ============================================================

class TCFormerModule(nn.Module):
    def __init__(self, n_channels, n_classes,
                 F1=16, temp_kernel_lengths=(20, 32, 64),
                 D=2, pool_length_1=8, pool_length_2=7,
                 dropout_conv=0.3, d_group=16, use_group_attn=True,
                 q_heads=4, kv_heads=2,
                 trans_depth=2, trans_dropout=0.4, drop_path_max=0.1,
                 tcn_depth=2, kernel_length_tcn=4, dropout_tcn=0.3):
        super().__init__()
        from einops.layers.torch import Rearrange
        n_groups      = len(temp_kernel_lengths)
        self.d_model  = d_group * n_groups
        self.d_group  = d_group
        self.n_groups = n_groups
        self.conv_block = MultiKernelConvBlock(
            n_channels, temp_kernel_lengths, F1, D,
            pool_length_1, pool_length_2, dropout_conv, d_group, use_group_attn,
        )
        self.mix    = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, 1, bias=False),
            nn.BatchNorm1d(self.d_model), nn.SiLU(),
        )
        self.to_seq = Rearrange("b c t -> b t c")
        dpr = (torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max).tolist()
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, q_heads, kv_heads, trans_dropout, dpr[i])
            for i in range(trans_depth)
        ])
        self.reduce = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(self.d_model, d_group, 1, bias=False),
            nn.BatchNorm1d(d_group), nn.SiLU(),
        )
        d_tcf         = d_group * (n_groups + 1)
        self.tcn_head = TCNHead(d_tcf, n_groups + 1, n_classes,
                                tcn_depth, kernel_length_tcn, dropout_tcn)
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        glorot_zero(self)

    def _rope_cache(self, T, dev):
        hd = self.transformer[0].attn.hd
        if self._cos is None or self._cos.shape[0] < T:
            self._cos, self._sin = _build_rope_cache(hd, T, dev)
        return self._cos, self._sin

    def get_features(self, x):
        conv_f   = self.conv_block(x)
        B, C, T  = conv_f.shape
        tok      = self.to_seq(self.mix(conv_f))
        cos, sin = self._rope_cache(T, x.device)
        for blk in self.transformer:
            tok = blk(tok, cos, sin)
        tran_f = self.reduce(tok)
        return torch.cat((conv_f, tran_f), dim=1)

    def forward(self, x):
        return self.tcn_head(self.get_features(x))

# ============================================================
# GAF MiniEncoder + AuxHead (invariati da v13)
# ============================================================

class GAFMiniEncoder(nn.Module):
    def __init__(self, n_channels=22, out_dim=64, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim),
            nn.ELU(), nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.pool_ch = nn.Sequential(
            nn.Linear(out_dim * 4, out_dim), nn.ELU(), nn.Dropout(dropout))
        glorot_zero(self)

    def forward(self, x):
        B, C, H, W = x.shape
        f = self.cnn(x.view(B * C, 1, H, W)).view(B, C, -1)
        return self.pool_ch(f.mean(1))

class AuxHead(nn.Module):
    def __init__(self, d_tcf, gaf_dim, n_classes, hidden=64, dropout=0.4):
        super().__init__()
        self.tcf_proj = nn.Sequential(nn.Linear(d_tcf, hidden), nn.LayerNorm(hidden))
        self.gaf_proj = nn.Sequential(nn.Linear(gaf_dim, hidden), nn.LayerNorm(hidden))
        self.cls = nn.Sequential(nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        glorot_zero(self)

    def forward(self, tcf_feat, gaf_emb):
        return self.cls(self.tcf_proj(tcf_feat.mean(-1)) + self.gaf_proj(gaf_emb))

# ============================================================
# Modello Completo (invariato da v13)
# ============================================================

class TCFormerWithAux(nn.Module):
    def __init__(self, n_channels, n_classes, cfg):
        super().__init__()
        self.use_gaf = cfg['use_gaf']
        self.tcformer = TCFormerModule(
            n_channels=n_channels, n_classes=n_classes,
            F1=cfg['F1'], temp_kernel_lengths=cfg['temp_kernel_lengths'],
            D=cfg['D'], pool_length_1=cfg['pool_length_1'],
            pool_length_2=cfg['pool_length_2'], dropout_conv=cfg['dropout_conv'],
            d_group=cfg['d_group'], use_group_attn=cfg['use_group_attn'],
            q_heads=cfg['q_heads'], kv_heads=cfg['kv_heads'],
            trans_depth=cfg['trans_depth'], trans_dropout=cfg['trans_dropout'],
            drop_path_max=cfg['drop_path_max'], tcn_depth=cfg['tcn_depth'],
            kernel_length_tcn=cfg['kernel_length_tcn'], dropout_tcn=cfg['dropout_tcn'],
        )
        if self.use_gaf:
            n_groups = len(cfg['temp_kernel_lengths'])
            d_tcf    = cfg['d_group'] * (n_groups + 1)
            gaf_dim  = cfg['gaf_aux_hidden']
            self.gaf_enc  = GAFMiniEncoder(n_channels, gaf_dim, cfg['gaf_aux_dropout'])
            self.aux_head = AuxHead(d_tcf, gaf_dim, n_classes,
                                    cfg['gaf_aux_hidden'], cfg['gaf_aux_dropout'])

    def forward(self, eeg, gaf=None):
        if self.use_gaf and gaf is not None:
            feat        = self.tcformer.get_features(eeg)
            logits_main = self.tcformer.tcn_head(feat)
            logits_aux  = self.aux_head(feat.detach(), self.gaf_enc(gaf))
            return logits_main, logits_aux
        return self.tcformer(eeg)

def build_model(cfg):
    return TCFormerWithAux(cfg['n_channels'], cfg['n_classes'], cfg)

# ============================================================
# Trainer (invariato da v13)
# ============================================================

class DistillationTrainer:
    def __init__(self, model, cfg):
        self.model      = model.to(cfg['device'])
        self.device     = cfg['device']
        self.cfg        = cfg
        self.use_gaf    = cfg['use_gaf']
        self.use_mixup  = cfg['use_mixup']
        self.ce         = nn.CrossEntropyLoss()
        self.opt        = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        self.sched      = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg['epochs'], eta_min=1e-5)
        self.best_acc   = 0.0
        self.best_state = None
        self.history    = dict(tr_loss=[], va_loss=[], tr_acc=[], va_acc=[], va_kap=[])

    def _lambda(self): return self.cfg['lambda_aux']

    def _epoch(self, loader, train):
        self.model.train(train)
        tot_loss, preds, trues = 0.0, [], []
        lam       = self._lambda()
        use_gaf   = self.use_gaf
        use_mixup = self.use_mixup
        with torch.set_grad_enabled(train):
            for b in loader:
                eeg = b['eeg'].to(self.device)
                gaf = b['gaf'].to(self.device)
                y   = b['label'].to(self.device)
                if train:
                    eeg = segment_and_reconstruct(eeg, y,
                        n_segments=self.cfg['n_segments'],
                        sr_prob=self.cfg['sr_prob'])
                    self.opt.zero_grad()
                    if use_mixup:
                        eeg, y_soft = mixup_batch(eeg, y,
                            n_classes=self.cfg['n_classes'],
                            mixup_prob=self.cfg['mixup_prob'],
                            alpha=self.cfg['mixup_alpha'])
                        if use_gaf:
                            logits_m, logits_a = self.model(eeg, gaf)
                            loss_m = -(y_soft * F.log_softmax(logits_m, -1)).sum(-1).mean()
                            loss_a = -(y_soft * F.log_softmax(logits_a, -1)).sum(-1).mean()
                            loss   = loss_m + lam * loss_a
                        else:
                            logits_m = self.model(eeg)
                            loss     = -(y_soft * F.log_softmax(logits_m, -1)).sum(-1).mean()
                    else:
                        if use_gaf:
                            logits_m, logits_a = self.model(eeg, gaf)
                            loss = self.ce(logits_m, y) + lam * self.ce(logits_a, y)
                        else:
                            logits_m = self.model(eeg)
                            loss     = self.ce(logits_m, y)
                    loss.backward(); self.opt.step()
                else:
                    logits_m = self.model(eeg, gaf=None) if use_gaf else self.model(eeg)
                    loss     = self.ce(logits_m, y)
                tot_loss += loss.item()
                preds.extend(logits_m.argmax(-1).cpu().numpy())
                trues.extend(y.cpu().numpy())
        acc   = accuracy_score(trues, preds) * 100
        kappa = cohen_kappa_score(trues, preds)
        return tot_loss / max(len(loader), 1), acc, kappa

    def fit(self, tr_ld, va_ld, seed=42):
        epochs = self.cfg['epochs']; pat = self.cfg['patience']; wait = 0
        print(f"\n{'='*65}")
        print(f"Mode: {_mode_tag(self.cfg)} | seed={seed} | "
              f"epochs={epochs} | patience={pat}")
        print(f"{'='*65}")
        for ep in range(epochs):
            tr_loss, tr_acc, _    = self._epoch(tr_ld, True)
            va_loss, va_acc, va_k = self._epoch(va_ld, False)
            self.sched.step()
            lr = self.opt.param_groups[0]['lr']
            for k, v in zip(['tr_loss','va_loss','tr_acc','va_acc','va_kap'],
                            [tr_loss, va_loss, tr_acc, va_acc, va_k]):
                self.history[k].append(v)
            if va_acc > self.best_acc:
                self.best_acc   = va_acc
                self.best_state = copy.deepcopy(self.model.state_dict())
                wait = 0; tag = ' ✨ BEST'
            else:
                wait += 1; tag = f' [{wait}/{pat}]'
            lam_str = f"λ={self._lambda():.3f} | " if self.use_gaf else ""
            print(f"Ep {ep+1:03d}/{epochs} | {lam_str}LR {lr:.1e} "
                  f"| Tr {tr_acc:.1f}% {tr_loss:.4f} "
                  f"| Va {va_acc:.1f}% {va_loss:.4f} k={va_k:.3f}{tag}")
            if wait >= pat:
                print(f"\n🛑 Early stop ep {ep+1}"); break
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        print(f"\n🏆 Best val: {self.best_acc:.2f}%")
        return self.history

# ============================================================
# evaluate (invariato da v13)
# ============================================================

def _tta_augment(x):
    x = x + torch.randn_like(x) * 0.02 * x.std()
    return x * torch.FloatTensor(1).uniform_(0.9, 1.1).to(x.device)

def evaluate(model, loader, device, cfg):
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

# ============================================================
# Within-Subject: run_subject / run_subject_multiseed (da v13)
# ============================================================

def run_subject(sub_id, seed=42, verbose=True):
    cfg = get_subject_cfg(sub_id)
    if verbose:
        print(f"\n{'─'*55}\nSoggetto {sub_id} | seed={seed}\n{'─'*55}")
    X_tr_raw, y_tr, X_te_raw, y_te = load_subject(sub_id, cfg)
    X_tr_t, X_te_t, X_tr_g, X_te_g = preprocess_subject(X_tr_raw, X_te_raw, cfg)
    tr_ds = MMDataset(X_tr_t, X_tr_g, y_tr, augment=True,  aug_prob=cfg['aug_prob'])
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
    cfg   = get_subject_cfg(sub_id)
    seeds = cfg['seeds']
    print(f"\n{'═'*65}")
    print(f"[WS] Soggetto {sub_id} | Multi-seed ({len(seeds)} run): {seeds}")
    print(f"{'═'*65}")
    X_tr_raw, y_tr, X_te_raw, y_te = load_subject(sub_id, cfg)
    X_tr_t, X_te_t, X_tr_g, X_te_g = preprocess_subject(X_tr_raw, X_te_raw, cfg)
    tr_ds = MMDataset(X_tr_t, X_tr_g, y_tr, augment=True,  aug_prob=cfg['aug_prob'])
    te_ds = MMDataset(X_te_t, X_te_g, y_te, augment=False, aug_prob=0.0)
    tr_ld = DataLoader(tr_ds, cfg['batch_size'], shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, cfg['batch_size'], shuffle=False, num_workers=0)
    seed_accs, seed_kappas = [], []
    for i, seed in enumerate(seeds):
        print(f"\n  ── Seed {i+1}/{len(seeds)}: {seed} ──")
        set_seed(seed)
        model   = build_model(cfg)
        trainer = DistillationTrainer(model, cfg)
        trainer.fit(tr_ld, te_ld, seed=seed)
        y_true, y_pred = evaluate(trainer.model, te_ld, cfg['device'], cfg)
        acc   = accuracy_score(y_true, y_pred) * 100
        kappa = cohen_kappa_score(y_true, y_pred)
        seed_accs.append(acc); seed_kappas.append(kappa)
        print(f"  ✓ seed={seed} → Acc: {acc:.2f}% | Kappa: {kappa:.4f}")
        del model, trainer; torch.cuda.empty_cache()
    mean_acc, std_acc     = np.mean(seed_accs),   np.std(seed_accs)
    mean_kappa, std_kappa = np.mean(seed_kappas), np.std(seed_kappas)
    print(f"\n  📊 S{sub_id:02d} [WS] → {mean_acc:.2f} ± {std_acc:.2f}% | κ={mean_kappa:.4f} ± {std_kappa:.4f}")
    return mean_acc, std_acc, mean_kappa, std_kappa, seed_accs

# ============================================================
# [v14] LOSO: run_loso_fold / run_loso_fold_multiseed
# ============================================================

def _make_dummy_gaf(n):
    return np.zeros((n, 1, 1, 1), dtype=np.float32)

def run_loso_fold(target_sub, cache, seed=42, verbose=True):
    """
    Singolo fold LOSO per target_sub con un seed specifico.
    cache: dict {sub_id: {'T': (X,y), 'E': (X,y)}} — già zscorato.
    Train = T+E di tutti i soggetti TRANNE target_sub.
    Test  = E di target_sub.
    """
    cfg = get_subject_cfg(target_sub)
    if verbose:
        print(f"\n{'─'*55}")
        print(f"[LOSO] Fold S{target_sub:02d} (test) | seed={seed}")
        print(f"{'─'*55}")

    # Costruisci train set: T+E di tutti i soggetti tranne il target
    tr_X_list, tr_y_list = [], []
    for s, data in cache.items():
        if s == target_sub:
            continue
        X_t, y_t = data['T']
        X_e, y_e = data['E']
        tr_X_list.append(X_t); tr_y_list.append(y_t)
        tr_X_list.append(X_e); tr_y_list.append(y_e)
    X_tr = np.concatenate(tr_X_list, axis=0)
    y_tr = np.concatenate(tr_y_list, axis=0)

    # Test set: E del soggetto target
    X_te, y_te = cache[target_sub]['E']

    # Dummy GAF (use_gaf=False)
    X_tr_g = _make_dummy_gaf(len(X_tr))
    X_te_g = _make_dummy_gaf(len(X_te))

    tr_ds = MMDataset(X_tr, X_tr_g, y_tr, augment=True,  aug_prob=cfg['aug_prob'])
    te_ds = MMDataset(X_te, X_te_g, y_te, augment=False, aug_prob=0.0)
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
    """
    Multi-seed LOSO per un singolo fold (soggetto target).
    La cache è condivisa e non viene ricaricata.
    """
    cfg   = get_subject_cfg(target_sub)
    seeds = cfg['seeds']
    print(f"\n{'═'*65}")
    print(f"[LOSO] Fold S{target_sub:02d} | Multi-seed ({len(seeds)} run): {seeds}")
    print(f"{'═'*65}")

    seed_accs, seed_kappas = [], []
    for i, seed in enumerate(seeds):
        print(f"\n  ── Seed {i+1}/{len(seeds)}: {seed} ──")
        acc, kappa = run_loso_fold(target_sub, cache, seed=seed, verbose=False)
        seed_accs.append(acc); seed_kappas.append(kappa)
        print(f"  ✓ seed={seed} → Acc: {acc:.2f}% | Kappa: {kappa:.4f}")
        torch.cuda.empty_cache()

    mean_acc, std_acc     = np.mean(seed_accs),   np.std(seed_accs)
    mean_kappa, std_kappa = np.mean(seed_kappas), np.std(seed_kappas)
    print(f"\n  📊 S{target_sub:02d} [LOSO] → {mean_acc:.2f} ± {std_acc:.2f}% | "
          f"κ={mean_kappa:.4f} ± {std_kappa:.4f}")
    return mean_acc, std_acc, mean_kappa, std_kappa, seed_accs

# ============================================================
# Entry point — [v14] biforcato su loso_approach
# ============================================================

SUBJECT_DEBUG = 1
RUN_ALL       = True

if not RUN_ALL:
    seed = CFG['seeds'][0] if CFG['multi_seed'] else 42
    if CFG['loso_approach']:
        cache = build_loso_cache(CFG)
        run_loso_fold(SUBJECT_DEBUG, cache, seed=seed)
    else:
        run_subject(SUBJECT_DEBUG, seed=seed)

else:
    results = {}
    proto_tag = "LOSO" if CFG['loso_approach'] else "WS"

    if CFG['loso_approach']:
        # ── LOSO: carica tutti i soggetti UNA SOLA VOLTA ────
        print("\n[v14] Modalità LOSO — caricamento cache globale...")
        cache = build_loso_cache(CFG)

        if CFG['multi_seed']:
            for s in range(1, CFG['n_subjects'] + 1):
                mean_acc, std_acc, mean_kappa, std_kappa, seed_accs = \
                    run_loso_fold_multiseed(s, cache)
                results[s] = {'acc': mean_acc, 'std': std_acc,
                              'kappa': mean_kappa, 'kappa_std': std_kappa,
                              'seeds': seed_accs}
        else:
            for s in range(1, CFG['n_subjects'] + 1):
                acc, kappa = run_loso_fold(s, cache, seed=42)
                results[s] = {'acc': acc, 'std': 0.0,
                              'kappa': kappa, 'kappa_std': 0.0, 'seeds': [acc]}

    else:
        # ── Within-Subject: comportamento identico a v13 ────
        if CFG['multi_seed']:
            for s in range(1, CFG['n_subjects'] + 1):
                mean_acc, std_acc, mean_kappa, std_kappa, seed_accs = \
                    run_subject_multiseed(s)
                results[s] = {'acc': mean_acc, 'std': std_acc,
                              'kappa': mean_kappa, 'kappa_std': std_kappa,
                              'seeds': seed_accs}
        else:
            for s in range(1, CFG['n_subjects'] + 1):
                acc, kappa = run_subject(s, seed=42)
                results[s] = {'acc': acc, 'std': 0.0,
                              'kappa': kappa, 'kappa_std': 0.0, 'seeds': [acc]}

    # ── Tabella finale ──────────────────────────────────────
    accs   = [results[s]['acc']   for s in range(1, 10)]
    kappas = [results[s]['kappa'] for s in range(1, 10)]
    n_s    = len(CFG['seeds']) if CFG['multi_seed'] else 1
    ms_tag = f"Multi-seed ({n_s}×)" if CFG['multi_seed'] else "Single-seed"

    print(f"\n{'='*60}")
    print(f"  {_mode_tag(CFG)} | {ms_tag}")
    print(f"{'='*60}")
    if CFG['multi_seed']:
        print(f"{'Sub':>4} | {'Mean (%)':>9} | {'Std':>6} | {'Kappa':>7}")
        print('-' * 36)
        for s in range(1, 10):
            tag = ' *' if s in HARD_SUBJECTS else '  '
            print(f" S{s:02d}{tag}| {results[s]['acc']:9.2f} | "
                  f"{results[s]['std']:6.2f} | {results[s]['kappa']:7.4f}")
        print('-' * 36)
        print(f" Avg | {np.mean(accs):9.2f} | "
              f"{np.mean([results[s]['std'] for s in range(1,10)]):6.2f} | "
              f"{np.mean(kappas):7.4f}")
        print(f" Std | {np.std(accs):9.2f} |        | {np.std(kappas):7.4f}")
    else:
        print(f"{'Sub':>4} | {'Acc (%)':>8} | {'Kappa':>7}")
        print('-' * 28)
        for s in range(1, 10):
            tag = ' *' if s in HARD_SUBJECTS else '  '
            print(f" S{s:02d}{tag}| {results[s]['acc']:8.2f} | {results[s]['kappa']:7.4f}")
        print('-' * 28)
        print(f" Avg | {np.mean(accs):8.2f} | {np.mean(kappas):7.4f}")
        print(f" Std | {np.std(accs):8.2f} | {np.std(kappas):7.4f}")
    print(f"  * = soggetto difficile (CFG override attivo)")
    print(f"\n✅ [{proto_tag}] Accuracy: {np.mean(accs):.2f} ± {np.std(accs):.2f}%")