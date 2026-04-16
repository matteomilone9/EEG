# preprocessing.py — Caricamento dati, normalizzazione, GAF
# ============================================================

import numpy as np
from tqdm.auto import tqdm

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from pyts.image import GramianAngularField

from config import CFG


def zscore(X: np.ndarray) -> np.ndarray:
    m = X.mean(axis=-1, keepdims=True)
    s = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - m) / s).astype(np.float32)


def minmax(X: np.ndarray) -> np.ndarray:
    lo = X.min(axis=-1, keepdims=True)
    hi = X.max(axis=-1, keepdims=True)
    return ((X - lo) / (hi - lo + 1e-8)).astype(np.float32)


def downsample(X: np.ndarray, n: int) -> np.ndarray:
    from scipy.signal import resample
    return resample(X, n, axis=-1).astype(np.float32) if X.shape[-1] != n else X.astype(np.float32)


def windows_to_numpy(ds):
    Xl, yl = [], []
    for X, y, _ in ds:
        Xl.append(X); yl.append(y)
    return np.stack(Xl).astype(np.float32), np.array(yl, dtype=np.int64)


def load_subject(sub_id: int, cfg: dict):
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


def load_subject_both(sub_id: int, cfg: dict) -> dict:
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
    return {'T': (zscore(X_t), y_t), 'E': (zscore(X_e), y_e)}


def make_gaf(X_raw: np.ndarray, cfg: dict) -> np.ndarray:
    gaf = GramianAngularField(image_size=cfg['image_size'], method=cfg['gaf_method'])
    X_ds = downsample(X_raw, cfg['downsample_to'])
    X_ds = minmax(X_ds)
    B, C, _ = X_ds.shape
    out = np.zeros((B, C, cfg['image_size'], cfg['image_size']), dtype=np.float32)
    for i in tqdm(range(B), desc='GAF', leave=False):
        out[i] = gaf.fit_transform(X_ds[i])
    return out


def preprocess_subject(X_tr_raw: np.ndarray, X_te_raw: np.ndarray, cfg: dict):
    X_tr_t = zscore(X_tr_raw)
    X_te_t = zscore(X_te_raw)
    if cfg['use_gaf']:
        X_tr_g = make_gaf(X_tr_raw, cfg)
        X_te_g = make_gaf(X_te_raw, cfg)
    else:
        X_tr_g = np.zeros((len(X_tr_t), 1, 1, 1), dtype=np.float32)
        X_te_g = np.zeros((len(X_te_t), 1, 1, 1), dtype=np.float32)
    return X_tr_t, X_te_t, X_tr_g, X_te_g


def build_loso_cache(cfg: dict) -> dict:
    cache = {}
    for s in range(1, cfg['n_subjects'] + 1):
        print(f" [LOSO cache] Caricamento soggetto {s}/{cfg['n_subjects']}...")
        cache[s] = load_subject_both(s, cfg)
    print(f" [LOSO cache] ✅ Tutti i {cfg['n_subjects']} soggetti in memoria.\n")
    return cache