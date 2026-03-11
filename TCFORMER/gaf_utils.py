# gaf_utils.py
import numpy as np
import torch
from pathlib import Path
from scipy.signal import resample


# ------------------------------------------------------------------ #
#  GAF computation (no external pyts dependency)
# ------------------------------------------------------------------ #

def _to_gasf(x_norm: np.ndarray) -> np.ndarray:
    """
    x_norm: (T,) array in [-1, 1]
    returns: (T, T) GASF matrix
    """
    phi = np.arccos(np.clip(x_norm, -1, 1))  # (T,)
    return np.cos(phi[:, None] + phi[None, :])  # (T, T)


def _to_gadf(x_norm: np.ndarray) -> np.ndarray:
    """
    x_norm: (T,) array in [-1, 1]
    returns: (T, T) GADF matrix
    """
    phi = np.arccos(np.clip(x_norm, -1, 1))
    return np.sin(phi[:, None] - phi[None, :])  # (T, T)


def _normalize_channel(x: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D signal to [-1, 1]."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return 2 * (x - mn) / (mx - mn) - 1


def compute_gaf(
        X: np.ndarray,  # (N, C, T)
        gaf_size: int = 128,
        mode: str = "both",  # "gasf" | "gadf" | "both"
) -> np.ndarray:
    """
    Convert EEG trials to GAF images per channel.

    Returns:
        np.ndarray of shape:
          (N, C, gaf_size, gaf_size)       if mode in {"gasf","gadf"}
          (N, C, 2, gaf_size, gaf_size)    if mode == "both"
    """
    N, C, T = X.shape
    n_img = 2 if mode == "both" else 1

    out_shape = (N, C, n_img, gaf_size, gaf_size) if mode == "both" \
        else (N, C, gaf_size, gaf_size)
    out = np.zeros(out_shape, dtype=np.float32)

    for i in range(N):
        for c in range(C):
            sig = X[i, c]  # (T,)
            # downsample to gaf_size
            sig_ds = resample(sig, gaf_size)  # (gaf_size,)
            sig_norm = _normalize_channel(sig_ds)  # [-1, 1]

            if mode == "gasf":
                out[i, c] = _to_gasf(sig_norm)
            elif mode == "gadf":
                out[i, c] = _to_gadf(sig_norm)
            else:  # both
                out[i, c, 0] = _to_gasf(sig_norm)
                out[i, c, 1] = _to_gadf(sig_norm)
    return out


def precompute_and_cache_gaf(
        X: np.ndarray,
        cache_path: Path,
        gaf_size: int = 128,
        mode: str = "both",
        force_recompute: bool = False,
) -> np.ndarray:
    """
    Compute GAF images and cache to disk as .npy file.
    Loads from cache if it already exists.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        print(f"  [GAF] Loading cache: {cache_path}")
        return np.load(cache_path)

    print(f"  [GAF] Computing {X.shape[0]} trials → {cache_path.name} ...")
    gaf = compute_gaf(X, gaf_size=gaf_size, mode=mode)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, gaf)
    print(f"  [GAF] Saved to {cache_path}  shape={gaf.shape}")
    return gaf
