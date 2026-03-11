# train_gaf_autoencoder.py
"""
Pre-trains a GAF autoencoder on BCI-IV 2a data.
Run once before the main TCFormer training:

    python train_gaf_autoencoder.py --dataset bcic2a --gaf_size 128 --latent_dim 128
"""
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from utils.get_datamodule_cls import get_datamodule_cls
from utils.seed import seed_everything
from gaf_utils import precompute_and_cache_gaf
from gaf_autoencoder import GAFAutoencoder

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
GAF_CACHE_DIR = Path(__file__).resolve().parent / "gaf_cache"
AE_CKPT_DIR   = Path(__file__).resolve().parent / "gaf_checkpoints"


def _extract_X(run):
    import numpy as np
    return np.stack([run[i][0] for i in range(len(run))])


def collect_all_train_data(datamodule_cls, preprocessing_dict, mode="loso"):
    """
    Collect ALL trials from ALL subjects (both sessions) to train the AE.
    The AE does not need labels — we use everything available.
    """
    all_X = []
    dm_class = datamodule_cls

    # We load all subjects at once via subject_id loop
    for subj_id in dm_class.all_subject_ids:
        dm = dm_class(preprocessing_dict, subject_id=subj_id)
        dm.prepare_data()
        dm.setup()
        # grab numpy arrays from the TensorDatasets
        X_train = dm.train_dataset.tensors[0].numpy()  # (N, 22, T)
        X_val   = dm.val_dataset.tensors[0].numpy()
        X_test  = dm.test_dataset.tensors[0].numpy()
        all_X.extend([X_train, X_val, X_test])
        print(f"  Subject {subj_id}: train={X_train.shape[0]}, "
              f"val={X_val.shape[0]}, test={X_test.shape[0]}")

    return np.concatenate(all_X, axis=0)   # (N_total, 22, T)


def build_per_channel_dataset(
    X_gaf: np.ndarray,   # (N, C, 2, H, W)  or  (N, C, H, W)
) -> TensorDataset:
    """
    Flatten the channel dimension into the batch dimension.
    Each item is one (channel, trial) GAF image.
    """
    N, C = X_gaf.shape[:2]
    rest = X_gaf.shape[2:]           # (2, H, W) or (H, W)
    X_flat = X_gaf.reshape(N * C, *rest)   # (N*C, 2, H, W)
    t = torch.from_numpy(X_flat)
    return TensorDataset(t)


def main(args):
    seed_everything(42)

    # --- load config ---
    config_path = CONFIG_DIR / "tcformer.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    preprocessing_dict = config["preprocessing"][args.dataset]
    preprocessing_dict["z_scale"] = config["z_scale"]

    # --- datamodule class ---
    datamodule_cls = get_datamodule_cls(f"{args.dataset}_loso")

    # --- collect data ---
    print("\n[1/4] Collecting EEG data from all subjects...")
    cache_X = GAF_CACHE_DIR / f"{args.dataset}_X_all.npy"
    if cache_X.exists() and not args.force_recompute:
        print(f"  Loading cached X from {cache_X}")
        X_all = np.load(cache_X)
    else:
        X_all = collect_all_train_data(datamodule_cls, preprocessing_dict)
        GAF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_X, X_all)
    print(f"  Total trials: {X_all.shape}  →  (N, C=22, T={X_all.shape[-1]})")

    # --- compute GAF ---
    print("\n[2/4] Computing GAF images...")
    gaf_cache = GAF_CACHE_DIR / f"{args.dataset}_gaf_size{args.gaf_size}_{args.mode}.npy"
    X_gaf = precompute_and_cache_gaf(
        X_all, gaf_cache,
        gaf_size=args.gaf_size,
        mode=args.mode,
        force_recompute=args.force_recompute,
    )
    # X_gaf: (N, 22, 2, 128, 128)  if mode=="both"
    print(f"  GAF shape: {X_gaf.shape}")

    # --- build dataset: flatten (N, C) → N*C samples ---
    print("\n[3/4] Building per-channel dataset...")
    full_ds = build_per_channel_dataset(X_gaf)
    n_val = int(len(full_ds) * 0.1)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    num_workers = 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"  Train samples: {n_train}  |  Val samples: {n_val}")

    # --- train autoencoder ---
    print("\n[4/4] Training GAF Autoencoder...")
    in_channels = 2 if args.mode == "both" else 1
    ae = GAFAutoencoder(
        in_channels=in_channels,
        latent_dim=args.latent_dim,
        lr=args.lr,
    )

    AE_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"gaf_ae_{args.dataset}_lat{args.latent_dim}_sz{args.gaf_size}"

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(
                dirpath=AE_CKPT_DIR,
                filename=ckpt_name + "_{epoch:03d}_{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
            ),
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
        ],
    )
    trainer.fit(ae, train_dl, val_dl)
    print(f"\n  Best checkpoint saved in: {AE_CKPT_DIR}")
    print(f"  Best val_loss: {trainer.callback_metrics.get('val_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str,   default="bcic2a")
    parser.add_argument("--gaf_size",       type=int,   default=128)
    parser.add_argument("--latent_dim",     type=int,   default=128)
    parser.add_argument("--mode",           type=str,   default="both",
                        choices=["gasf", "gadf", "both"])
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--force_recompute",action="store_true")
    args = parser.parse_args()
    main(args)
