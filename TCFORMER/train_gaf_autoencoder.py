# train_gaf_autoencoder.py
"""
Pre-trains a GAF autoencoder on BCI-IV 2a data.
Run once before the main TCFormer training:

    python train_gaf_autoencoder.py --dataset bcic2a --gaf_size 128 --latent_dim 128
"""
import argparse
import gc
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
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


# ------------------------------------------------------------------ #
#  Lazy Dataset: legge dal .npy memory-mapped, non carica tutto in RAM
# ------------------------------------------------------------------ #
class GAFDataset(Dataset):
    def __init__(self, gaf_path: Path):
        self.data = np.load(gaf_path, mmap_mode='r')  # (N, C, 2, H, W)
        self.N, self.C = self.data.shape[:2]

    def __len__(self):
        return self.N * self.C

    def __getitem__(self, idx):
        trial_idx = idx // self.C
        chan_idx   = idx  % self.C
        x = torch.from_numpy(self.data[trial_idx, chan_idx].copy())
        return (x,)


def _extract_X(run):
    return np.stack([run[i][0] for i in range(len(run))])


def collect_all_train_data(datamodule_cls, preprocessing_dict):
    """
    Collect ALL trials from ALL subjects (both sessions) to train the AE.
    The AE does not need labels — we use everything available.
    """
    all_X = []
    for subj_id in datamodule_cls.all_subject_ids:
        dm = datamodule_cls(preprocessing_dict, subject_id=subj_id)
        dm.prepare_data()
        dm.setup()
        X_train = dm.train_dataset.tensors[0].numpy()
        X_val   = dm.val_dataset.tensors[0].numpy()
        X_test  = dm.test_dataset.tensors[0].numpy()
        all_X.extend([X_train, X_val, X_test])
        print(f"  Subject {subj_id}: train={X_train.shape[0]}, "
              f"val={X_val.shape[0]}, test={X_test.shape[0]}")
    return np.concatenate(all_X, axis=0)   # (N_total, 22, T)


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
    precompute_and_cache_gaf(
        X_all, gaf_cache,
        gaf_size=args.gaf_size,
        mode=args.mode,
        force_recompute=args.force_recompute,
    )
    # libera X_all dalla RAM — non serve più, le GAF sono su disco
    del X_all
    gc.collect()

    # --- build dataset lazy ---
    print("\n[3/4] Building per-channel dataset (lazy)...")
    full_ds = GAFDataset(gaf_cache)
    print(f"  GAF shape on disk: {full_ds.data.shape}")
    n_val   = int(len(full_ds) * 0.1)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    print(f"  Train samples: {n_train}  |  Val samples: {n_val}")

    # train_dl = DataLoader(train_ds, batch_size=args.batch_size,
    #                       shuffle=True,  num_workers=0, pin_memory=True,
    #                       persistent_workers=True)
    # val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
    #                       shuffle=False, num_workers=0, pin_memory=True,
    #                       persistent_workers=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=False)

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
        num_sanity_val_steps=0,
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
    parser.add_argument("--dataset",         type=str,   default="bcic2a")
    parser.add_argument("--gaf_size",        type=int,   default=128)
    parser.add_argument("--latent_dim",      type=int,   default=128)
    parser.add_argument("--mode",            type=str,   default="both",
                        choices=["gasf", "gadf", "both"])
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    main(args)
