# gaf_autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------------ #
#  Encoder
# ------------------------------------------------------------------ #
class GAFEncoder(nn.Module):
    """
    Input:  (B, in_channels, 128, 128)   in_channels = 2 if mode=="both", else 1
    Output: (B, latent_dim)
    """
    def __init__(self, in_channels: int = 2, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        def _block(ci, co, stride=2):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ELU(inplace=True),
            )

        # 128 → 64 → 32 → 16 → 8 → 4
        self.enc = nn.Sequential(
            _block(in_channels, 32),   # → (B, 32,  64, 64)
            _block(32, 64),            # → (B, 64,  32, 32)
            _block(64, 128),           # → (B, 128, 16, 16)
            _block(128, 256),          # → (B, 256,  8,  8)
            _block(256, 256),          # → (B, 256,  4,  4)
        )
        self.flatten = nn.Flatten()                     # → (B, 256*4*4 = 4096)
        self.proj    = nn.Linear(4096, latent_dim)      # → (B, latent_dim)
        self.bn_lat  = nn.BatchNorm1d(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn_lat(self.proj(self.flatten(self.enc(x))))


# ------------------------------------------------------------------ #
#  Decoder
# ------------------------------------------------------------------ #
class GAFDecoder(nn.Module):
    """
    Input:  (B, latent_dim)
    Output: (B, in_channels, 128, 128)
    """
    def __init__(self, in_channels: int = 2, latent_dim: int = 128):
        super().__init__()

        self.unproj = nn.Linear(latent_dim, 4096)   # → (B, 4096)

        def _block(ci, co):
            return nn.Sequential(
                nn.ConvTranspose2d(ci, co, 3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ELU(inplace=True),
            )

        # 4 → 8 → 16 → 32 → 64 → 128
        self.dec = nn.Sequential(
            _block(256, 256),          # → (B, 256, 8,  8)
            _block(256, 128),          # → (B, 128, 16, 16)
            _block(128, 64),           # → (B, 64,  32, 32)
            _block(64, 32),            # → (B, 32,  64, 64)
            # last layer: no BN, tanh output (GAF is in [-1,1])
            nn.ConvTranspose2d(32, in_channels, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.Tanh(),                 # → (B, in_channels, 128, 128)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.unproj(z).view(-1, 256, 4, 4)
        return self.dec(x)


# ------------------------------------------------------------------ #
#  Full Autoencoder (Lightning Module)
# ------------------------------------------------------------------ #
class GAFAutoencoder(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 2,      # 2 if mode="both" (GASF+GADF), 1 otherwise
        latent_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = GAFEncoder(in_channels, latent_dim)
        self.decoder = GAFDecoder(in_channels, latent_dim)

    # ---- forward ----
    def encode(self, x):  return self.encoder(x)
    def decode(self, z):  return self.decoder(z)
    def forward(self, x): return self.decode(self.encode(x))

    # ---- loss ----
    def _step(self, batch, stage):
        x = batch[0]          # (B, in_ch, H, W)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
        return [opt], [sched]
