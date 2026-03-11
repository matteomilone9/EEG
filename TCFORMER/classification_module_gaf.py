# classification_module_gaf.py
"""
ClassificationModule con auxiliary GAF distillation loss.
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from torchmetrics.classification import MulticlassCohenKappa, MulticlassConfusionMatrix

import pytorch_lightning as pl
from utils.lr_scheduler import linear_warmup_cosine_decay
from gaf_utils import compute_gaf
from gaf_autoencoder import GAFAutoencoder


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def select_random_channels(x, keep_ratio=0.9):
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    return x[:, keep_indices, :], keep_indices

def random_channel_mask(x, keep_ratio=0.9):
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    mask = torch.zeros_like(x)
    mask[:, keep_indices, :] = 1
    return x * mask


# --------------------------------------------------------------------------- #
#  GAF encoder utility
# --------------------------------------------------------------------------- #
class GAFLatentExtractor(nn.Module):
    """
    Wrapper frozen attorno a GAFAutoencoder.encoder.
    Prende EEG raw (B, C, T) → media dei latent per canale → (B, latent_dim).
    """
    def __init__(self, ae_ckpt_path: str, gaf_size: int = 128, gaf_mode: str = "both"):
        super().__init__()
        self.gaf_size = gaf_size
        self.gaf_mode = gaf_mode

        ae = GAFAutoencoder.load_from_checkpoint(ae_ckpt_path)
        self.encoder = ae.encoder
        self.latent_dim = ae.encoder.latent_dim

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        B, C, T = x_raw.shape
        device = x_raw.device

        x_np  = x_raw.cpu().numpy()
        gaf_np = compute_gaf(x_np, gaf_size=self.gaf_size, mode=self.gaf_mode)

        if self.gaf_mode == "both":
            gaf_t = torch.from_numpy(gaf_np).to(device)
            gaf_t = gaf_t.view(B * C, 2, self.gaf_size, self.gaf_size)
        else:
            gaf_t = torch.from_numpy(gaf_np).to(device)
            gaf_t = gaf_t.view(B * C, 1, self.gaf_size, self.gaf_size)

        z = self.encoder(gaf_t.float())
        return z.view(B, C, -1).mean(dim=1)   # (B, latent_dim)


# --------------------------------------------------------------------------- #
#  Projector
# --------------------------------------------------------------------------- #
class AuxProjector(nn.Module):
    def __init__(self, d_tcf: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_tcf, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  ClassificationModuleGAF
# --------------------------------------------------------------------------- #
class ClassificationModuleGAF(pl.LightningModule):
    """
    Drop-in replacement di ClassificationModule con auxiliary GAF distillation.

    L_total = L_cls  +  lambda_aux * L_aux
    L_aux   = 1 - cosine_similarity(projector(tcf_features), z_gaf).mean()

    A inference time: solo TCFormer + projector, nessun overhead GAF.
    """
    def __init__(
        self,
        model,
        n_classes: int,
        ae_ckpt_path: str,
        d_tcf: int,
        gaf_size: int = 128,
        gaf_mode: str = "both",
        lambda_aux: float = 0.1,
        lambda_aux_final: float = 0.0,
        use_lambda_schedule: bool = True,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: bool = False,
        max_epochs: int = 1000,
        warmup_epochs: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.gaf_extractor = GAFLatentExtractor(ae_ckpt_path, gaf_size, gaf_mode)
        latent_dim = self.gaf_extractor.latent_dim
        self.aux_projector = AuxProjector(d_tcf, latent_dim)

        self.test_kappa   = MulticlassCohenKappa(num_classes=n_classes)
        self.test_cm      = MulticlassConfusionMatrix(num_classes=n_classes)
        self.test_confmat = None

    # ------------------------------------------------------------------ #
    def forward(self, x):
        return self.model(x)

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Usa get_features() esposto da TCFormerModule.
        Returns: (B, d_tcf)  — media sul tempo
        """
        features = self.model.get_features(x)   # (B, d_tcf, T')
        return features.mean(dim=-1)             # (B, d_tcf)

    def _lambda_aux(self) -> float:
        if not self.hparams.use_lambda_schedule:
            return self.hparams.lambda_aux
        progress = self.current_epoch / max(self.hparams.max_epochs - 1, 1)
        return self.hparams.lambda_aux + progress * (
            self.hparams.lambda_aux_final - self.hparams.lambda_aux
        )

    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        params = list(self.model.parameters()) + list(self.aux_projector.parameters())
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr,
                                         betas=betas,
                                         weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr,
                                          betas=betas,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            sched = LambdaLR(
                optimizer,
                linear_warmup_cosine_decay(self.hparams.warmup_epochs,
                                           self.hparams.max_epochs)
            )
            return [optimizer], [sched]
        return [optimizer]

    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="val")
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="test")
        return {"test_loss": loss, "test_acc": acc}

    # ------------------------------------------------------------------ #
    def shared_step(self, batch, batch_idx, mode: str = "train"):
        x, y = batch

        if mode == "train":
            if self.hparams.get("random_channel_masking", False):
                x = random_channel_mask(x, self.hparams.get("keep_ratio", 0.9))
            if self.hparams.get("random_channel_selection", False):
                x, _ = select_random_channels(x, self.hparams.get("keep_ratio", 0.9))

        y_hat = self.forward(x)
        l_cls = F.cross_entropy(y_hat, y)

        l_aux = torch.tensor(0.0, device=x.device)
        if mode == "train":
            tcf_feat = self._get_features(x)                 # (B, d_tcf)
            z_proj   = self.aux_projector(tcf_feat)          # (B, latent_dim)
            z_gaf    = self.gaf_extractor(x)                 # (B, latent_dim)
            l_aux    = 1.0 - F.cosine_similarity(z_proj, z_gaf, dim=-1).mean()

        lam  = self._lambda_aux() if mode == "train" else 0.0
        loss = l_cls + lam * l_aux

        acc = accuracy(y_hat, y, task="multiclass",
                       num_classes=self.hparams.n_classes)
        self.log(f"{mode}_loss", loss, prog_bar=True,  on_step=False, on_epoch=True)
        self.log(f"{mode}_acc",  acc,  prog_bar=True,  on_step=False, on_epoch=True)
        if mode == "train":
            self.log("aux_loss",   l_aux, prog_bar=False, on_step=False, on_epoch=True)
            self.log("lambda_aux", lam,   prog_bar=False, on_step=False, on_epoch=True)

        if mode == "test":
            preds = torch.argmax(y_hat, dim=-1)
            self.test_kappa.update(preds, y)
            self.test_cm.update(preds, y)
            self.log("test_kappa", self.test_kappa,
                     prog_bar=False, on_step=False, on_epoch=True)

        return loss, acc

    def on_test_epoch_end(self):
        cm_counts = self.test_cm.compute()
        self.test_cm.reset()
        with torch.no_grad():
            row_sums = cm_counts.sum(dim=1, keepdim=True).clamp_min(1)
            cm_percent = cm_counts.float() / row_sums * 100.0
        self.test_confmat = cm_percent.cpu()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(self.forward(x), dim=-1)
