from typing import Dict, Optional

import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import os

from utils.interaug import interaug


# Must be defined at module level (not inside a function) to be picklable
# on Windows with multiprocessing DataLoader workers.
class CollateFn:
    """Picklable collate function that optionally applies interaug."""
    def __init__(self, preproc: dict):
        self.use_interaug = preproc.get("interaug", False)

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs)                        # [B, C, T]
        y = torch.tensor(ys, dtype=torch.long)
        if self.use_interaug:
            x, y = interaug([x, y])
        return x, y


class BaseDataModule(pl.LightningDataModule):
    dataset = None
    train_dataset = None
    test_dataset = None

    def __init__(self, preprocessing_dict: Dict, subject_id: int):
        super(BaseDataModule, self).__init__()
        self.preprocessing_dict = preprocessing_dict
        self.subject_id = subject_id

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        num_workers = self.preprocessing_dict.get("num_workers", os.cpu_count() // 2)
        return DataLoader(self.train_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=num_workers > 0,  # only if workers > 0
                          prefetch_factor=4 if num_workers > 0 else None,
                          collate_fn=CollateFn(self.preprocessing_dict)  # picklable
                          )

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        num_workers = self.preprocessing_dict.get("num_workers", os.cpu_count() // 2)
        return DataLoader(self.test_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=num_workers > 0,  # only if workers > 0
                          prefetch_factor=4 if num_workers > 0 else None,
                          )

    @staticmethod
    def _z_scale(X, X_test):
        s, c, t = X.shape
        X_2d      = X.transpose(1, 0, 2).reshape(c, -1).T
        X_test_2d = X_test.transpose(1, 0, 2).reshape(c, -1).T

        sc = StandardScaler().fit(X_2d)
        X      = sc.transform(X_2d).T.reshape(c, s, t).transpose(1, 0, 2)
        X_test = sc.transform(X_test_2d).T.reshape(c, X_test.shape[0], t).transpose(1, 0, 2)
        return X, X_test

    def _z_scale_tvt(X, X_val, X_test):
        s, c, t = X.shape
        X_2d      = X.transpose(1, 0, 2).reshape(c, -1).T
        X_val_2d  = X_val.transpose(1, 0, 2).reshape(c, -1).T
        X_test_2d = X_test.transpose(1, 0, 2).reshape(c, -1).T

        sc = StandardScaler().fit(X_2d)
        X      = sc.transform(X_2d).T.reshape(c, s, t).transpose(1, 0, 2)
        X_val  = sc.transform(X_val_2d).T.reshape(c, X_val.shape[0], t).transpose(1, 0, 2)
        X_test = sc.transform(X_test_2d).T.reshape(c, X_test.shape[0], t).transpose(1, 0, 2)
        return X, X_val, X_test

    @staticmethod
    def _make_tensor_dataset(X, y):
        return TensorDataset(torch.Tensor(X), torch.Tensor(y).type(torch.LongTensor))