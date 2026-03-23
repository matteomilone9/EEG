# bcic4_2a_ea.py
# Copia di bcic4_2a.py con Euclidean Alignment integrato in BCICIV2aLOSO_EA

from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4
from sklearn.model_selection import train_test_split
import os


def _extract_X(run):
    return np.stack([run[i][0] for i in range(len(run))])


# ------------------------------------------------------------------ #
# Euclidean Alignment
# ------------------------------------------------------------------ #
def euclidean_alignment(X_train, X_val, X_test):
    """
    Euclidean Alignment per-dataset.
    Calcola R_mean sui trial di train, applica R^{-1/2} a train/val/test.
    X: (N, C, T) → output: (N, C, T) stesso shape
    """
    N, C, T = X_train.shape
    R = np.zeros((C, C), dtype=np.float64)
    for i in range(N):
        xi = X_train[i].astype(np.float64)  # (C, T)
        R += xi @ xi.T / T
    R /= N

    # R^{-1/2} tramite eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-10)  # stabilità numerica
    R_inv_sqrt = (eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T).astype(np.float64)

    def _apply(X):
        return np.einsum("ij,njt->nit", R_inv_sqrt, X).astype(np.float32)

    return _apply(X_train), _apply(X_val), _apply(X_test)


# ------------------------------------------------------------------ #
# DataModules originali (invariati)
# ------------------------------------------------------------------ #
class BCICIV2a(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        splitted_ds = self.dataset.split("session")
        train_dataset, test_dataset = splitted_ds["0train"], splitted_ds["1test"]

        X = np.concatenate(
            [_extract_X(run) for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for run in train_dataset.datasets], axis=0)
        X_test = np.concatenate(
            [_extract_X(run) for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class BCICIV2aTVT(BaseDataModule):
    val_dataset = None
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        splitted_ds = self.dataset.split("session")
        session1 = splitted_ds["0train"]
        session2 = splitted_ds["1test"]

        X = np.concatenate([_extract_X(run) for run in session1.datasets], axis=0)
        y = np.concatenate([run.y for run in session1.datasets], axis=0)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2,
            random_state=self.preprocessing_dict.get("seed", 42), stratify=y)

        X_test = np.concatenate([_extract_X(run) for run in session2.datasets], axis=0)
        y_test = np.concatenate([run.y for run in session2.datasets], axis=0)

        if self.preprocessing_dict["z_scale"]:
            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True)


# ------------------------------------------------------------------ #
# BCICIV2aLOSO_EA — LOSO con Euclidean Alignment
# ------------------------------------------------------------------ #
class BCICIV2aLOSO_EA(BCICIV2a):
    """
    LOSO cross-subject con Euclidean Alignment applicato dopo z_scale.
    Attivabile via preprocessing_dict["euclidean_alignment"] = True (default True).
    """
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=self.all_subject_ids, dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]

        train_datasets = [splitted_ds[str(subj_id)].split("session")["0train"]
                          for subj_id in train_subjects]
        val_datasets = [splitted_ds[str(subj_id)].split("session")["1test"]
                        for subj_id in train_subjects]
        test_dataset = splitted_ds[str(self.subject_id)].split("session")["1test"]

        X = np.concatenate([_extract_X(run) for train_dataset in
                            train_datasets for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets
                            for run in train_dataset.datasets], axis=0)
        X_val = np.concatenate([_extract_X(run) for val_dataset in
                                val_datasets for run in val_dataset.datasets], axis=0)
        y_val = np.concatenate([run.y for val_dataset in val_datasets
                                for run in val_dataset.datasets], axis=0)
        X_test = np.concatenate([_extract_X(run) for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        # z_scale
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        # Euclidean Alignment (default attivo)
        if self.preprocessing_dict.get("euclidean_alignment", True):
            X, X_val, X_test = euclidean_alignment(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True)
