# src/__init__.py
from ._preprocessing import (
    load_eeg_data,
    filter_eeg,
    create_labels,
)

from ._dataset import EEGDataset
from ._models import EEGAttentionNet, SimpleEEGNet
from ._train import train_epoch, validate
from ._evaluate import evaluate_model
from ._utils import save_checkpoint, compute_class_weights