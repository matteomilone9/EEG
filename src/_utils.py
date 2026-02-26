"""
Funzioni di supporto
"""
import yaml
import os
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, best_loss=None, patience_counter=None):
    """
    ✅ CORRETTA - Salva TUTTI gli stati
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # ✅ AGGIUNTO
        'loss': loss,
    }
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss
    if patience_counter is not None:
        checkpoint['patience_counter'] = patience_counter

    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint salvato: {filepath}")


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)