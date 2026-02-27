# main_spectrogram.py

import torch
import numpy as np
import os
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report

from src import load_eeg_data, filter_eeg, create_labels
from src import compute_class_weights
from src_spect import EEGSpectrogramDataset, SpectrogramNet


# ─────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath,
                    best_loss=None, patience_counter=None):
    ckpt = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss":                 loss,
    }
    if best_loss       is not None: ckpt["best_loss"]        = best_loss
    if patience_counter is not None: ckpt["patience_counter"] = patience_counter
    torch.save(ckpt, filepath)
    print(f"💾 Checkpoint salvato: {filepath}")


def evaluate(model, loader, device):
    """Valutazione finale con classification report."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()

    print("\n📊 Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Riposo", "Immaginazione"],
        digits=3, zero_division=0,
    ))
    return acc


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(dim=1) == y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out  = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct    += (out.argmax(dim=1) == y).sum().item()
            total      += y.size(0)
    return total_loss / len(loader), correct / total


# ─────────────────────────────────────────────
def main():

    # ── 1. Config ────────────────────────────
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    window_size = config["training_spectrogram"]["window_size"]   # 500
    stride      = config["training_spectrogram"].get("stride", 250)
    fs          = config["data"]["fs"]                # 500

    # ── 2. Preprocessing ─────────────────────
    print("\n" + "="*55)
    print("1. Preprocessing...")
    print("="*55)

    df = load_eeg_data(config["data"]["edf_path"])
    df = filter_eeg(
        df,
        lowcut=config["preprocessing"]["filter_band"][0],
        highcut=config["preprocessing"]["filter_band"][1],
        fs=fs,
    )

    labels = create_labels(config["data"]["events_path"], fs=fs)

    # pad/truncate labels a T
    T = df.shape[1]
    if len(labels) < T:
        labels = np.pad(labels, (0, T - len(labels)), constant_values=0)
    else:
        labels = labels[:T]

    print(f"EEG: {df.shape} | labels: {len(labels)} | dist: {np.bincount(labels)}")

    # ── 3. Split ─────────────────────────────
    train_range = tuple(config["split"]["train"])
    val_range   = tuple(config["split"]["val"])
    test_range  = tuple(config["split"]["test"])

    print("\n" + "="*55)
    print("2. Dataset spettrogrammi...")
    print("="*55)

    train_dataset = EEGSpectrogramDataset(
        df, labels, train_range, window_size, stride=stride, fs=fs, augment=True
    )
    val_dataset = EEGSpectrogramDataset(
        df, labels, val_range, window_size, stride=stride, fs=fs, augment=False
    )
    test_dataset = EEGSpectrogramDataset(
        df, labels, test_range, window_size, stride=stride, fs=fs, augment=False
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(train_dataset, batch_size=config["training_spectrogram"]["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=config["training_spectrogram"]["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=config["training_spectrogram"]["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=pin_memory)

    # ── 4. Modello ───────────────────────────
    print("\n" + "="*55)
    print("3. Modello SpectrogramNet...")
    print("="*55)

    model = SpectrogramNet(
        n_channels=config["model"]["n_channels"],
        n_classes=config["model"]["n_classes"],
        dropout=0.25,
    ).to(device)

    # ── 5. Ottimizzatore, loss ────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training_spectrogram"]["lr"],
        weight_decay=config["training_spectrogram"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_labels_subset = labels[train_range[0]:train_range[1]]
    class_weights = compute_class_weights(train_labels_subset).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"LR: {config['training_spectrogram']['lr']} | WD: {config['training_spectrogram']['weight_decay']}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # ── 6. Checkpoint dirs ───────────────────
    ckpt_dir       = "./checkpoints_spectrogram"
    best_path      = f"{ckpt_dir}/best_model.pth"
    last_path      = f"{ckpt_dir}/last_checkpoint.pth"
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch      = 0
    best_val_loss    = float("inf")
    patience_counter = 0

    # ── 7. Training ──────────────────────────
    print("\n" + "="*55)
    print("INIZIO TRAINING — SpectrogramNet")
    print("="*55)

    for epoch in range(start_epoch, config["training_spectrogram"]["epochs"]):
        print(f"\n📌 Epoch {epoch+1}/{config['training_spectrogram']['epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"  Val   Loss: {val_loss:.4f}   | Acc: {val_acc:.2%} | LR: {current_lr:.2e}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path)
            print(f"  ✅ Best model! (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{config['training_spectrogram']['patience']}")

        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, last_path,
                        best_loss=best_val_loss, patience_counter=patience_counter)

        if patience_counter >= config["training_spectrogram"]["patience"]:
            print(f"\n🛑 Early stopping dopo {config['training_spectrogram']['patience']} epoche.")
            break

    # ── 8. Test finale ───────────────────────
    print("\n" + "="*55)
    print("VALUTAZIONE FINALE SUL TEST SET")
    print("="*55)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Caricato best model (epoca {ckpt['epoch']+1}, loss={ckpt['loss']:.4f})")

    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2%}")

    os.makedirs("./results", exist_ok=True)
    with open("./results/results_spectrogram.txt", "w") as f:
        f.write(f"Model:         SpectrogramNet\n")
        f.write(f"Test Accuracy: {test_acc:.2%}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Best Epoch:    {ckpt['epoch']+1}\n")

    print("Risultati salvati in ./results/results_spectrogram.txt")


if __name__ == "__main__":
    main()
