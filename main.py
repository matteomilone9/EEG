# main.py

import torch
import numpy as np
import os
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src import (
    load_eeg_data,
    filter_eeg,
    create_labels,
    EEGDataset,
    EEGAttentionNet,
    SimpleEEGNet,
    train_epoch,
    validate,
    evaluate_model,
    compute_class_weights,
)


# ─────────────────────────────────────────────
# UTILITY: salva checkpoint
# ─────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath,
                    best_loss=None, patience_counter=None):
    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss":                 loss,
    }
    if best_loss is not None:
        checkpoint["best_loss"] = best_loss
    if patience_counter is not None:
        checkpoint["patience_counter"] = patience_counter
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint salvato: {filepath}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    # ── 1. Config ────────────────────────────
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Modello: {config['model']['name']}")

    window_size = config["training"]["window_size"]

    # ── 2. Preprocessing ─────────────────────
    print("\n" + "="*50)
    print("1. Caricamento e preprocessing dati...")
    print("="*50)

    df = load_eeg_data(config["data"]["edf_path"])
    print(f"Filtraggio {config['preprocessing']['filter_band']} Hz...")
    df = filter_eeg(
        df,
        lowcut=config["preprocessing"]["filter_band"][0],
        highcut=config["preprocessing"]["filter_band"][1],
    )

    print(f"Shape df: {df.shape}")
    print(f"Range: [{df.values.min():.4f}, {df.values.max():.4f}] V")
    print(f"Media: {df.values.mean():.4f} V | Std: {df.values.std():.4f} V")

    # ── 3. Etichette ─────────────────────────
    print("\n" + "="*50)
    print("2. Creazione etichette...")
    print("="*50)

    labels = create_labels(config["data"]["events_path"])
    print(f"Distribuzione etichette: {np.bincount(labels)}")
    print(f"Totale campioni etichettati: {len(labels)}")

    # ── 4. Split ─────────────────────────────
    # FIX: range (start, end) basati sui confini dei trial,
    #      non più liste di indici per evitare data leakage
    train_range = tuple(config["split"]["train"])
    val_range   = tuple(config["split"]["val"])
    test_range  = tuple(config["split"]["test"])

    # FIX: stride = window_size → finestre NON sovrapposte (nessun leakage)
    stride = config["training"].get("stride", window_size)

    print("\n" + "="*50)
    print("3. Creazione dataset e DataLoader...")
    print("="*50)
    print(f"Window size: {window_size} campioni ({window_size / config['data']['fs']:.1f}s)")
    print(f"Stride:      {stride} campioni ({stride / config['data']['fs']:.1f}s)")

    train_dataset = EEGDataset(df, labels, train_range, window_size, stride=stride)
    val_dataset   = EEGDataset(df, labels, val_range,   window_size, stride=stride)
    test_dataset  = EEGDataset(df, labels, test_range,  window_size, stride=stride)

    print(f"Train: {len(train_dataset)} finestre")
    print(f"Val:   {len(val_dataset)} finestre")
    print(f"Test:  {len(test_dataset)} finestre")

    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    # ── 5. Modello ───────────────────────────
    print("\n" + "="*50)
    print(f"4. Creazione modello: {config['model']['name']}...")
    print("="*50)

    if config["model"]["name"] == "EEGAttentionNet":
        model = EEGAttentionNet(
            n_channels=config["model"]["n_channels"],
            n_samples=window_size,
            n_classes=config["model"]["n_classes"],
        ).to(device)
    else:
        model = SimpleEEGNet(
            n_channels=config["model"]["n_channels"],
            n_samples=window_size,
            n_classes=config["model"]["n_classes"],
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {total_params:,}")

    # ── 6. Ottimizzatore, scheduler, loss ────
    print("\n" + "="*50)
    print("5. Ottimizzatore e loss...")
    print("="*50)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=5, min_lr=1e-6
    )

    # Class weights calcolati SOLO sul training set
    train_labels_subset = labels[train_range[0]:train_range[1]]
    class_weights = compute_class_weights(train_labels_subset).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"Learning rate:  {config['training']['lr']}")
    print(f"Weight decay:   {config['training']['weight_decay']}")
    print(f"Class weights:  {class_weights.cpu().numpy()}")
    print(f"Early stopping patience: {config['training']['patience']}")

    # ── 7. Resume da checkpoint ──────────────
    resume_path    = "checkpoints/last_checkpoint.pth"
    best_model_path = "checkpoints/best_model.pth"
    resume = config["training"].get("resume", True)

    start_epoch     = 0
    best_val_loss   = float("inf")
    patience_counter = 0

    os.makedirs("checkpoints", exist_ok=True)

    if resume and os.path.exists(resume_path):
        print(f"\nCaricamento LAST checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch      = checkpoint["epoch"] + 1
            best_val_loss    = checkpoint.get("best_loss", checkpoint["loss"])
            patience_counter = checkpoint.get("patience_counter", 0)
            print(f"Ripresa da epoca {start_epoch} | best_val_loss={best_val_loss:.4f} | patience={patience_counter}")
        except Exception as e:
            print(f"Errore caricamento checkpoint: {e} — inizio da zero.")

    elif os.path.exists(best_model_path):
        print(f"\nCaricamento BEST modello: {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"BEST caricato | loss={checkpoint['loss']:.4f}")
        except Exception as e:
            print(f"Errore caricamento best model: {e}")

    # ── 8. Training loop ─────────────────────
    print("\n" + "="*60)
    print("INIZIO TRAINING")
    print("="*60)

    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"\n📌 Epoch {epoch + 1}/{config['training']['epochs']}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f" Train Loss: {train_loss:.4f}")
        print("\n")
        print(f" Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.3f} | LR: {current_lr:.2e}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_model_path)
            print(f" ✅ Miglior modello! (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f" ⏳ Patience: {patience_counter}/{config['training']['patience']}")

        # Salva sempre il last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss,
            resume_path,
            best_loss=best_val_loss,
            patience_counter=patience_counter,
        )

        # Early stopping
        if patience_counter >= config["training"]["patience"]:
            print(f"\n🛑 Early stopping dopo {config['training']['patience']} epoche senza miglioramenti.")
            break

    # ── 9. Valutazione finale sul test set ───
    print("\n" + "="*60)
    print("VALUTAZIONE FINALE SUL TEST SET")
    print("="*60)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Caricato miglior modello (epoca {checkpoint['epoch'] + 1}, loss={checkpoint['loss']:.4f})")

    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2%}")

    # Salva risultati
    final_epoch = checkpoint["epoch"] + 1
    with open("./results.txt", "w") as f:
        f.write(f"Test Accuracy:  {test_acc:.2%}\n")
        f.write(f"Best Val Loss:  {best_val_loss:.4f}\n")
        f.write(f"Final Epoch:    {final_epoch}\n")

    print("Training completato! Risultati in results.txt")
    print(f"Checkpoints: {best_model_path} | {resume_path}")


if __name__ == "__main__":
    main()
