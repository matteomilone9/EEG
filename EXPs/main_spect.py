# main_spectrogram.py
import torch
import numpy as np
import os
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report

from src import load_eeg_data, filter_eeg, create_labels, compute_class_weights
from src_spect import EEGSpectrogramDataset, SpectrogramNet


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath,
                    best_loss=None, patience_counter=None):
    ckpt = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss":                 loss,
    }
    if best_loss        is not None: ckpt["best_loss"]        = best_loss
    if patience_counter is not None: ckpt["patience_counter"] = patience_counter
    torch.save(ckpt, filepath)
    print(f"💾 Checkpoint salvato: {filepath}")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x    = x.to(device)
            pred = model(x).argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())
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


def main():

    # ── 1. Config ────────────────────────────
    with open("./config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg         = config["training_spectrogram"]
    window_size = cfg["window_size"]
    stride      = cfg["stride"]
    dropout     = cfg["dropout"]
    resume      = cfg["resume"]
    fs          = config["data"]["fs"]

    print(f"🖥️  Dispositivo : {device}")
    print(f"🤖 Modello     : SpectrogramNet")
    print(f"📐 Window size : {window_size} | stride: {stride}")
    print(f"🔄 Resume      : {resume}")

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

    labels = create_labels(
        config["data"]["events_path"],
        total_samples=df.shape[1],
        fs=fs,
    )
    vals, counts = np.unique(labels, return_counts=True)
    label_names  = {-1: "Non assegnato", 0: "Riposo", 1: "Motor Imagery", 2: "Preparazione"}
    print(f"EEG shape: {df.shape} | labels: {len(labels)}")
    for v, c in zip(vals, counts):
        print(f"  {label_names.get(int(v), str(v)):20s}: {c} ({100*c/len(labels):.1f}%)")

    # ── 3. Dataset ───────────────────────────
    train_range = tuple(config["split"]["train"])
    val_range   = tuple(config["split"]["val"])
    test_range  = tuple(config["split"]["test"])

    print("\n" + "="*55)
    print("2. Dataset spettrogrammi...")
    print("="*55)

    train_dataset = EEGSpectrogramDataset(
        df, labels, train_range, window_size, stride=stride, fs=fs, augment=True
    )
    val_dataset   = EEGSpectrogramDataset(
        df, labels, val_range,   window_size, stride=stride, fs=fs, augment=False
    )
    test_dataset  = EEGSpectrogramDataset(
        df, labels, test_range,  window_size, stride=stride, fs=fs, augment=False
    )

    print(f"📦 Train finestre: {len(train_dataset)}")
    print(f"📦 Val finestre  : {len(val_dataset)}")
    print(f"📦 Test finestre : {len(test_dataset)}")

    pin_memory = device.type == "cuda"
    bs = cfg["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=pin_memory)

    # ── 4. Modello ───────────────────────────
    print("\n" + "="*55)
    print("3. Modello SpectrogramNet...")
    print("="*55)

    model = SpectrogramNet(
        n_channels=config["model"]["n_channels"],
        n_classes=config["model"]["n_classes"],
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {total_params:,}")

    # ── 5. Ottimizzatore, loss ────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_labels_raw   = labels[train_range[0]:train_range[1]]
    train_labels_valid = train_labels_raw[
        (train_labels_raw == 0) | (train_labels_raw == 1)
    ]
    class_weights = compute_class_weights(train_labels_valid).to(device)
    criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"⚙️  LR            : {cfg['lr']} | WD: {cfg['weight_decay']}")
    print(f"⚖️  Class weights : {class_weights.cpu().numpy()}")
    print(f"🕐 Patience      : {cfg['patience']}")

    # ── 6. Checkpoint dir ────────────────────
    ckpt_dir    = "./results/checkpoints/SPECTROGRAM"
    best_path   = f"{ckpt_dir}/best_model.pth"
    last_path   = f"{ckpt_dir}/last_checkpoint.pth"
    result_file = f"{ckpt_dir}/results.txt"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss    = float("inf")
    patience_counter = 0
    start_epoch      = 0

    # ── 6b. Resume ───────────────────────────
    if resume and os.path.exists(last_path):
        print(f"\n🔄 Caricamento LAST checkpoint: {last_path}")
        try:
            checkpoint       = torch.load(last_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch      = checkpoint["epoch"] + 1
            best_val_loss    = checkpoint.get("best_loss", checkpoint["loss"])
            patience_counter = checkpoint.get("patience_counter", 0)
            print(f"▶️  Ripresa da epoca {start_epoch} | best_val_loss={best_val_loss:.4f} | patience={patience_counter}")
        except RuntimeError as e:
            print(f"⚠️  Architettura cambiata — checkpoint incompatibile: {e}")
            print("   Inizio da zero.")
        except Exception as e:
            print(f"⚠️  Errore caricamento checkpoint: {e} — inizio da zero.")

    elif resume and os.path.exists(best_path):
        print(f"\n📂 Caricamento BEST modello: {best_path}")
        try:
            checkpoint       = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch      = checkpoint["epoch"] + 1
            best_val_loss    = checkpoint.get("best_loss", checkpoint["loss"])
            print(f"✅ BEST caricato | loss={checkpoint['loss']:.4f} | ripresa da epoca {start_epoch}")
        except RuntimeError as e:
            print(f"⚠️  Architettura cambiata — checkpoint incompatibile: {e}")
        except Exception as e:
            print(f"⚠️  Errore caricamento best model: {e}")

    # ── 7. Training ──────────────────────────
    print("\n" + "="*55)
    print("🚀 INIZIO TRAINING — SpectrogramNet")
    print("="*55)

    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\n📌 Epoch {epoch+1}/{cfg['epochs']}")

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
            print(f"  🏆 Nuovo best model! (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{cfg['patience']}")

        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, last_path,
                        best_loss=best_val_loss, patience_counter=patience_counter)

        if patience_counter >= cfg["patience"]:
            print(f"\n🛑 Early stopping dopo {cfg['patience']} epoche.")
            break

    # ── 8. Test finale ───────────────────────
    print("\n" + "="*55)
    print("🏁 VALUTAZIONE FINALE SUL TEST SET")
    print("="*55)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"📂 Caricato best model (epoca {ckpt['epoch']+1}, loss={ckpt['loss']:.4f})")

    test_acc = evaluate(model, test_loader, device)
    print(f"\n🎯 Test Accuracy: {test_acc:.2%}")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Model:         SpectrogramNet\n")
        f.write(f"Window size:   {window_size}\n")
        f.write(f"Stride:        {stride}\n")
        f.write(f"Test Accuracy: {test_acc:.2%}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Best Epoch:    {ckpt['epoch']+1}\n")

    print(f"📄 Risultati salvati in {result_file}")


if __name__ == "__main__":
    main()
