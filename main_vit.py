# main_vit.py
# Training pipeline per EEGViT (vit_small o vit_base su immagini GAF)

import torch
import numpy as np
import os
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report

from src import load_eeg_data, filter_eeg, create_labels
from src import compute_class_weights
from src_vit import EEGViTDataset, EEGViT


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
            out  = model(x)
            pred = out.argmax(dim=1)
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


# ─────────────────────────────────────────────
def main():

    # ── 1. Config ────────────────────────────
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    cfg        = config["training_vit"]
    window_size = cfg["window_size"]
    stride      = cfg.get("stride", 250)
    gaf_type    = cfg.get("gaf_type", "GASF")
    image_size  = cfg.get("image_size", 32)
    target_size = cfg.get("target_size", 224)
    model_name  = cfg.get("model_name", "vit_small_patch16_224")
    freeze_vit  = cfg.get("freeze_vit", False)
    fs          = config["data"]["fs"]

    print(f"Modello:    {model_name}")
    print(f"GAF:        {gaf_type} {image_size}→{target_size}px")
    print(f"Freeze ViT: {freeze_vit}")

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
    T      = df.shape[1]
    labels = labels[:T] if len(labels) > T else np.pad(labels, (0, T - len(labels)))

    print(f"EEG: {df.shape} | labels: {len(labels)} | dist: {np.bincount(labels)}")

    # ── 3. Dataset ───────────────────────────
    train_range = tuple(config["split"]["train"])
    val_range   = tuple(config["split"]["val"])
    test_range  = tuple(config["split"]["test"])

    print("\n" + "="*55)
    print(f"2. Dataset ViT ({model_name})...")
    print("="*55)

    train_dataset = EEGViTDataset(
        df, labels, train_range, window_size, stride=stride,
        gaf_type=gaf_type, image_size=image_size,
        target_size=target_size, augment=True
    )
    val_dataset = EEGViTDataset(
        df, labels, val_range, window_size, stride=stride,
        gaf_type=gaf_type, image_size=image_size,
        target_size=target_size, augment=False
    )
    test_dataset = EEGViTDataset(
        df, labels, test_range, window_size, stride=stride,
        gaf_type=gaf_type, image_size=image_size,
        target_size=target_size, augment=False
    )

    # metti questo subito dopo create_labels in main_vit.py e dimmi l'output
    print(np.bincount(labels))
    print(f"Train finestre: {len(train_dataset)}")
    print(f"Val finestre:   {len(val_dataset)}")
    print(f"Test finestre:  {len(test_dataset)}")

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
    print(f"3. Modello EEGViT ({model_name})...")
    print("="*55)

    model = EEGViT(
        n_channels=config["model"]["n_channels"],
        n_classes=config["model"]["n_classes"],
        model_name=model_name,
        pretrained=True,
        dropout=cfg.get("dropout", 0.3),
        freeze_vit=freeze_vit,
        target_size=target_size,
    ).to(device)

    # ── 5. Ottimizzatore ─────────────────────
    # Se ViT è congelato, allena solo channel_proj + classifier
    # Se ViT è libero, usa lr differenziato (ViT più basso)
    if freeze_vit:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.channel_proj.parameters(), "lr": cfg["lr"]},
            {"params": model.vit.parameters(),          "lr": cfg["lr"] * 0.1},
            {"params": model.classifier.parameters(),   "lr": cfg["lr"]},
        ], weight_decay=cfg["weight_decay"])

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=1e-6,
    )

    train_labels_sub = labels[train_range[0]:train_range[1]]
    class_weights    = compute_class_weights(train_labels_sub).to(device)
    criterion        = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── 6. Checkpoint ────────────────────────
    safe_name = model_name.replace("/", "_")
    ckpt_dir  = f"./checkpoints_vit/{safe_name}"
    best_path = f"{ckpt_dir}/best_model.pth"
    last_path = f"{ckpt_dir}/last_checkpoint.pth"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss    = float("inf")
    patience_counter = 0

    # ── 7. Training ──────────────────────────
    print("\n" + "="*55)
    print(f"INIZIO TRAINING — EEGViT ({model_name})")
    print("="*55)

    for epoch in range(cfg["epochs"]):
        print(f"\n📌 Epoch {epoch+1}/{cfg['epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"  Val   Loss: {val_loss:.4f}   | Acc: {val_acc:.2%} | LR: {current_lr:.2e}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path)
            print(f"  ✅ Best model! (loss: {best_val_loss:.4f})")
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
    print(f"VALUTAZIONE FINALE — {model_name}")
    print("="*55)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Caricato best model (epoca {ckpt['epoch']+1}, loss={ckpt['loss']:.4f})")

    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2%}")

    os.makedirs("./results", exist_ok=True)
    result_file = f"./results/results_vit_{safe_name}.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Model:         EEGViT\n")
        f.write(f"Backbone:      {model_name}\n")
        f.write(f"GAF type:      {gaf_type}\n")
        f.write(f"Image size:    {image_size}→{target_size}px\n")
        f.write(f"Freeze ViT:    {freeze_vit}\n")
        f.write(f"Window size:   {window_size}\n")
        f.write(f"Test Accuracy: {test_acc:.2%}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Best Epoch:    {ckpt['epoch']+1}\n")

    print(f"Risultati salvati in {result_file}")


if __name__ == "__main__":
    main()
