import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import yaml
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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, best_loss=None, patience_counter=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss
    if patience_counter is not None:
        checkpoint['patience_counter'] = patience_counter

    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint salvato: {filepath}")


def main():
    # 1. Carica configurazione
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    print(
        f"📋 Config: {config['model']['name']}, "
        f"window={config['training']['window_size']}"
    )

    # 2. Preprocessing
    print("\n📂 Caricamento dati...")
    df = load_eeg_data(config['data']['edf_path'])

    print(f"\n🔧 Filtraggio {config['preprocessing']['filter_band']} Hz...")
    df = filter_eeg(
        df,
        lowcut=config['preprocessing']['filter_band'][0],
        highcut=config['preprocessing']['filter_band'][1],
    )

    # DEBUG dati
    print("\n" + "=" * 50)
    print("🔍 VERIFICA DATI DOPO FILTRO")
    print("=" * 50)
    print(f"Shape df: {df.shape}")
    print(f"Range: [{df.values.min():.4f}, {df.values.max():.4f}] µV")
    print(f"Media globale: {df.values.mean():.4f} µV")
    print(f"Std globale: {df.values.std():.4f} µV")

    # 3. Etichette e split
    print("\n🏷️ Creazione etichette...")
    labels = create_labels(config['data']['events_path'])
    print(f"Distribuzione etichette: {np.bincount(labels)}")

    train_indices = list(range(*config['split']['train']))
    val_indices = list(range(*config['split']['val']))
    test_indices = list(range(*config['split']['test']))

    print(f"\n📊 Split:")
    print(f" Train: {len(train_indices)} campioni")
    print(f" Val: {len(val_indices)} campioni")
    print(f" Test: {len(test_indices)} campioni")

    # 4. Dataset e DataLoader
    window_size = config['training']['window_size']
    print(f"\n📦 Creazione dataset con window_size={window_size}...")

    train_dataset = EEGDataset(df, labels, train_indices, window_size)
    val_dataset = EEGDataset(df, labels, val_indices, window_size)
    test_dataset = EEGDataset(df, labels, test_indices, window_size)

    print(f" Train dataset: {len(train_dataset)} campioni")
    print(f" Val dataset: {len(val_dataset)} campioni")
    print(f" Test dataset: {len(test_dataset)} campioni")

    pin_memory = True if device.type == 'cuda' else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    # 5. Modello
    print(f"\n🤖 Creazione modello {config['model']['name']}...")

    if config['model']['name'] == 'EEGAttentionNet':
        model = EEGAttentionNet(
            n_channels=config['model']['n_channels'],
            n_samples=window_size,
            n_classes=config['model']['n_classes'],
        ).to(device)
    else:
        model = SimpleEEGNet(
            n_channels=config['model']['n_channels'],
            n_samples=window_size,
            n_classes=config['model']['n_classes'],
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f" Parametri totali: {total_params:,}")

    # 6. Ottimizzatore, scheduler e loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    train_labels_subset = labels[train_indices]
    class_weights = compute_class_weights(train_labels_subset).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f" Learning rate: {config['training']['lr']}")
    print(f" Weight Decay: {config['training']['weight_decay']}")
    print(f" Class weights: {class_weights.cpu().numpy()}")
    print(" Patience: 5")

    # 7. ✅ CHECKPOINT RESUME - CORRETTO
    resume_path = './checkpoints/last_checkpoint.pth'
    best_model_path = './checkpoints/best_model.pth'
    resume = config['training'].get('resume', True)

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    # Crea cartella checkpoints
    os.makedirs('./checkpoints', exist_ok=True)

    # Caricamento LAST checkpoint (completo)
    if resume and os.path.exists(resume_path):
        print(f"\n📂 Caricamento LAST checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)

            # Debug contenuti
            print("🔍 Contenuti checkpoint:")
            for key in checkpoint.keys():
                print(f"  - {key}: ✓ presente")

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_loss', checkpoint['loss'])
            patience_counter = checkpoint.get('patience_counter', 0)

            print(f"✅ Ripresa da epoca {start_epoch}")
            print(f"   Best val loss: {best_val_loss:.4f}")
            print(f"   Patience counter: {patience_counter}")
        except Exception as e:
            print(f"❌ Errore caricamento last checkpoint: {e}")
            print("   Inizio da zero...")

    # Fallback: best_model.pth (parziale)
    elif os.path.exists(best_model_path):
        print(f"\n📂 Caricamento BEST modello: {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Carica optimizer/scheduler solo se presenti
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            print(f"✅ BEST caricato (loss: {checkpoint['loss']:.4f})")
            print("   ⚠️  Optimizer/scheduler ripartiti da zero se assenti")
        except Exception as e:
            print(f"❌ Errore caricamento best_model: {e}")

    # 8. Training loop ✅ CORRETTO
    print("\n" + "=" * 60)
    print("🚀 INIZIO TRAINING")
    print("=" * 60)

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n📌 Epoch {epoch + 1}/{config['training']['epochs']}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]['lr']

        print(f" Train Loss: {train_loss:.4f}")
        print(
            f" Val Loss: {val_loss:.4f}, "
            f"Acc: {val_acc:.2%}, F1: {val_f1:.3f}, LR: {current_lr:.2e}"
        )

        # Step scheduler
        scheduler.step(val_loss)

        # Early stopping e checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, best_model_path
            )
            print(f"💾 BEST salvato: {best_model_path}")
            print(f" ✅ Miglior modello! (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        print(f" ✅ PATIENCE: ", patience_counter)

        # Salva SEMPRE last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, resume_path,
            best_loss=best_val_loss, patience_counter=patience_counter
        )
        print(f"💾 LAST salvato: {resume_path}")

        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(
                f" ⏹️ Early stopping dopo "
                f"{config['training']['patience']} epoche senza miglioramenti"
            )
            break

    # 9. Test finale ✅ CORRETTO
    print("\n" + "=" * 60)
    print("📊 VALUTAZIONE FINALE")
    print("=" * 60)

    # Carica miglior modello per test
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        f"📂 Caricato miglior modello "
        f"(epoca {checkpoint['epoch'] + 1}, loss={checkpoint['loss']:.4f})"
    )

    test_acc = evaluate_model(model, test_loader, device)
    print(f"\n🎯 Test Accuracy: {test_acc:.2%}")

    # ✅ Fix final_epoch
    final_epoch = locals().get('epoch', config['training']['epochs'] - 1) + 1

    # Salva risultati
    with open('./results.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2%}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Final Epoch: {final_epoch}\n")

    print("\n✅ Training completato! Risultati in results.txt")
    print(f"💾 Checkpoints: {best_model_path}, {resume_path}")


if __name__ == "__main__":
    main()
