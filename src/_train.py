import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import psutil
import os


def train_epoch(model, loader, optimizer, criterion, device,
                use_amp=False, scaler=None, epoch=None, total_epochs=None):
    """
    Allenamento per una epoca con metriche in tempo reale

    Args:
        model: Il modello PyTorch
        loader: DataLoader per training
        optimizer: Ottimizzatore
        criterion: Funzione di loss
        device: Device (cuda/cpu)
        use_amp: Usa mixed precision training
        scaler: GradScaler per AMP
        epoch: Numero epoca corrente (per logging)
        total_epochs: Numero totale epoche (per logging)

    Returns:
        float: Loss media dell'epoca
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    # Crea descrizione per la barra
    if epoch is not None and total_epochs is not None:
        desc = f"📚 Epoca {epoch}/{total_epochs}"
    else:
        desc = "📚 Training"

    # Barra di progresso personalizzata
    pbar = tqdm(
        loader,
        desc=desc,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        ncols=100,
        colour='green',
        leave=False
    )

    for batch_idx, (x, y) in enumerate(pbar):
        # Sposta dati sul device
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Più efficiente di zero_grad()

        # Forward pass (con o senza mixed precision)
        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                output = model(x)
                loss = criterion(output, y)

            # Backward con scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Calcola metriche batch
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        batch_loss = loss.item()
        total_loss += batch_loss
        batch_acc = 100 * (pred == y).sum().item() / y.size(0)

        # Aggiorna barra con metriche in tempo reale
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{batch_acc:.2f}%',
            'avg_loss': f'{(total_loss/(batch_idx+1)):.4f}'
        })

    # Calcola metriche finali
    avg_loss = total_loss / len(loader)
    epoch_acc = 100 * correct / total
    epoch_time = time.time() - start_time

    # Logging aggiuntivo (non nella barra)
    print(f"\n  ✓ Train: loss={avg_loss:.4f}, acc={epoch_acc:.2f}%, tempo={epoch_time:.1f}s")

    return avg_loss


def validate(model, loader, criterion, device, return_details=False):
    """
    Validazione con metriche complete

    Args:
        model: Il modello PyTorch
        loader: DataLoader per validazione
        criterion: Funzione di loss
        device: Device (cuda/cpu)
        return_details: Se True, restituisce anche predictions e labels

    Returns:
        tuple: (loss_media, accuracy, f1_score) o metriche più dettagliate
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # Per probabilità (utile per ROC curve)

    # Barra di progresso per validazione
    pbar = tqdm(
        loader,
        desc="📊 Validazione",
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        ncols=100,
        colour='blue',
        leave=False
    )

    start_time = time.time()

    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item()

            # Probabilità tramite softmax
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Aggiorna barra con loss corrente
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calcola metriche complete
    loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Metriche aggiuntive per classificazione binaria
    if len(set(all_labels)) == 2:
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
    else:
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

    val_time = time.time() - start_time

    # Logging compatto
    print(f"  ✓ Val: loss={loss:.4f}, acc={acc:.2%}, f1={f1:.3f}, tempo={val_time:.1f}s")

    if return_details:
        return loss, acc, f1, precision, recall, all_preds, all_labels, all_probs

    return loss, acc, f1


def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion,
                              device, config, scheduler=None):
    """
    Training completo con early stopping, checkpointing e metriche

    Args:
        model: Il modello PyTorch
        train_loader: DataLoader training
        val_loader: DataLoader validazione
        optimizer: Ottimizzatore
        criterion: Funzione di loss
        device: Device
        config: Dizionario con configurazione
        scheduler: Learning rate scheduler (opzionale)

    Returns:
        dict: Storico delle metriche
    """

    # Parametri da config
    epochs = config.get('epochs', 100)
    patience = config.get('patience', 15)
    use_amp = config.get('use_amp', False)
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')

    # Crea directory checkpoint se non esiste
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Per mixed precision
    scaler = torch.amp.GradScaler() if use_amp else None

    # Storico metriche
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    print("\n" + "="*70)
    print("🚀 INIZIO TRAINING")
    print(f"   Modello: {model.__class__.__name__}")
    print(f"   Epoche: {epochs}, Early stopping: {patience}")
    print(f"   Device: {device}, Mixed precision: {use_amp}")
    print("="*70)

    # Barra principale per le epoche
    epoch_pbar = tqdm(
        range(epochs),
        desc="📈 Progresso",
        position=0,
        leave=True,
        ncols=100,
        colour='cyan'
    )

    try:
        for epoch in epoch_pbar:
            # Training
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                use_amp=use_amp, scaler=scaler,
                epoch=epoch+1, total_epochs=epochs
            )

            # Validazione
            val_loss, val_acc, val_f1 = validate(
                model, val_loader, criterion, device
            )

            # Salva storico
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)

            # Aggiorna scheduler se presente
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Aggiorna barra epoche
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.3f}',
                'val_loss': f'{val_loss:.3f}',
                'val_acc': f'{val_acc:.2%}',
                'lr': f'{current_lr:.2e}'
            })

            # Checkpoint e early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Salva miglior modello
                checkpoint_path = f"{checkpoint_dir}/best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'accuracy': val_acc,
                    'config': config
                }, checkpoint_path)

                print(f"\n  ✅ Miglior modello salvato! (loss: {best_val_loss:.4f}, acc: {val_acc:.2%})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  ⏹️ Early stopping dopo {patience} epoche senza miglioramenti")
                    break

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrotto dall'utente")

    finally:
        # Salva modello finale
        final_path = f"{checkpoint_dir}/final_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'history': history
        }, final_path)
        print(f"\n💾 Modello finale salvato in: {final_path}")

    print("\n" + "="*70)
    print("🏁 TRAINING COMPLETATO")
    print(f"   Miglior validation loss: {best_val_loss:.4f}")
    print(f"   Ultima validation accuracy: {val_acc:.2%}")
    print("="*70)

    return history


def get_memory_usage():
    """Restituisce l'utilizzo di memoria RAM/GPU"""
    memory_info = {}

    # RAM
    process = psutil.Process(os.getpid())
    memory_info['ram'] = process.memory_info().rss / 1024 / 1024  # MB

    # GPU se disponibile
    if torch.cuda.is_available():
        memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        memory_info['gpu_cached'] = torch.cuda.memory_reserved() / 1024 / 1024  # MB

    return memory_info