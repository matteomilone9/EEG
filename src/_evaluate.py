"""
Valutazione sul test set
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            output = model(x)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    # Report completo
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['Riposo', 'Immaginazione']))

    # Matrice di confusione
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Riposo', 'Immaginazione'],
                yticklabels=['Riposo', 'Immaginazione'])
    plt.title('Matrice di Confusione')
    plt.ylabel('Reale')
    plt.xlabel('Predetto')
    plt.savefig('confusion_matrix.png')

    accuracy = np.trace(cm) / np.sum(cm)
    return accuracy