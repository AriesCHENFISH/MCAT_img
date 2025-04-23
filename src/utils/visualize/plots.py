import matplotlib.pyplot as plt

def plot_metrics(epochs, train_losses, val_losses, train_accuracies, val_accuracies, train_auc, val_auc):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # AUC plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_auc, label='Train AUC', color='blue', marker='o')
    plt.plot(epochs, val_auc, label='Validation AUC', color='red', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('Train and Validation AUC')
    plt.legend()
    plt.show()
