def log_training_metrics(log_file_path, metrics):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Epoch {metrics['epoch']}: Train Loss: {metrics['train_loss']} | Train Acc: {metrics['train_acc']} | AUC: {metrics['train_auc']}\n")
        log_file.write(f"Val Loss: {metrics['val_loss']} | Val Acc: {metrics['val_acc']} | AUC: {metrics['val_auc']}\n")
