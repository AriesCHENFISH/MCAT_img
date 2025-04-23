import os
import torch
import torch.optim as optim
from data_loader.dataset import MCATDataset
from models.model_cottan_3D import MCAT_Surv
from training.train import train_model
from training.evaluate import evaluate_model
from training.logger import log_training_metrics
from results.plots import plot_metrics

def main():
    ct_feature_dir = "/path/to/CT_features/"
    dsa_feature_dir = "/path/to/DSA_features/"
    label_path = "/path/to/label_file.json"
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    train_dataset, val_dataset = split_data_from_label_file(label_path, ct_feature_dir, dsa_feature_dir)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = MCAT_Surv(fusion='concat', dropout=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()

    log_file_path = "training_log.txt"
    epochs = 30
    for epoch in range(epochs):
        train_loss, train_acc, train_auc = train_model(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, val_auc = evaluate_model(model, val_loader, criterion, device)

        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc
        }
        log_training_metrics(log_file_path, metrics)
        plot_metrics(metrics)

if __name__ == "__main__":
    main()
