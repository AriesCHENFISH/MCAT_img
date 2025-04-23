import torch
from sklearn.metrics import roc_auc_score

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    all_labels, all_probs, all_ids = [], [], []

    with torch.no_grad():
        for ct_features, dsa_features, labels, sample_ids in dataloader:
            ct_features, dsa_features, labels = ct_features.to(device), dsa_features.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            outputs, _ = model(ct_features, dsa_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total_samples += labels.numel()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(preds.cpu().numpy().flatten())
            all_ids.extend(sample_ids)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, auc
