import torch
from torch.utils.data import DataLoader
from dsa_dataset import DSADataset
from model.resnet_autoencoder import ResNetAutoEncoder
import matplotlib.pyplot as plt

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

dataset = DSADataset('/mnt/sdc/chenxi/new_no/DSA')
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = ResNetAutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 15
loss_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        loss, _ = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'resnet_dsa_pretrained.pth')

# 绘制loss曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.title('Training Loss Curve (ResNet Autoencoder)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('resnet_loss_curve.png')
plt.show()
