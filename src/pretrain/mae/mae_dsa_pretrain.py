import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dsa_dataset import DSADataset
from model.mae2d import MAE2D
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 数据加载
dataset = DSADataset([
    '/mnt/sdc/chenxi/new_no/DSA'
])
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 初始化模型
model = MAE2D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        loss, _ = model(batch)  # 计算损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'mae2d_dsa_pretrained.pth')
