import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ct_dataset import CTDataset
from model.mae3d import MAE3D  # 你需要导入 MAE3D 模型
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
dataset = CTDataset([
    '/mnt/sdc/chenxi/CT_DSA_data/CT_nii'
])

loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 初始化模型
model = MAE3D(embed_dim=512, depth=8).to(device)  # 增加嵌入维度和 Transformer 层数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

# 训练过程
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

# 保存整个模型（推荐简单方式）
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mae3d_ct_pretrained.pth")
