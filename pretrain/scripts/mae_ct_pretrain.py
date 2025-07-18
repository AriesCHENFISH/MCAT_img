import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.datasets.ct_dataset import CTDataset
from models.mae3d import MAE3D  
import os

device = torch.device('cuda')

dataset = CTDataset([
    '../raw_data/CT_nii'
])


loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = MAE3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

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
    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "mae3d_ct_pretrained.pth")

