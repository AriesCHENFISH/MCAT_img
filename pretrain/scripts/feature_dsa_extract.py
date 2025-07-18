import os
import torch
from torch.utils.data import DataLoader
from data.datasets.dsa_dataset import DSADataset
# from models.mae2d import MAE2D
from models.resnet_autoencoder import ResNetAutoEncoder

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

all_paths = [
    '../raw_data/DSA'
]

dataset = DSADataset(all_paths)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

file_names = []
for root_dir in all_paths:
    file_names.extend([
        os.path.splitext(f)[0] for f in os.listdir(root_dir) if f.endswith('.jpg')
    ])

model = ResNetAutoEncoder().to(device)
model.load_state_dict(torch.load('outputs/resnet_dsa_pretrained.pth', map_location=device))
model.eval()

os.makedirs("data/features/features_dsa_resnet", exist_ok=True)

with torch.no_grad():
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        feature = model.encode_only(batch)  # [1, embed_dim]
        filename = file_names[idx]
        save_path = os.path.join("data/features/features_dsa_resnet", f"{filename}.pt")
        torch.save(feature.cpu(), save_path)

print("已保存至 data/features/features_dsa/ 目录")
