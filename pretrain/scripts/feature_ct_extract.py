import torch
from torch.utils.data import DataLoader
from data.datasets.ct_dataset import CTDataset
from models.mae3d import MAE3D
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_path = [
    '..raw_data/CT_nii'
]
dataset = CTDataset([
    '..raw_data/CT_nii'
])
loader = DataLoader(dataset, batch_size=1, shuffle=False)
file_names = []
for root_dir in all_path:
    file_names.extend([
        os.path.splitext(os.path.splitext(f)[0])[0] 
        for f in os.listdir(root_dir) if f.endswith('.nii.gz')
    ])

model = MAE3D().to(device)
model.load_state_dict(torch.load("outputs/mae3d_ct_pretrained.pth", map_location=device))

model.eval()

os.makedirs("data/features/features_ct", exist_ok=True)

with torch.no_grad():
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        feature = model.encode_only(batch)  # [1, embed_dim]
        filename = file_names[idx]
        save_path = os.path.join("data/features/features_ct", f"{filename}.pt")
        torch.save(feature.cpu(), save_path)

print("已保存至 data/features/features_ct/ 目录")