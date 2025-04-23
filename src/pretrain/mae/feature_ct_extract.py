import torch
import os
from torch.utils.data import DataLoader
from ct_dataset import CTDataset
from model.mae3d import MAE3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_path = [
    '/mnt/sdc/chenxi/new_no/CT_nii'
]
# 数据加载
dataset = CTDataset([
    '/mnt/sdc/chenxi/new_no/CT_nii'
])
loader = DataLoader(dataset, batch_size=1, shuffle=False)
file_names = []
for root_dir in all_path:
    file_names.extend([
        os.path.splitext(os.path.splitext(f)[0])[0]  # 去掉 .nii.gz
        for f in os.listdir(root_dir) if f.endswith('.nii.gz')
    ])
# 加载预训练模型
model = MAE3D(embed_dim=512, depth=8).to(device)  # 使用同样的设置
model.load_state_dict(torch.load("models/mae3d_ct_pretrained.pth", map_location=device))

model.eval()

# 创建保存目录
os.makedirs("features_ct", exist_ok=True)

# 特征提取并保存为同名 .pt 文件
with torch.no_grad():
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        feature = model.encode_only(batch)  # [1, N, embed_dim] 特征
        filename = file_names[idx]
        save_path = os.path.join("features_ct", f"{filename}.pt")
        torch.save(feature.cpu(), save_path)

print("✅ 特征提取完成，已保存至 features_ct/ 目录")
