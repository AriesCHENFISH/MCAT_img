import torch
import os
from torch.utils.data import DataLoader
from dsa_dataset import DSADataset
from model.mae2d import MAE2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 解析 ROI 文件的函数
def parse_roi(roi_path):
    """解析 ROI 文件，获取 ROI 中心坐标、宽度和高度"""
    with open(roi_path, 'r') as f:
        roi_data = f.readline().strip().split()  # 读取并拆分
        # 将 x_center, y_center, width, height 转换为浮点数
        x_center, y_center, width, height = map(float, roi_data[1:])  # 转为浮点数
    return x_center, y_center, width, height

# 数据加载
dataset = DSADataset([
    '/mnt/sdc/chenxi/new_no/DSA'
])
loader = DataLoader(dataset, batch_size=1, shuffle=False)

file_names = []
for root_dir in ['/mnt/sdc/chenxi/new_no/DSA']:
    file_names.extend([os.path.splitext(f)[0] for f in os.listdir(root_dir) if f.endswith('.jpg')])

# 加载预训练模型
model = MAE2D().to(device)
model.load_state_dict(torch.load('mae2d_dsa_pretrained.pth', map_location=device))
model.eval()

# 创建保存目录
os.makedirs("features_dsa_whole", exist_ok=True)

# 特征提取并保存为同名 .pt 文件
with torch.no_grad():
    for idx, batch in enumerate(loader):
        batch = batch.to(device)

        # 获取 ROI 信息并转换为张量
        roi_path = f"/mnt/sdc/chenxi/new_no/TXT/{file_names[idx]}.txt"
        x_center, y_center, width, height = parse_roi(roi_path)
        roi_embedding = torch.tensor([x_center, y_center, width, height]).float().unsqueeze(0).to(device)

        # 提取特征
        feature = model.encode_only(batch, roi_embedding)  # [1, embed_dim]

        # 保存特征和 ROI 信息
        filename = file_names[idx]
        save_path = os.path.join("features_dsa_whole", f"{filename}.pt")
        torch.save(feature.cpu(), save_path)
        # torch.save({"feature": feature.cpu(), "roi": [x_center, y_center, width, height]}, save_path)

print("✅ DSA 特征提取完成，已保存至 features_dsa/ 目录")
