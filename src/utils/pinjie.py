import os
import torch
from collections import defaultdict

# 原始特征目录
feature_dir = "/home/chenxi/MCAT/mydata/CT_try/"

# 保存整合后的目录
output_dir = "/home/chenxi/MCAT/mydata/CT_merged/"
os.makedirs(output_dir, exist_ok=True)

# 1. 收集所有patch文件
all_files = [f for f in os.listdir(feature_dir) if f.endswith(".pt")]

# 2. 根据id归类
id_to_files = defaultdict(list)
for f in all_files:
    id_part = f.split("_patch_")[0]
    id_to_files[id_part].append(f)

# 3. 遍历每个id，读取对应patch特征，拼接
for id_part, patch_files in id_to_files.items():
    patch_files = sorted(patch_files, key=lambda x: int(x.split("_patch_")[1].split(".pt")[0]))  # 按patch编号排序
    features = []

    for pf in patch_files:
        feature = torch.load(os.path.join(feature_dir, pf))  # 读取每个patch特征
        features.append(feature.unsqueeze(0))  # 加batch维度方便拼接

    merged_feature = torch.cat(features, dim=0)  # (n_patches, 512)
    
    # 保存
    save_path = os.path.join(output_dir, f"{id_part}.pt")
    torch.save(merged_feature, save_path)

    print(f"Saved {id_part}: {merged_feature.shape}")

print("✅ 所有ID特征拼接完成！")
