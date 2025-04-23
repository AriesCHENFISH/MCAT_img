import os
import json
import torch
from torch.utils.data import Dataset

class MCATDataset(Dataset):
    def __init__(self, ct_dir, dsa_dir, data_file):
        """
        Args:
            ct_dir (str): CT特征文件夹路径
            dsa_dir (str): DSA特征文件夹路径
            data_file (str): 存放训练数据和标签的JSON文件路径
        """
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.ct_dir = ct_dir
        self.dsa_dir = dsa_dir
        self.samples = self.data['samples']
        self.labels = self.data['labels']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        ct_path = os.path.join(self.ct_dir, f"{sample_id}.pt")
        ct_feature = torch.load(ct_path).unsqueeze(1)

        dsa_path = os.path.join(self.dsa_dir, f"{sample_id}.pt")
        dsa_feature = torch.load(dsa_path)

        return ct_feature, dsa_feature, label, sample_id
