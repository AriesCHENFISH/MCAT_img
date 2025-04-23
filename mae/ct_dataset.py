import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class CTDataset(Dataset):
    def __init__(self, root_dirs, target_shape=(16, 16, 16)):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.file_paths = []
        for root_dir in root_dirs:
            self.file_paths.extend([
                os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".nii.gz")
            ])
        self.target_shape = target_shape

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = nib.load(path).get_fdata().astype(np.float32)
        if img.ndim == 4:
            img = img[..., 0]  # 只取第一个通道，变为 [D, H, W]

        img = self._normalize(img)
        img = self._resize_to_target(img)
        img = torch.from_numpy(img).unsqueeze(0)  # shape: [1, D, H, W]
        return img

    def _normalize(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-5)

    def _resize_to_target(self, volume):
        """
        volume: numpy array [D, H, W]
        """
        volume = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
        volume = F.interpolate(volume, size=self.target_shape, mode='trilinear', align_corners=False)
        volume = volume.squeeze().numpy()  # [D, H, W]
        return volume

