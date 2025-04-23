import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DSADataset(Dataset):
    def __init__(self, root_dirs, image_size=224):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        self.file_paths = []
        for root_dir in root_dirs:
            self.file_paths.extend([
                os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')
            ])

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # -> [C, H, W]
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('L')  # 灰度图 [1, H, W]
        image = self.transform(image)
        return image
