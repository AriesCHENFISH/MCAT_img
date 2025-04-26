import os
import pydicom
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 设置路径
input_root = "/mnt/sdc/chenxi/new_no/CT"
output_root = "/mnt/sdc/chenxi/new_no/CT_patches"

os.makedirs(output_root, exist_ok=True)

# patch大小
patch_size = (34, 256, 256)  # (Depth, Height, Width)
stride = patch_size  # 不重叠

def load_dicom_series(folder_path):
    dicom_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".dcm")],
        key=lambda x: int(pydicom.dcmread(os.path.join(folder_path, x)).InstanceNumber)
    )
    slices = [pydicom.dcmread(os.path.join(folder_path, f)) for f in dicom_files]
    volume = np.stack([s.pixel_array for s in slices], axis=0)  # (D, H, W)
    return volume, slices

def extract_patches(volume):
    volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    patches = volume.unfold(2, patch_size[0], stride[0])\
                    .unfold(3, patch_size[1], stride[1])\
                    .unfold(4, patch_size[2], stride[2])  # (1,1,num_d,num_h,num_w,D,H,W)
    patches = patches.contiguous().view(-1, patch_size[0], patch_size[1], patch_size[2])  # (n_patches, D, H, W)
    return patches

def save_patch(patch_tensor, slices_template, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    patch_np = patch_tensor.numpy()
    D, H, W = patch_np.shape

    for idx in range(D):
        # 复制原来的一张slice header作为模板
        ds = slices_template[idx % len(slices_template)].copy()
        ds.PixelData = patch_np[idx].astype(np.uint16).tobytes()
        ds.Rows, ds.Columns = H, W
        ds.InstanceNumber = idx + 1
        ds.SliceLocation = idx  # 你可以根据需要微调
        save_path = os.path.join(save_dir, f"{idx:04d}.dcm")
        ds.save_as(save_path)

def process_one_volume(id_folder):
    id_path = os.path.join(input_root, id_folder)
    volume, slices_template = load_dicom_series(id_path)
    patches = extract_patches(volume)

    for patch_idx, patch in enumerate(patches):
        patch_save_dir = os.path.join(output_root, f"{id_folder}_patch_{patch_idx}")
        save_patch(patch, slices_template, patch_save_dir)

def main():
    all_ids = sorted(os.listdir(input_root))
    for id_folder in tqdm(all_ids, desc="Processing CT Volumes"):
        try:
            process_one_volume(id_folder)
        except Exception as e:
            print(f"❗ Error processing {id_folder}: {e}")

if __name__ == "__main__":
    main()
