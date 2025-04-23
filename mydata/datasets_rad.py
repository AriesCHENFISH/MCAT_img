# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import pdb

import PIL
import SimpleITK as sitk
import numpy as np
import torch
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from torchvision import datasets, transforms

from timm.data import create_transform  # 根据指定的参数配置创建一个图像数据预处理的转换器可以包括对图像进行大小调整、裁剪、标准化、数据增强等操作
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

output_size = (128, 128, 128)

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    # both_none = extensions is None and is_valid_file is None
    # both_something = extensions is not None and is_valid_file is not None
    # if both_none or both_something:
    #     raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, fnames, _ in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if os.path.isdir(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        # if extensions is not None:
        #     msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        # raise FileNotFoundError(msg)
    # pdb.set_trace()
    return instances
# class DICOMFolder(datasets.ImageFolder):
#     def __init__(self, root, transform=None, target_transform=None, loader = None):
#         super().__init__(root, transform=transform, target_transform=target_transform)
#         self.loader = loader


#     # def _find_classes(self, dir):
#     #     classes = [d.name for d in os.scandir(dir) if d.is_dir()]
#     #     classes.sort()
#     #     class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#     #     return classes, class_to_idx
#     @staticmethod
#     def make_dataset(
#         directory: str,
#         class_to_idx: Dict[str, int],
#         extensions: Optional[Tuple[str, ...]] = None,
#         is_valid_file: Optional[Callable[[str], bool]] = None,
#         **kwargs  # 忽略 `allow_empty`
#     ) -> List[Tuple[str, int]]:
#         """Generates a list of samples of a form (path_to_sample, class)."""
#         if class_to_idx is None:
#             raise ValueError("The class_to_idx parameter can't be None.")
        
#         # 确保 `allow_empty` 参数不会导致错误
#         if "allow_empty" in kwargs:
#             print("⚠️ Warning: `allow_empty` argument was ignored.")  # 打印警告
        
#         return make_dataset(directory, class_to_idx, extensions=None, is_valid_file=is_valid_file)


import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import os

output_size = (128, 128, 128)

class DICOMDataset(Dataset):  
    def __init__(self, root):
        """
        读取 root 目录下的所有子文件夹，每个子文件夹都是一个 CT 序列（多个 .dcm 文件）。
        """
        self.root = root
        # 获取所有子文件夹（每个子文件夹是一个 CT 序列）
        self.series_folders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    def __len__(self):
        return len(self.series_folders)

    def __getitem__(self, index):
        """
        读取一个 CT 序列（整个子文件夹），返回处理后的体数据 (1, 1, D, H, W)
        """
        folder_path = self.series_folders[index]
        dicom_tensor = self.load_dicom_series(folder_path)

        # 目录名称作为样本 ID
        sample_id = os.path.basename(folder_path)
        return dicom_tensor, sample_id

    def load_dicom_series(self, folder_path):
        """
        读取 DICOM 序列（子文件夹内的所有 DICOM 文件）
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)

        if not dicom_names:
            raise RuntimeError(f"在 {folder_path} 中未找到 DICOM 文件")

        reader.SetFileNames(dicom_names)
        image = reader.Execute()  # ✅ 读取 DICOM 序列

        # 体数据重采样
        new_spacing = [(old_sz * old_spc) / new_sz for old_sz, old_spc, new_sz in
                       zip(image.GetSize(), image.GetSpacing(), output_size)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(output_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)

        # ✅ 修正：ResampleImageFilter 需要传入 image
        output_image = resampler.Execute(image)  

        output_array = sitk.GetArrayFromImage(output_image).astype(np.float32)

        # 归一化
        output_array = output_array / 1000.0  # 归一化处理

        # 转换为 PyTorch Tensor
        output_tensor = torch.tensor(output_array, dtype=torch.float32)
        output_tensor = output_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        print(f"📂 读取 {folder_path}, 输出维度: {output_tensor.shape}")
        return output_tensor




names=[]
def my_loader(path: str):
    try:
        # 读取输入文件夹中的所有 DICOM 文件
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        #print(dicom_names)
        if not dicom_names:
            raise RuntimeError(f"在目录 {path} 中未找到 DICOM 文件")

        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        name = os.path.basename(os.path.dirname(dicom_names[0]))
        names.append(name)

        # 设置输出图像的大小
        new_spacing = [(old_sz * old_spc) / new_sz for old_sz, old_spc, new_sz in
                       zip(image.GetSize(), image.GetSpacing(), output_size)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(output_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)

        # 执行重采样
        output_image = resampler.Execute(image)

        # 将 SimpleITK 图像转换为 NumPy 数组，并将像素值转换为浮点数类型
        output_array = sitk.GetArrayFromImage(output_image).astype(np.float32)

        # 归一化操作
        normalized_array = output_array / 1000.0  # 这个除以一千的操作不太严谨，你自己再斟酌一下

        # 将 NumPy 数组转换为 PyTorch Tensor
        output_tensor = torch.tensor(normalized_array, dtype=torch.float32)
        output_tensor = output_tensor.unsqueeze(0).unsqueeze(0)
        
        print(output_tensor.shape)
        return output_tensor

    except Exception as e:
        print(f"加载图像时出错：{e}")
        return None
# def my_loader(path: str):
#     # 读取输入文件夹中的所有 DICOM 文件
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(path)
#     reader.SetFileNames(dicom_names)
#     image = reader.Execute()
#     global names
#     name=os.path.basename(os.path.dirname(dicom_names[0]))
#     names.append(name)
#     # 设置输出图像的大小
#     new_spacing = [(old_sz * old_spc) / new_sz for old_sz, old_spc, new_sz in
#                    zip(image.GetSize(), image.GetSpacing(), output_size)]
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetSize(output_size)
#     resampler.SetOutputSpacing(new_spacing)
#     resampler.SetOutputDirection(image.GetDirection())
#     resampler.SetOutputOrigin(image.GetOrigin())
#     resampler.SetInterpolator(sitk.sitkLinear)

#     # 执行重采样
#     output_image = resampler.Execute(image)

#     # 将 SimpleITK 图像转换为 NumPy 数组，并将像素值转换为浮点数类型
#     output_array = sitk.GetArrayFromImage(output_image).astype(np.float32)

#     normalized_array = output_array / 1000#这个除以一千的操作不太严谨，你自己再斟酌一下

#     # 将 NumPy 数组转换为 PyTorch Tensor
#     output_tensor = torch.tensor(normalized_array, dtype=torch.float32)  # 30*224*224
#     output_tensor = output_tensor.unsqueeze(0)
#     output_tensor = output_tensor.unsqueeze(0)
#     print(output_tensor.shape)
#     return output_tensor


# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)
#     root = os.path.join(args.data_path)
#     dataset= DICOMFolder(root, transform=None, loader = my_loader)

#     print("输入成功")
#     global names
#     print(dataset)

#     return dataset,names


def build_dataset(is_train, args):
    """
    构建 DICOM 数据集，并返回 names（即每个子文件夹的名称）
    """
    root = os.path.join(args.data_path)
    dataset = DICOMDataset(root)

    # 获取所有样本的 ID（子文件夹名称）
    names = dataset.series_folders  # 直接返回子文件夹路径列表
    names = [os.path.basename(name) for name in names]  # 提取文件夹名

    print("✅ 数据集构建成功")
    print(f"📊 数据集大小: {len(dataset)}")
    print(f"📂 样本 IDs: {names}")

    first_item = dataset[0]
    if isinstance(first_item, tuple):
        image = first_item[0]
    else:
        image = first_item
    print(f"🧾 第一个样本的类型: {type(image)}")
    print(f"🔍 第一个样本的形状: {image.shape}")

    return dataset, names



def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    # t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_transform_mri(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
