from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util
import scipy
import scipy.io
import os.path
import re
import numpy as np
import warnings

try:
    import rasterio
except ImportError:
    print("警告: rasterio 库未安装。物理数据加载功能将无法使用。")
    print("请运行 'pip install rasterio' 来安装它。")
    rasterio = None

# --- 核心修复：重写 load_img_name_list 以处理带空格的路径 ---
def load_img_name_list(dataset_path):
    """
    从文本文件中加载图像文件名列表。
    新功能：通过查找第一个'.png'来正确解析包含空格的路径。
    """
    expanded_path = os.path.expanduser(dataset_path)
    img_name_list = []
    try:
        with open(expanded_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 找到第一个.png出现的位置，并提取整个路径
                end_pos = line.find('.png')
                if end_pos != -1:
                    first_path = line[:end_pos + 4]
                    img_name_list.append(first_path)
                else:
                    # 如果行中没有.png，但有内容，可能是个问题，先用split作为备用
                    warnings.warn(f"在行中找不到'.png'，将使用split()作为备用: '{line}'")
                    img_name_list.append(line.split()[0])

    except FileNotFoundError:
        print(f"错误: 列表文件未找到 {expanded_path}")
        return np.array([])
    except Exception as e:
        print(f"加载图像列表时出错 {expanded_path}: {e}")
        return np.array([])
    
    if not img_name_list:
        print(f"警告: 从 {expanded_path} 加载的图像列表为空。")
        
    return np.array(img_name_list)


class CDDataset(Dataset):
    def __init__(self, dataroot, physical_data_root=None, resolution=256, split='train', data_len=-1):
        """
        Args:
            dataroot: list 文件所在的根目录（必需）
            physical_data_root: 实际数据文件所在的根目录（可选）
                               如果为 None，则使用 dataroot
                               如果提供，则会在 train/test/val 文件夹中搜索文件
            resolution: 图像分辨率
            split: 'train', 'test', 或 'val'
            data_len: 数据集长度限制，-1 表示使用全部数据
        """
        self.res = resolution
        self.split = split
        self.dataroot = dataroot

        # 如果没有指定 physical_data_root，使用 dataroot（向后兼容）
        self.physical_data_root = physical_data_root if physical_data_root is not None else dataroot

        if physical_data_root is not None and physical_data_root != dataroot:
            print(f"[{self.split} 数据集] 将从 '{self.physical_data_root}' 加载数据（支持跨 split 搜索）")
        else:
            print(f"[{self.split} 数据集] 将从 '{self.dataroot}' 加载数据（传统模式）")

        self.list_path = os.path.join(self.dataroot, 'list', self.split + '.txt')
        self.img_name_list = load_img_name_list(self.list_path)
        self.dataset_len = len(self.img_name_list)
        
        # 添加这行
        self.landslide_types = self._load_landslide_labels()
    
        if self.dataset_len == 0:
            raise IOError(f"在 {self.list_path} 中未找到任何有效数据。")

        self.data_len = self.dataset_len if data_len <= 0 else min(data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def get_paths(self, a_path):
        """
        新逻辑：直接使用 a_path,并从中推断其他路径。
        适配新的目录结构：train/Region/pre_event/png/file.png

        关键修复：由于 list 文件中的路径可能与实际文件位置不符
        （例如 test.txt 里写 test/Region/... 但文件实际在 train/Region/...）
        因此需要在 train/test/val 三个文件夹中搜索文件
        """
        try:
            path_parts = a_path.split(os.sep)
            event_index = -1
            if 'pre_event' in path_parts:
                event_index = path_parts.index('pre_event')
            elif 'post_event' in path_parts:
                event_index = path_parts.index('post_event')

            if event_index == -1:
                 raise ValueError(f"路径中未找到 'pre_event' 或 'post_event' 文件夹: {a_path}")

            # 提取 region 和 filename
            region_name = path_parts[event_index - 1]
            patch_base_name = path_parts[-1]
            patch_stem = os.path.splitext(patch_base_name)[0]

            # 关键修改：在 train/test/val 三个文件夹中搜索文件
            actual_split = None
            for candidate_split in ['train', 'test', 'val']:
                candidate_path = os.path.join(
                    self.physical_data_root,
                    candidate_split,
                    region_name,
                    "pre_event",
                    "png",
                    patch_base_name
                )
                if os.path.exists(candidate_path):
                    actual_split = candidate_split
                    break

            # 如果找不到文件，使用 list 文件中指定的 split（即使文件不存在）
            if actual_split is None:
                if event_index >= 2:
                    actual_split = path_parts[event_index - 2]
                else:
                    actual_split = self.split
                warnings.warn(f"无法在 train/test/val 中找到文件 {patch_base_name}，region={region_name}，使用默认 split={actual_split}")

            # 使用找到的 actual_split 构建所有路径
            base_path = os.path.join(self.physical_data_root, actual_split, region_name)
            
            # === 修复：添加 png 子目录 ===
            paths = {
                'A': os.path.join(base_path, "pre_event", "png", patch_base_name),
                'B': os.path.join(base_path, "post_event", "png", patch_base_name),
                'L': os.path.join(base_path, "mask", "png", patch_base_name)
            }
            
            # ✅ 修复：移除重复的self.split
            dem_file = f"{patch_stem}.tif"
            slope_file = f"{patch_stem}.tif"
            paths['dem'] = os.path.join(base_path, "dem", "tiff", dem_file)
            paths['slope'] = os.path.join(base_path, "slope", "tiff", slope_file)
                
            return paths
        except Exception as e:
            print(f"在 get_paths 中为路径 '{a_path}' 构建路径时发生错误: {e}")
            return {'A': None}
            
    def _load_landslide_labels(self):
        """加载滑坡类型标签"""
        import pandas as pd
        from collections import Counter
        
        label_file = os.path.join(self.dataroot, f'landslide_types_{self.split}.csv')
        
        if os.path.exists(label_file):
            df = pd.read_csv(label_file)
            label_dict = dict(zip(df['image_name'], df['landslide_type']))
            
            print(f"[{self.split}集] 加载了 {len(label_dict)} 个滑坡类型标签")
            type_counts = Counter(df['landslide_type'])
            print(f"  类型分布: {dict(type_counts)}")
            
            return label_dict
        else:
            print(f"警告: 未找到标签文件 {label_file}")
            return {}


    def __getitem__(self, index):
        unique_a_path = self.img_name_list[index % self.dataset_len]
        paths = self.get_paths(unique_a_path)

        # 检查路径构建是否成功
        if paths.get('A') is None:
            print(f"错误: 无法为 ID '{unique_a_path}' 构建有效路径。跳过此样本。")
            return {
                'A': torch.zeros(3, self.res, self.res),  # ✓ 改回3通道
                'B': torch.zeros(3, self.res, self.res),  # ✓ 改回3通道
                'L': torch.zeros(self.res, self.res, dtype=torch.long),
                'physical': torch.zeros(2, self.res, self.res),  # ✓ 新增：物理数据字段
                'Index': index,
                'ID': "PATH_ERROR"
            }

        try:
            img_A_pil = Image.open(paths['A']).convert("RGB")
            img_B_pil = Image.open(paths['B']).convert("RGB")
            img_L_pil = Image.open(paths['L'])
        except FileNotFoundError as e:
            print(f"错误: 找不到文件 {e.filename}")
            return {
                'A': torch.zeros(3, self.res, self.res),
                'B': torch.zeros(3, self.res, self.res),
                'L': torch.zeros(self.res, self.res, dtype=torch.long),
                'physical': torch.zeros(2, self.res, self.res),
                'Index': index,
                'ID': "FILE_NOT_FOUND"
            }

        img_A = Util.transform_augment_cd(img_A_pil, split=self.split, min_max=(-1, 1), res=self.res)
        img_B = Util.transform_augment_cd(img_B_pil, split=self.split, min_max=(-1, 1), res=self.res)
        img_L_tensor = Util.transform_augment_cd(img_L_pil, split=self.split, min_max=(0, 1), is_mask=True, res=self.res)
        
        if img_L_tensor.dim() == 3:
            img_L_tensor = img_L_tensor.squeeze(0)
        img_L = (img_L_tensor > 0.5).long()

        # === 修改：物理数据单独处理，不拼接到图像上 ===
        physical_data = None
        if self.physical_data_root and rasterio:
            physical_tensors = []
            for phys_type in ['dem', 'slope']:
                phys_path = paths.get(phys_type)
                # 添加调试输出（只打印第一个）
                if index == 0:
                    print(f"[DEBUG] 查找物理数据: {phys_path}")
                    print(f"[DEBUG] 文件存在: {os.path.exists(phys_path) if phys_path else False}")

                if phys_path and os.path.exists(phys_path):
                    try:
                        with rasterio.open(phys_path) as src:
                            data = torch.from_numpy(src.read(1)).float().unsqueeze(0)
                            min_val, max_val = data.min(), data.max()
                            if max_val > min_val:
                                data = (data - min_val) / (max_val - min_val)
                            if data.shape[1:] != (self.res, self.res):
                                data = torch.nn.functional.interpolate(
                                    data.unsqueeze(0),
                                    size=(self.res, self.res),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0)
                            physical_tensors.append(data)
                    except Exception as e:
                        warnings.warn(f"加载物理数据 {phys_path} 时出错: {e}。将使用零值填充。")
                        physical_tensors.append(torch.zeros(1, self.res, self.res))
                else:
                    warnings.warn(f"物理数据文件未找到: {phys_path}。将使用零值填充。")
                    physical_tensors.append(torch.zeros(1, self.res, self.res))

            if physical_tensors:
                physical_data = torch.cat(physical_tensors, dim=0)  # [2, H, W]
            else:
                physical_data = torch.zeros(2, self.res, self.res)
        else:
            physical_data = torch.zeros(2, self.res, self.res)

        # === 关键修改：不再拼接，而是单独返回 ===
        # 删除这些行：
        # img_A = torch.cat([img_A, physical_stack], dim=0)
        # img_B = torch.cat([img_B, physical_stack], dim=0)
        
        # 添加滑坡类型
        filename = os.path.basename(unique_a_path)
        if filename not in self.landslide_types:
            raise ValueError(f"文件 {filename} 缺少滑坡类型标注")
        landslide_type = self.landslide_types.get(filename, 'compound')

        return {
            'A': img_A,              # [3, H, W] - RGB only
            'B': img_B,              # [3, H, W] - RGB only
            'L': img_L,              # [H, W] - Label
            'physical': physical_data,  # [2, H, W] - DEM + slope
            'Index': index,
            'ID': unique_a_path
        }
