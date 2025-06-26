from io import BytesIO
# import lmdb # 如果不使用lmdb，可以注释掉
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util # 假设这个模块存在且包含所需的 transform_augment_cd
import scipy
import scipy.io
import os.path
import re # 用于解析文件名

import numpy as np

# 更新文件夹名称以匹配新的裁剪脚本输出
IMG_FOLDER_NAME = "pre_event"  # 对应 img1.png
IMG_POST_FOLDER_NAME = 'post_event' # 对应 img2.png
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "mask"  # 对应 ref.png
# label_suffix = ".png" # 如果文件名已包含后缀，则不需要

### 该方法因使用了np.loadtxt而无法正确处理文件名中包含空格的情况，会将其视为分隔符而解析为两列。 ###
# def load_img_name_list(dataset_path):
#     try:
#         img_name_list = np.loadtxt(dataset_path, dtype=str, ndmin=1) # ndmin=1 确保即使只有一行也是数组
#     except Exception as e:
#         print(f"Error loading image name list from {dataset_path}: {e}")
#         return np.array([]) # 返回空数组以避免后续错误

#     if img_name_list.ndim == 2: # 如果loadtxt读取了多列（例如，如果文件名中有空格且未正确处理）
#         print(f"Warning: Image name list at {dataset_path} appears to have multiple columns. Using the first column.")
#         return img_name_list[:, 0]
#     return img_name_list

def load_img_name_list(dataset_path):
    """
    从文本文件中加载图像文件名列表，每行一个文件名。
    能够处理文件名中包含空格的情况。
    """
    expanded_path = os.path.expanduser(dataset_path)
    img_name_list = []
    try:
        with open(expanded_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip() #去除首尾空白字符
                if line: #确保行不为空
                    img_name_list.append(line)
    except FileNotFoundError:
        print(f"Error: List file not found at {expanded_path} (expanded from {dataset_path}).")
        return np.array([]) # 返回空numpy数组以便后续代码兼容
    except Exception as e:
        print(f"Error loading image name list from {expanded_path}: {e}")
        return np.array([])
    
    if not img_name_list: # 如果列表为空
        print(f"Warning: No image names loaded from {expanded_path}.")
        return np.array([])
        
    return np.array(img_name_list) # 返回numpy数组以便后续代码兼容

# 修改路径构建函数以适应新的 unique_patch_id 格式
# 这些函数现在接收 unique_patch_id，并从中解析出 region 和 base_filename
def parse_unique_patch_id(unique_patch_id):
    """
    解析 unique_patch_id (例如 'RegionName_patch_y_x.png') 
    返回 (region_name, patch_base_name)
    假设 patch_base_name 的格式总是 'patch_数字_数字.png'
    """
    match = re.match(r"^(.*?)_(patch_\d+_\d+\.png)$", unique_patch_id)
    if match:
        region_name = match.group(1)
        patch_base_name = match.group(2)
        return region_name, patch_base_name
    else:
        # 如果不匹配，可能意味着文件名格式不符合预期，或者它是列表中的最后一行（可能是空行）
        # 或者，如果文件名本身不包含 "patch_" 前缀（例如，如果原始文件名被直接使用）
        # 这是一个更通用的分割，假设最后一个下划线之前是区域，之后是文件名
        parts = unique_patch_id.split('_')
        if len(parts) > 1 and parts[-1].endswith(".png") and parts[-2].isdigit() and parts[-3] == "patch":
             patch_base_name = "_".join(parts[-3:])
             region_name = "_".join(parts[:-3])
             return region_name, patch_base_name
        print(f"Warning: Could not parse unique_patch_id '{unique_patch_id}' into region and base filename using primary regex. Trying simpler split.")
        # 尝试一个更简单的分割：最后一个下划线前的所有内容作为区域名
        # 这可能不适用于所有区域名称（如果区域名称本身包含下划线且后面不是patch_y_x.png格式）
        # 但对于 'RegionName_patch_y_x.png' 应该有效
        last_underscore_idx = unique_patch_id.rfind('_patch_')
        if last_underscore_idx != -1:
            region_name = unique_patch_id[:last_underscore_idx]
            patch_base_name = unique_patch_id[last_underscore_idx+1:]
            return region_name, patch_base_name
        else: # 如果还是不行，就返回None，让调用者处理
            print(f"Error: Could not reliably parse unique_patch_id: {unique_patch_id}")
            return None, None


def get_img_post_path(root_dir, unique_patch_id):
    region_name, patch_base_name = parse_unique_patch_id(unique_patch_id)
    if region_name is None or patch_base_name is None:
        return None # 或者抛出异常
    return os.path.join(root_dir, region_name, IMG_POST_FOLDER_NAME, patch_base_name)


def get_img_path(root_dir, unique_patch_id):
    region_name, patch_base_name = parse_unique_patch_id(unique_patch_id)
    if region_name is None or patch_base_name is None:
        return None
    return os.path.join(root_dir, region_name, IMG_FOLDER_NAME, patch_base_name)


def get_label_path(root_dir, unique_patch_id):
    region_name, patch_base_name = parse_unique_patch_id(unique_patch_id)
    if region_name is None or patch_base_name is None:
        return None
    return os.path.join(root_dir, region_name, ANNOT_FOLDER_NAME, patch_base_name)

class CDDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        
        self.res = resolution
        # self.data_len = data_len # data_len 的逻辑在后面处理
        self.split = split

        self.root_dir = dataroot # 例如 "../GVLM_CD_cropped_with_metadata"
        # self.split = split  #train | val | test # 重复赋值
        
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        
        self.img_name_list = load_img_name_list(self.list_path)
        if self.img_name_list.size == 0 and os.path.exists(self.list_path):
            print(f"Warning: Loaded empty image name list from {self.list_path}, though the file exists.")
        elif not os.path.exists(self.list_path):
             print(f"Error: List file not found at {self.list_path}")
             self.img_name_list = np.array([]) # 确保是空数组

        self.dataset_len = len(self.img_name_list)
        print(f"Dataset split: {self.split}, Number of unique patch IDs found: {self.dataset_len} from {self.list_path}")


        if data_len <= 0: # 使用传入的 data_len
            self.data_len = self.dataset_len
        else:
            self.data_len = min(data_len, self.dataset_len) # 使用传入的 data_len

        if self.dataset_len == 0 :
            print(f"Warning: Dataset length is 0 for split '{self.split}'. DataLoader will be empty.")
            self.data_len = 0


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.dataset_len == 0:
            raise IndexError(f"Attempting to get item from an empty dataset (split: {self.split})")

        # 使用 unique_patch_id 从列表中获取
        current_unique_patch_id = self.img_name_list[index % self.dataset_len] # 使用 self.dataset_len 进行模运算以循环数据集

        A_path = get_img_path(self.root_dir, current_unique_patch_id)
        B_path = get_img_post_path(self.root_dir, current_unique_patch_id)
        L_path = get_label_path(self.root_dir, current_unique_patch_id)

        if A_path is None or B_path is None or L_path is None:
            # 这种情况理论上不应该发生，除非 parse_unique_patch_id 失败且列表文件中有无效条目
            print(f"Error: Could not construct paths for unique_patch_id: {current_unique_patch_id}")
            # 返回一个虚拟的错误样本或者抛出异常，取决于您的错误处理策略
            # 为了简单起见，这里我们可能需要跳过这个样本或返回None，但DataLoader通常期望有效数据
            # 更好的做法是在初始化时就过滤掉无效的img_name_list条目
            # 这里我们假设路径总是能构建成功，如果不能，说明列表文件或解析逻辑有问题
            # 为了让代码能跑通，我们返回一个占位符，但实际应用中需要更鲁棒的处理
            dummy_tensor = torch.zeros((3, self.res, self.res)) 
            dummy_label = torch.zeros((self.res, self.res), dtype=torch.long)
            return {'A': dummy_tensor, 'B': dummy_tensor, 'L': dummy_label, 'Index': index, 'ID': "ERROR_PARSING_ID"}


        try:
            img_A   = Image.open(A_path).convert("RGB")
            img_B   = Image.open(B_path).convert("RGB")
            img_lbl_pil = Image.open(L_path) # 保持原始模式，让transform_augment_cd处理
        except FileNotFoundError:
            print(f"Error: Image file not found for ID {current_unique_patch_id}. Paths:")
            print(f"  A: {A_path}")
            print(f"  B: {B_path}")
            print(f"  L: {L_path}")
            # 返回一个虚拟的错误样本
            dummy_tensor = torch.zeros((3, self.res, self.res))
            dummy_label = torch.zeros((self.res, self.res), dtype=torch.long)
            return {'A': dummy_tensor, 'B': dummy_tensor, 'L': dummy_label, 'Index': index, 'ID': "ERROR_FILE_NOT_FOUND"}


        # 假设 Util.transform_augment_cd 能够处理PIL Image对象
        # 并进行必要的转换，例如将标签转换为单通道的tensor
        img_A_tensor   = Util.transform_augment_cd(img_A, split=self.split, min_max=(-1, 1), res=self.res)
        img_B_tensor   = Util.transform_augment_cd(img_B, split=self.split, min_max=(-1, 1), res=self.res)
        
        # 对于标签，通常期望是单通道的，并且像素值在0到C-1之间（C是类别数）
        # 或者是一个0-1的二值掩膜。transform_augment_cd应该处理好这一点。
        # DDPM-CD的标签似乎是 (0,1) 范围的浮点数，然后取第一个通道
        img_lbl_tensor = Util.transform_augment_cd(img_lbl_pil, split=self.split, min_max=(0, 1), is_mask=True, res=self.res) # 假设有个is_mask参数
        
        if img_lbl_tensor.dim() == 3 and img_lbl_tensor.shape[0] > 1: # 如果是多通道，例如RGB
            # 假设标签信息在第一个通道，或者需要转换为灰度
            # 如果原始标签就是单通道灰度图，convert('L')后，transform_augment_cd应正确处理
            img_lbl_tensor = img_lbl_tensor[0] # 取第一个通道
        elif img_lbl_tensor.dim() == 2: # 已经是单通道 (H, W)
            pass # 不需要操作
        elif img_lbl_tensor.dim() == 3 and img_lbl_tensor.shape[0] == 1: # (1, H, W)
            img_lbl_tensor = img_lbl_tensor.squeeze(0) # 移除通道维度变为 (H,W)
        else:
            print(f"Warning: Label tensor for {current_unique_patch_id} has unexpected shape: {img_lbl_tensor.shape}")
            # 可能需要根据实际情况调整

        # 确保标签是期望的数据类型，例如 torch.long 用于交叉熵损失，或 torch.float 用于二元交叉熵
        # 如果是二值变化检测，通常是torch.float，像素值为0或1
        
        return {'A': img_A_tensor, 'B': img_B_tensor, 'L': img_lbl_tensor.float(), 'Index': index, 'ID': current_unique_patch_id}