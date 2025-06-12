import os
import torch
import torchvision
import random
import numpy as np
import torchvision.transforms as T

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


totensor = T.ToTensor() # 使用 T.ToTensor() 以清晰
hflip = T.RandomHorizontalFlip()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)

def get_paths_from_mat(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_mat_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
rcrop = torchvision.transforms.RandomCrop(size=256)
resize = torchvision.transforms.Resize(size=256)

# augmentations for images
def transform_augment(img, split='val', min_max=(0, 1), res=256):
    img = totensor(img)
    if split == 'train':
        if img.size(1) < res:
            img = resize(img)
        elif img.size(1) > res:
            img = rcrop(img)
        else:
            img=img
        img = hflip(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img

# def transform_augment_cd(img, split='val', min_max=(0, 1)):
#     img = totensor(img)
#     ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
#     return ret_img

def transform_augment_cd(img, split='val', min_max=(0, 1), res=256, is_mask=False): # 添加 res 和 is_mask 参数
    # 1. Resize a imagem (PIL Image)
    if img.size != (res, res):
        # 对于掩膜，使用最近邻插值以避免产生新的像素值
        # 对于普通图像，可以使用双线性或双三次插值
        interpolation_mode = Image.NEAREST if is_mask else Image.BICUBIC 
        img = img.resize((res, res), interpolation_mode)

    # 2. 数据增强 (只在训练时对非掩膜图像进行)
    if split == 'train' and not is_mask:
        # 这里可以添加您需要的增强操作，例如随机水平翻转
        # img = hflip(img) # hflip 是 torchvision.transforms.RandomHorizontalFlip，作用于PIL Image
        # 注意：如果在这里使用 torchvision的 hflip，它返回的是PIL Image。
        # 或者在转换为Tensor后使用 torch.fliplr。为简单起见，先只做resize和ToTensor。
        # 如果您的原始 hflip 是针对Tensor的，那么增强逻辑需要放在ToTensor之后。
        # 鉴于原始 util.py 中 hflip 是 torchvision.transforms，它作用于PIL Image。
        if random.random() < 0.5: # 示例：50%概率水平翻转
             img = T.functional.hflip(img)


    # 3. 转换为Tensor (将像素值从 [0, 255] 缩放到 [0.0, 1.0])
    img_tensor = totensor(img)

    # 4. 像素值范围调整
    ret_img = img_tensor * (min_max[1] - min_max[0]) + min_max[0]
    
    # 5. 对于掩膜，确保是单通道 (如果原始掩膜是L模式，ToTensor后是 [1, H, W])
    if is_mask:
        if ret_img.shape[0] != 1: # 如果不是单通道 (例如，有些掩膜可能被错误地加载为RGB)
            # 假设掩膜信息在第一个通道，或者需要进一步处理
            # 如果原始就是L或1模式，ToTensor后就是[1,H,W]
            # print(f"Warning: Mask tensor has shape {ret_img.shape}, expected single channel. Taking first channel or converting.")
            # 如果img是PIL，确保它是'L'或'1'模式再ToTensor
            pass # ToTensor对于 'L' 或 '1' 模式的PIL Image会产生 [1, H, W] 的Tensor
        # 通常在CDDataset的__getitem__中 .squeeze(0) 来得到 [H,W]

    return ret_img
