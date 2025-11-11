# 配置文件使用指南

## 概述

本项目支持两种数据加载模式：
1. **传统模式** - 使用原始的 `CDDataset_GVLM_CD.py`
2. **物理数据模式** - 使用新的 `CDDataset_GVLM_CD_physical.py`，支持跨 split 文件搜索

## 模式 1: 传统模式（向后兼容）

如果你的数据集结构和 list 文件路径完全一致，使用原始配置：

```json
{
    "datasets": {
        "train": {
            "name": "GVLM-CD-256",
            "dataroot": "/path/to/dataset/",
            "resolution": 256,
            "batch_size": 1,
            "num_workers": 8,
            "use_shuffle": true,
            "data_len": -1
        }
    }
}
```

**特点：**
- 不需要 `use_physical_data` 字段（或设置为 `false`）
- 只需要 `dataroot` 参数
- list 文件路径必须与实际文件位置一致

## 模式 2: 物理数据模式（推荐）

如果你的 list 文件路径与实际文件位置不一致，使用新的配置：

```json
{
    "datasets": {
        "train": {
            "name": "GVLM-CD-Physical-256",
            "use_physical_data": true,
            "dataroot": "/path/to/dataroot/with/list/files",
            "physical_data_root": "/home/user/GVLM_CD_split_dataset_geotiff",
            "resolution": 256,
            "batch_size": 1,
            "num_workers": 8,
            "use_shuffle": true,
            "data_len": -1
        }
    }
}
```

**特点：**
- 设置 `"use_physical_data": true`
- `dataroot`: list 文件所在的目录（包含 `list/train.txt` 等）
- `physical_data_root`: 实际数据文件所在的目录
- 自动在 `train/test/val` 三个文件夹中搜索文件

**解决的问题：**
- list 文件写 `test/Region/file.png`，但文件实际在 `train/Region/file.png`
- 不需要物理移动文件
- RGB 图像和物理数据（DEM, Slope）保持一致性

## 目录结构示例

### 传统模式目录结构
```
dataroot/
├── list/
│   ├── train.txt
│   ├── test.txt
│   └── val.txt
├── train/
│   └── RegionA/
│       ├── pre_event/png/
│       ├── post_event/png/
│       └── mask/png/
├── test/
│   └── ...
└── val/
    └── ...
```

### 物理数据模式目录结构
```
# list 文件位置（dataroot）
/path/to/dataroot/
└── list/
    ├── train.txt  # 内容可能是: test/RegionA/pre_event/png/file.png
    ├── test.txt
    └── val.txt

# 实际数据位置（physical_data_root）
/home/user/GVLM_CD_split_dataset_geotiff/
├── train/
│   └── RegionA/
│       ├── pre_event/png/
│       ├── post_event/png/
│       ├── mask/png/
│       ├── dem/tiff/
│       └── slope/tiff/
├── test/
│   └── ...
└── val/
    └── ...
```

## 示例配置文件

完整示例请查看：
- 传统模式: `config/gvlm_cd.json`
- 物理数据模式: `config/gvlm_cd_physical.json`

## 使用方法

```bash
# 使用传统模式
python ddpm_cd.py -c config/gvlm_cd.json

# 使用物理数据模式
python ddpm_cd.py -c config/gvlm_cd_physical.json
```

## 注意事项

1. **`use_physical_data` 字段**
   - 如果为 `true`，使用 `CDDataset_GVLM_CD_physical.py`
   - 如果为 `false` 或不存在，使用 `CDDataset_GVLM_CD.py`

2. **`physical_data_root` 字段**
   - 只在 `use_physical_data=true` 时有效
   - 如果不提供，默认使用 `dataroot`

3. **向后兼容**
   - 旧的配置文件无需修改即可使用
   - 新功能通过可选参数实现

4. **物理数据**
   - DEM 和 Slope 数据需要在 `dem/tiff/` 和 `slope/tiff/` 目录下
   - 需要安装 `rasterio` 库：`pip install rasterio`
