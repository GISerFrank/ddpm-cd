#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试物理数据加载：验证 DEM 和 slope 也能从正确的 split 加载
"""

import sys
import os
import tempfile
import shutil

def get_paths_with_physical(physical_data_root, a_path):
    """
    完整模拟 get_paths 逻辑，包括物理数据
    """
    path_parts = a_path.split(os.sep)

    # 找到 event 索引
    event_index = -1
    if 'pre_event' in path_parts:
        event_index = path_parts.index('pre_event')
    elif 'post_event' in path_parts:
        event_index = path_parts.index('post_event')

    if event_index == -1:
        raise ValueError(f"路径中未找到 'pre_event' 或 'post_event': {a_path}")

    # 提取 region 和 filename
    region_name = path_parts[event_index - 1]
    patch_base_name = path_parts[-1]
    patch_stem = os.path.splitext(patch_base_name)[0]

    # 在 train/test/val 三个文件夹中搜索
    actual_split = None
    for candidate_split in ['train', 'test', 'val']:
        candidate_path = os.path.join(
            physical_data_root,
            candidate_split,
            region_name,
            "pre_event",
            "png",
            patch_base_name
        )
        if os.path.exists(candidate_path):
            actual_split = candidate_split
            break

    if actual_split is None:
        return None

    # 使用找到的 actual_split 构建所有路径
    base_path = os.path.join(physical_data_root, actual_split, region_name)

    paths = {
        'A': os.path.join(base_path, "pre_event", "png", patch_base_name),
        'B': os.path.join(base_path, "post_event", "png", patch_base_name),
        'L': os.path.join(base_path, "mask", "png", patch_base_name),
        'dem': os.path.join(base_path, "dem", "tiff", f"{patch_stem}.tif"),
        'slope': os.path.join(base_path, "slope", "tiff", f"{patch_stem}.tif")
    }

    return paths, actual_split


print("=" * 70)
print("物理数据加载测试")
print("=" * 70)
print()

# 创建临时测试目录结构
temp_dir = tempfile.mkdtemp()
try:
    region = "Santa Catarina_Brazil"
    filename = "Santa_Catarina_Brazil_patch_612_3978.png"
    stem = "Santa_Catarina_Brazil_patch_612_3978"

    # 创建完整的目录结构（包括物理数据）在 train 文件夹下
    for subdir in ["pre_event/png", "post_event/png", "mask/png", "dem/tiff", "slope/tiff"]:
        dir_path = os.path.join(temp_dir, "train", region, subdir)
        os.makedirs(dir_path, exist_ok=True)

    # 创建测试文件
    test_files = {
        "pre_event/png": filename,
        "post_event/png": filename,
        "mask/png": filename,
        "dem/tiff": f"{stem}.tif",
        "slope/tiff": f"{stem}.tif"
    }

    for subdir, fname in test_files.items():
        file_path = os.path.join(temp_dir, "train", region, subdir, fname)
        with open(file_path, 'w') as f:
            f.write("test data")

    print("测试场景：")
    print(f"  实际文件位置：train/{region}/...")
    print(f"  包含：RGB 图像 + DEM + Slope")
    print()

    # 测试用例：list 文件说在 test，但实际在 train
    list_path = f"test/{region}/pre_event/png/{filename}"

    result = get_paths_with_physical(temp_dir, list_path)

    if result is None:
        print("✗ FAIL - 未找到文件")
        sys.exit(1)

    paths, found_split = result

    print(f"list 文件路径: {list_path}")
    print(f"找到的 split: {found_split}")
    print()
    print("构建的路径：")

    all_exist = True
    for key, path in paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {key:6s}: {path}")
        if not exists:
            all_exist = False

    print()
    print("=" * 70)

    if all_exist and found_split == "train":
        print("✓ 所有测试通过！")
        print()
        print("验证结果：")
        print(f"1. ✓ 正确识别文件在 '{found_split}' 文件夹")
        print("2. ✓ RGB 图像路径构建正确")
        print("3. ✓ DEM 物理数据路径构建正确")
        print("4. ✓ Slope 物理数据路径构建正确")
        print("5. ✓ 所有文件都存在")
        print()
        print("关键点：")
        print("- 即使 list 写 test/，物理数据也从 train/ 加载")
        print("- RGB 和物理数据使用同一个 actual_split")
        print("- 保证数据一致性")
    else:
        print("✗ 测试失败")
        if found_split != "train":
            print(f"  期望 split='train'，实际 split='{found_split}'")
        if not all_exist:
            print("  某些文件路径不存在")
        sys.exit(1)

    print("=" * 70)

finally:
    # 清理临时目录
    shutil.rmtree(temp_dir)
