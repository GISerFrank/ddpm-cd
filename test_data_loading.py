#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据加载修复：验证能在 train/test/val 三个文件夹中搜索文件
"""

import sys
import os
import tempfile
import shutil

def search_file_in_splits(physical_data_root, a_path):
    """
    模拟修改后的 get_paths 逻辑：
    从 list 路径中提取 region 和 filename，
    然后在 train/test/val 三个文件夹中搜索文件
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

    return actual_split, region_name, patch_base_name


print("=" * 70)
print("数据加载修复测试 - 文件搜索逻辑")
print("=" * 70)
print()

# 创建临时测试目录结构
temp_dir = tempfile.mkdtemp()
try:
    # 设置测试场景：
    # test.txt 里写 test/Santa Catarina_Brazil/pre_event/png/file.png
    # 但文件实际在 train/Santa Catarina_Brazil/pre_event/png/file.png

    region = "Santa Catarina_Brazil"
    filename = "Santa_Catarina_Brazil_patch_612_3978.png"

    # 创建目录结构
    actual_file_path = os.path.join(temp_dir, "train", region, "pre_event", "png")
    os.makedirs(actual_file_path, exist_ok=True)

    # 创建测试文件
    test_file = os.path.join(actual_file_path, filename)
    with open(test_file, 'w') as f:
        f.write("test data")

    print(f"测试场景：")
    print(f"  实际文件位置：train/{region}/pre_event/png/{filename}")
    print()

    # 测试用例
    test_cases = [
        # (list文件路径, 期望找到的split)
        (f"test/{region}/pre_event/png/{filename}", "train"),  # list说test，实际在train
        (f"val/{region}/pre_event/png/{filename}", "train"),   # list说val，实际在train
        (f"train/{region}/pre_event/png/{filename}", "train"), # list说train，实际在train
    ]

    all_passed = True
    for list_path, expected_split in test_cases:
        try:
            found_split, found_region, found_filename = search_file_in_splits(temp_dir, list_path)

            if found_split == expected_split:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
                all_passed = False

            print(f"{status}")
            print(f"  list 文件路径: {list_path}")
            print(f"  找到的 split: {found_split} (期望: {expected_split})")
            print(f"  找到的 region: {found_region}")
            print(f"  找到的 filename: {found_filename}")
            print()

        except Exception as e:
            print(f"✗ FAIL - 异常")
            print(f"  list 文件路径: {list_path}")
            print(f"  错误: {e}")
            print()
            all_passed = False

    # 测试找不到文件的情况
    print("测试找不到文件的情况：")
    missing_file_path = f"test/{region}/pre_event/png/missing_file.png"
    found_split, found_region, found_filename = search_file_in_splits(temp_dir, missing_file_path)
    if found_split is None:
        print("✓ PASS - 正确返回 None")
    else:
        print(f"✗ FAIL - 应该返回 None，但返回了 {found_split}")
        all_passed = False
    print()

    print("=" * 70)
    if all_passed:
        print("✓ 所有测试通过！")
        print()
        print("工作原理：")
        print("1. 从 list 文件路径中提取 region 和 filename")
        print("2. 在 train/test/val 三个文件夹中依次搜索该文件")
        print("3. 返回第一个找到的文件所在的 split")
        print()
        print("解决的问题：")
        print("- test.txt 里写 test/Region/file.png")
        print("- 但文件实际在 train/Region/file.png")
        print("- 搜索逻辑能自动找到正确的文件位置")
    else:
        print("✗ 部分测试失败")
        sys.exit(1)

    print("=" * 70)

finally:
    # 清理临时目录
    shutil.rmtree(temp_dir)
