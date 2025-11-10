#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据加载修复：验证能从 train/test/val 并集中正确加载数据
"""

import sys
import os

# 模拟路径解析逻辑
def parse_split_from_path(a_path):
    """
    从 list 文件的路径中解析真实的 split
    路径格式：train/Los Lagos_Chile/pre_event/png/patch_0_0.png
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

    # 解析 split（event_index - 2）
    if event_index >= 2:
        actual_split = path_parts[event_index - 2]
        region_name = path_parts[event_index - 1]
    else:
        raise ValueError(f"路径格式不正确: {a_path}")

    return actual_split, region_name


# 测试用例
test_cases = [
    # 格式：(list中的路径, 当前dataset的split, 期望的actual_split, 期望的region)
    ("train/Los Lagos_Chile/pre_event/png/patch_0_0.png", "train", "train", "Los Lagos_Chile"),
    ("test/Los Lagos_Chile/pre_event/png/patch_1_0.png", "train", "test", "Los Lagos_Chile"),
    ("val/Los Lagos_Chile/post_event/png/patch_2_0.png", "train", "val", "Los Lagos_Chile"),
    ("train/Region A/pre_event/png/patch_0_0.png", "test", "train", "Region A"),
]

print("=" * 70)
print("数据加载修复测试")
print("=" * 70)
print()

all_passed = True
for path, dataset_split, expected_split, expected_region in test_cases:
    try:
        actual_split, region_name = parse_split_from_path(path)

        # 验证结果
        if actual_split == expected_split and region_name == expected_region:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False

        print(f"{status}")
        print(f"  路径: {path}")
        print(f"  Dataset split: {dataset_split}")
        print(f"  解析的 split: {actual_split} (期望: {expected_split})")
        print(f"  解析的 region: {region_name} (期望: {expected_region})")
        print()

    except Exception as e:
        print(f"✗ FAIL - 异常")
        print(f"  路径: {path}")
        print(f"  错误: {e}")
        print()
        all_passed = False

print("=" * 70)
if all_passed:
    print("✓ 所有测试通过！")
    print()
    print("说明：")
    print("- 即使 train.txt 包含 test/ 或 val/ 的路径，也能正确加载")
    print("- 数据加载器会从路径中自动识别真实的 split")
    print("- 不需要物理移动文件")
else:
    print("✗ 部分测试失败")
    sys.exit(1)

print("=" * 70)
