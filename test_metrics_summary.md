# Test 阶段 Metrics 输出总结

## 概述
这份代码在test阶段输出了**非常详细**的metrics，包括：
1. **全局指标** (整个测试集的统计)
2. **样本级别指标** (每个样本单独计算)
3. **可视化结果** (保存预测图像)

---

## 📊 输出的 Metrics 详情

### 1. 全局指标 (Global Metrics)
**位置**: 第 1047-1058 行

从 `change_detection.running_metric.get_scores()` 获取，包含：

#### 整体指标
- **accuracy** (acc): 准确率
- **mF1**: 平均 F1 分数
- **mIoU**: 平均 IoU

#### 变化类指标 (Class 1 - Change)
- **F1_1**: 变化类的 F1 分数
- **iou_1**: 变化类的 IoU
- **precision_1**: 变化类的精确率
- **recall_1**: 变化类的召回率

#### 无变化类指标 (Class 0 - No Change)
- **F1_0**: 无变化类的 F1 分数
- **iou_0**: 无变化类的 IoU

**控制台输出示例**:
```
📊 测试结果:
   准确率 (Accuracy): 0.95234
   平均F1 (mF1): 0.87654
   平均IoU (mIoU): 0.78123
   变化类F1 (F1_1): 0.85432
   变化类IoU (IoU_1): 0.74567
   变化类精确率 (Precision_1): 0.88901
   变化类召回率 (Recall_1): 0.82345
   无变化类F1 (F1_0): 0.89876
   无变化类IoU (IoU_0): 0.81679
```

---

### 2. 样本级别指标 (Per-Sample Metrics)
**位置**: 第 855-890 行

为**每个测试样本**单独计算以下指标：

- **sample_name**: 样本名称
- **accuracy**: 样本准确率
- **precision**: 样本精确率
- **recall**: 样本召回率
- **f1**: 样本 F1 分数
- **iou**: 样本 IoU
- **tp**: True Positive 像素数
- **tn**: True Negative 像素数
- **fp**: False Positive 像素数
- **fn**: False Negative 像素数

**计算方式**:
```python
# 混淆矩阵
tp = ((pred == 1) & (label == 1)).sum()
tn = ((pred == 0) & (label == 0)).sum()
fp = ((pred == 1) & (label == 0)).sum()
fn = ((pred == 0) & (label == 1)).sum()

# 指标
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
iou = tp / (tp + fp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)
```

---

## 📁 保存位置

### 1. WandB 记录
**位置**: 第 892-902 行 (每个样本) + 第 1061-1099 行 (全局)

#### Per-Sample 记录 (每个样本)
```python
wandb.log({
    "test/sample_name": img_name,
    "test/sample_accuracy": accuracy,
    "test/sample_precision": precision,
    "test/sample_recall": recall,
    "test/sample_f1": f1,
    "test/sample_iou": iou,
    "test/sample_step": current_step,
})
```

#### 全局最终结果
```python
wandb.log({
    "test_final/accuracy": test_metrics['acc'],
    "test_final/mF1": test_metrics['mf1'],
    "test_final/mIoU": test_metrics['miou'],
    "test_final/F1_change": test_metrics['F1_1'],
    "test_final/IoU_change": test_metrics['iou_1'],
    "test_final/precision_change": test_metrics['precision_1'],
    "test_final/recall_change": test_metrics['recall_1'],
    "test_final/F1_no_change": test_metrics['F1_0'],
    "test_final/IoU_no_change": test_metrics['iou_0'],
    "test_final/total_samples": len(test_loader),
})
```

#### WandB Summary
```python
wandb.run.summary["test_accuracy"] = test_metrics['acc']
wandb.run.summary["test_mF1"] = test_metrics['mf1']
wandb.run.summary["test_mIoU"] = test_metrics['miou']
wandb.run.summary["test_total_samples"] = len(test_loader)
```

#### 样本指标分布直方图
```python
wandb.log({
    "test_distribution/f1_histogram": wandb.Histogram(f1_scores),
    "test_distribution/iou_histogram": wandb.Histogram(iou_scores),
})
```

#### 样本指标表格
```python
table = wandb.Table(
    columns=list(sample_metrics[0].keys()),
    data=[list(m.values()) for m in sample_metrics]
)
wandb.log({"test_results/sample_metrics": table})
```

---

### 2. 本地 JSON 文件
**位置**: 第 1101-1112 行
**文件路径**: `results/test/test_results.json`

```json
{
  "global_metrics": {
    "acc": 0.95234,
    "mf1": 0.87654,
    "miou": 0.78123,
    "F1_1": 0.85432,
    "iou_1": 0.74567,
    "precision_1": 0.88901,
    "recall_1": 0.82345,
    "F1_0": 0.89876,
    "iou_0": 0.81679
  },
  "sample_metrics": [
    {
      "sample_name": "sample_001",
      "accuracy": 0.96,
      "precision": 0.89,
      "recall": 0.85,
      "f1": 0.87,
      "iou": 0.77,
      "tp": 1234,
      "tn": 5678,
      "fp": 123,
      "fn": 234
    },
    ...
  ],
  "config": {
    "model": "v8",
    "dataset": "GVLM-CD"
  }
}
```

---

### 3. 预测图像
**位置**: 第 908-1030 行
**保存路径**: `results/test/{sample_name}_{key}.png`

保存的图像包括：
- 输入图像A (A)
- 输入图像B (B)
- 真实标签 (L)
- 预测结果 (change_prediction)
- 其他可视化结果

---

## 🎯 WandB 可视化组织

### 测试过程中
- `test/sample_*`: 每个样本的指标 (逐样本记录)
- `test_vis/*`: 每10个样本的可视化图像

### 测试结束后
- `test_final/*`: 全局最终指标
- `test_distribution/*`: F1和IoU的分布直方图
- `test_results/sample_metrics`: 所有样本的详细指标表格

### WandB Summary
- `test_accuracy`: 全局准确率
- `test_mF1`: 全局平均F1
- `test_mIoU`: 全局平均IoU
- `test_total_samples`: 测试样本总数

---

## 📈 指标计算流程

```
1. 对每个测试样本:
   ├─ 前向传播 → 获得预测结果
   ├─ 计算样本级别指标 (TP/TN/FP/FN → Precision/Recall/F1/IoU)
   ├─ 记录到 sample_metrics 列表
   ├─ 记录到 WandB (test/sample_*)
   └─ 保存预测图像

2. 测试结束后:
   ├─ 从 running_metric 获取全局指标
   ├─ 打印控制台输出
   ├─ 记录到 WandB (test_final/*)
   ├─ 创建分布直方图
   ├─ 创建样本指标表格
   └─ 保存到本地 JSON 文件
```

---

## 💡 主要特点

### ✅ 优点
1. **详细完整**: 同时记录全局和样本级别的指标
2. **多种输出**: 控制台、WandB、本地JSON文件
3. **可视化丰富**: 
   - 预测图像
   - 指标分布直方图
   - 样本指标表格
4. **包含混淆矩阵**: 记录TP/TN/FP/FN，便于分析

### 📊 特别有用的功能
- **样本级别指标**: 可以找出表现最好/最差的样本
- **分布直方图**: 了解模型在不同样本上的表现分布
- **WandB表格**: 可交互式查看和排序所有样本的指标

---

## 🔍 与训练/验证的区别

| 阶段 | 全局指标 | 样本指标 | 混淆矩阵 | 图像保存 | 分布可视化 |
|------|---------|---------|---------|---------|-----------|
| 训练 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 验证 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 测试 | ✅ | ✅ | ✅ | ✅ | ✅ |

**测试阶段是最详细的**，包含了所有可能的metrics和可视化！

---

## 📝 使用建议

### 查看全局性能
```python
# 读取JSON文件
import json
with open('results/test/test_results.json', 'r') as f:
    results = json.load(f)
    
print(results['global_metrics'])
```

### 分析样本性能
```python
import pandas as pd

# 转换为DataFrame便于分析
df = pd.DataFrame(results['sample_metrics'])

# 找出表现最差的5个样本
worst_samples = df.nsmallest(5, 'f1')
print(worst_samples)

# 统计分析
print(df.describe())
```

### 在WandB中查看
1. 打开WandB项目
2. 查看 `test_final/*` 面板 → 全局指标
3. 查看 `test_distribution/*` → 性能分布
4. 查看 `test_results/sample_metrics` 表格 → 所有样本详情
5. 查看 `test_vis/*` → 可视化结果

---

## 🎯 总结

这份代码的test部分输出了**非常全面的metrics**：

**9个全局指标**:
1. accuracy
2. mF1
3. mIoU
4. F1_1
5. iou_1
6. precision_1
7. recall_1
8. F1_0
9. iou_0

**每个样本的10个指标**:
1. sample_name
2. accuracy
3. precision
4. recall
5. f1
6. iou
7. tp
8. tn
9. fp
10. fn

**额外输出**:
- 预测图像
- 指标分布直方图
- 样本指标表格
- JSON结果文件

这是一个**非常完善的测试评估系统**！✨
