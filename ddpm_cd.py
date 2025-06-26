import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist  # 分布式训练
import torch.multiprocessing as mp  # 多进程
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式DataParallel
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器

import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from model.cd_modules.cd_head import cd_head 
from misc.print_diffuse_feats import print_feats
import time
from contextlib import contextmanager
import csv
import sys
import inspect
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用4个GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA调用，便于调试

# ==================== 优化版标签验证器 ====================
# 修改 LabelValidator 类，使其也能处理物理数据

class LabelValidator:
    """高效标签验证器 - 单例模式"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.is_normalized = None
            self.validation_done = False
            self.label_stats = {}
            self.physical_data_validated = False  # 新增：物理数据验证标志
            LabelValidator._initialized = True
    
    def validate_and_fix_labels(self, data, phase="train"):
        """
        高效标签验证 - 只在第一次详细检查，后续快速处理
        """
        if 'L' not in data:
            return False
        
        labels = data['L']
        
        # 快速通道：如果已经验证过，直接处理
        if self.validation_done:
            if self.is_normalized:
                data['L'] = (labels >= 0.5).long()
            else:
                # This 'fixed_labels' is local to this block and does not cause issues.
                fixed_labels = labels.clone()
                if 255 in torch.unique(labels):
                    fixed_labels[labels == 255] = 1
                data['L'] = torch.clamp(fixed_labels, 0, 1).long()
            
            # 新增：为新CD模型添加label字段
            data['label'] = data['L']
            return True
        
        # --- 第一次详细验证 (从您的原始函数中完整恢复) ---
        unique_vals = torch.unique(labels)
        min_val, max_val = labels.min().item(), labels.max().item()
        
        print(f"\n🔍 [{phase}] 标签验证（仅显示一次）:")
        print(f"   形状: {labels.shape}, 数据类型: {labels.dtype}")
        print(f"   值范围: [{min_val}, {max_val}]")
        print(f"   唯一值: {unique_vals.tolist()}")
        
        # 判断标签类型并进行修复
        self.is_normalized = (max_val <= 1.0 and min_val >= 0.0)
        
        if self.is_normalized:
            print("   🔧 检测到归一化标签，使用阈值二值化（阈值=0.5）")
            fixed_labels = (labels >= 0.5).long()
        else:
            print("   🔧 检测到标准标签，映射255→1")
            fixed_labels = labels.clone()
            if 255 in unique_vals:
                fixed_labels[labels == 255] = 1
            fixed_labels = torch.clamp(fixed_labels, 0, 1).long()
        
        # 验证修复结果并打印统计信息
        final_unique = torch.unique(fixed_labels)
        zero_count = (fixed_labels == 0).sum().item()
        one_count = (fixed_labels == 1).sum().item()
        total_pixels = fixed_labels.numel()
        
        if total_pixels > 0:
            print(f"   ✅ 修复完成: 唯一值{final_unique.tolist()}")
            print(f"   📊 像素分布: 无变化={100 * zero_count / total_pixels:.1f}%, 有变化={100 * one_count / total_pixels:.1f}%")
        print("   ✅ 标签验证设置完成，后续批次将快速处理\n")
        
        # 保存统计信息
        if total_pixels > 0:
            self.label_stats = {
                'zero_ratio': zero_count / total_pixels,
                'one_ratio': one_count / total_pixels,
                'is_normalized': self.is_normalized
            }
        
        # 将修复后的标签应用到数据字典中
        data['L'] = fixed_labels
        # 新增：同时设置label字段以适配新模型
        data['label'] = fixed_labels
        
        self.validation_done = True
        return True
    
    def validate_physical_data(self, data, phase="train"):
        """
        新增：验证物理数据
        """
        if 'physical_data' not in data:
            return False
        
        if not self.physical_data_validated:
            physical = data['physical_data']
            print(f"\n🔍 [{phase}] 物理数据验证（仅显示一次）:")
            print(f"   形状: {physical.shape}, 数据类型: {physical.dtype}")
            print(f"   通道数: {physical.shape[1]}")
            
            # 打印每个通道的信息
            channel_names = ['DEM', '坡度', '坡向', '地质类型', '植被覆盖']
            for i in range(min(physical.shape[1], len(channel_names))):
                channel_data = physical[:, i]
                print(f"   通道{i} ({channel_names[i]}): "
                      f"范围[{channel_data.min():.2f}, {channel_data.max():.2f}]")
            
            self.physical_data_validated = True
            print("   ✅ 物理数据验证完成\n")
        
        return True

# 全局标签验证器
label_validator = LabelValidator()

# ==================== 内存管理工具 ====================
@contextmanager
def memory_efficient_context():
    """内存管理上下文"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.step_times = []
        self.memory_usage = []
        self.start_time = time.time()
    
    def log_step(self, step_time, memory_mb=None):
        self.step_times.append(step_time)
        if memory_mb:
            self.memory_usage.append(memory_mb)
    
    def get_stats(self):
        if not self.step_times:
            return "无统计数据"
        
        avg_time = np.mean(self.step_times[-100:])  # 最近100步平均
        total_time = time.time() - self.start_time
        
        stats = f"平均步时: {avg_time:.2f}s, 总时间: {total_time/60:.1f}min"
        
        if self.memory_usage:
            avg_memory = np.mean(self.memory_usage[-10:])
            stats += f", 显存: {avg_memory:.1f}MB"
        
        return stats
    
# ==================== 兼容性辅助函数 ====================
def ensure_cd_model_compatibility(change_detection, opt):
    """
    确保CD模型具有所有必需的方法和属性
    """
    # 检查是否需要添加性能指标跟踪器
    if not hasattr(change_detection, 'running_metric'):
        from misc.metric_tools import ConfuseMatrixMeter
        change_detection.running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        print("✅ 添加了性能指标跟踪器")
    
    # 确保有log_dict
    if not hasattr(change_detection, 'log_dict'):
        change_detection.log_dict = OrderedDict()
    
    # 添加必要的方法
    if not hasattr(change_detection, '_clear_cache'):
        change_detection._clear_cache = lambda: change_detection.running_metric.clear() if hasattr(change_detection, 'running_metric') else None
    
    if not hasattr(change_detection, '_update_metric'):
        def _update_metric():
            if hasattr(change_detection, 'change_prediction') and hasattr(change_detection, 'label'):
                G_pred = change_detection.change_prediction.detach()
                G_pred = torch.argmax(G_pred, dim=1)
                current_score = change_detection.running_metric.update_cm(
                    pr=G_pred.cpu().numpy(), 
                    gt=change_detection.label.detach().cpu().numpy()
                )
                return current_score
            return 0.0
        change_detection._update_metric = _update_metric
    
    if not hasattr(change_detection, '_collect_running_batch_states'):
        def _collect_running_batch_states():
            change_detection.running_acc = change_detection._update_metric()
            change_detection.log_dict['running_acc'] = change_detection.running_acc.item() if hasattr(change_detection.running_acc, 'item') else change_detection.running_acc
        change_detection._collect_running_batch_states = _collect_running_batch_states
    
    if not hasattr(change_detection, '_collect_epoch_states'):
        def _collect_epoch_states():
            scores = change_detection.running_metric.get_scores()
            change_detection.epoch_acc = scores['mf1']
            change_detection.log_dict['epoch_acc'] = change_detection.epoch_acc.item() if hasattr(change_detection.epoch_acc, 'item') else change_detection.epoch_acc
            for k, v in scores.items():
                change_detection.log_dict[k] = v
        change_detection._collect_epoch_states = _collect_epoch_states
    
    if not hasattr(change_detection, '_update_lr_schedulers'):
        def _update_lr_schedulers():
            if hasattr(change_detection, 'schedulers'):
                for scheduler in change_detection.schedulers:
                    scheduler.step()
            elif hasattr(change_detection, 'exp_lr_scheduler_netCD'):
                change_detection.exp_lr_scheduler_netCD.step()
            elif hasattr(change_detection, 'optimizer'):
                # 如果没有调度器但有优化器，创建一个
                from misc.torchutils import get_scheduler
                change_detection.exp_lr_scheduler_netCD = get_scheduler(
                    optimizer=change_detection.optimizer, 
                    args=opt['train']
                )
                change_detection.exp_lr_scheduler_netCD.step()
        change_detection._update_lr_schedulers = _update_lr_schedulers
    
    # 确保save_network方法
    if not hasattr(change_detection, 'save_network'):
        if hasattr(change_detection, 'save'):
            change_detection.save_network = lambda epoch, is_best_model=False: change_detection.save(epoch, is_best_model)
        else:
            print("⚠️  警告：CD模型没有save或save_network方法")
    
    return change_detection

# ==================== 新增：单样本指标计算函数 ====================
def calculate_per_sample_metrics(pred_tensor, label_tensor):
    """
    计算单个样本的性能指标。
    输入是单个[H, W]的预测和标签张量（值为0或1）。
    """
    metrics = {}
    
    # 确保张量是布尔类型以便进行逻辑运算
    pred = pred_tensor.bool()
    label = label_tensor.bool()

    # 基本统计量
    tp = (pred & label).sum().item()
    fp = (pred & ~label).sum().item()
    fn = (~pred & label).sum().item()
    tn = (~pred & ~label).sum().item()
    
    # 防止除以零
    epsilon = 1e-6

    # --- 变化类别 (Class 1) ---
    precision_1 = tp / (tp + fp + epsilon)
    recall_1 = tp / (tp + fn + epsilon)
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + epsilon)
    iou_1 = tp / (tp + fp + fn + epsilon)
    
    # --- 无变化类别 (Class 0) ---
    precision_0 = tn / (tn + fn + epsilon)
    recall_0 = tn / (tn + fp + epsilon)
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + epsilon)
    iou_0 = tn / (tn + fp + epsilon)
    
    # --- 总体指标 ---
    oa = (tp + tn) / (tp + tn + fp + fn + epsilon)
    mf1 = (f1_1 + f1_0) / 2
    miou = (iou_1 + iou_0) / 2
    
    # 返回字典
    return {
        'OA': oa,
        'mF1': mf1,
        'mIoU': miou,
        'F1_change': f1_1,
        'IoU_change': iou_1,
        'Precision_change': precision_1,
        'Recall_change': recall_1,
        'F1_no_change': f1_0,
        'IoU_no_change': iou_0,
        'Precision_no_change': precision_0,
        'Recall_no_change': recall_0,
    }

# ==================== 优化版特征重排 ====================
def apply_feature_reordering_optimized(f_A, f_B, train_data, change_detection, opt, phase="train"):
    """
    内存优化的特征重排方案 - 适配新CD模型接口
    """
    try:
        feat_scales = opt['model_cd']['feat_scales']  # [2, 5, 8, 11, 14]
        cd_expected_order = sorted(feat_scales, reverse=True)  # [14, 11, 8, 5, 2]
        
        # 只在第一次显示信息
        if not hasattr(apply_feature_reordering_optimized, '_logged'):
            print("🎯 使用优化的特征重排方案")
            print("   保持原始多尺度配置的完整语义")
            for i, scale in enumerate(cd_expected_order):
                print(f"     Block{i}: 使用layer{scale}特征")
            apply_feature_reordering_optimized._logged = True
        
        # 高效重排：直接在原地修改
        reordered_f_A = []
        reordered_f_B = []
        
        for fa, fb in zip(f_A, f_B):
            if isinstance(fa, list) and len(fa) > max(feat_scales):
                timestep_A = [fa[scale] for scale in cd_expected_order]
                timestep_B = [fb[scale] for scale in cd_expected_order]
                reordered_f_A.append(timestep_A)
                reordered_f_B.append(timestep_B)
            else:
                raise ValueError(f"特征格式错误: 期望list长度>{max(feat_scales)}, 实际{type(fa)}")
        
        # 清理原始特征释放内存
        del f_A, f_B
        
        # 新CD模型接口适配
        if hasattr(change_detection, 'feed_data'):
            # 新模型使用单一feed_data接口
            change_detection.feed_data(train_data)
            # 保存特征供后续使用
            change_detection._temp_features_A = reordered_f_A
            change_detection._temp_features_B = reordered_f_B
        else:
            # 旧模型接口
            change_detection.feed_data(reordered_f_A, reordered_f_B, train_data)
        
        # 清理重排后的特征
        del reordered_f_A, reordered_f_B
        
        return True
        
    except Exception as e:
        print(f"❌ 特征重排失败: {e}")
        print("🔄 使用回退方案...")
        
        # 简化回退方案（同样适配新接口）
        target_layers = [12, 13, 14]
        corrected_f_A = []
        corrected_f_B = []
        
        for fa, fb in zip(f_A, f_B):
            timestep_A = [fa[i] for i in target_layers if i < len(fa)]
            timestep_B = [fb[i] for i in target_layers if i < len(fb)]
            corrected_f_A.append(timestep_A)
            corrected_f_B.append(timestep_B)
        
        del f_A, f_B
        
        if hasattr(change_detection, 'feed_data') and hasattr(change_detection, '_temp_features_A'):
            change_detection.feed_data(train_data)
            change_detection._temp_features_A = corrected_f_A
            change_detection._temp_features_B = corrected_f_B
        else:
            change_detection.feed_data(corrected_f_A, corrected_f_B, train_data)
        
        del corrected_f_A, corrected_f_B
        
        return False

# ==================== 训练优化设置 ====================
def setup_training_optimization(diffusion, change_detection):
    """设置训练优化"""
    print("🚀 设置训练优化...")
    
    # 启用CUDA优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 检查混合精度支持
    use_amp = False
    if torch.cuda.is_available():
        try:
            from torch.cuda.amp import autocast, GradScaler
            use_amp = True
            print("   ✅ 支持混合精度训练")
        except ImportError:
            print("   ⚠️  不支持混合精度训练")
    
    # 设置diffusion模型为eval模式（如果不需要训练）
    if hasattr(diffusion.netG, 'eval'):
        diffusion.netG.eval()
        print("   ✅ Diffusion模型设置为eval模式")
    
    # 检查多GPU设置
    if torch.cuda.device_count() > 1:
        print(f"   ✅ 检测到{torch.cuda.device_count()}个GPU")
        
        # 显示GPU状态
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"     GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    
    print("🚀 训练优化设置完成\n")
    
    return use_amp

# ==================== 批量处理优化 ====================
# def process_batch_efficiently(train_data, diffusion, change_detection, opt, phase="train"):
#     """高效的批量处理"""
#     with memory_efficient_context():
#         # 1. 快速标签验证
#         label_validator.validate_and_fix_labels(train_data, phase)
        
#         # 2. 特征提取
#         diffusion.feed_data(train_data)
        
#         # 3. 收集特征
#         f_A, f_B = [], []
#         for t in opt['model_cd']['t']:
#             fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t)
#             if opt['model_cd']['feat_type'] == "dec":
#                 f_A.append(fd_A_t)
#                 f_B.append(fd_B_t)
#                 del fe_A_t, fe_B_t  # 立即清理
#             else:
#                 f_A.append(fe_A_t)
#                 f_B.append(fe_B_t)
#                 del fd_A_t, fd_B_t  # 立即清理
        
#         # 4. 特征重排
#         apply_feature_reordering_optimized(f_A, f_B, train_data, change_detection, opt, phase)

def process_features_for_cd(change_detection, features_A, features_B, data, current_epoch=None, phase='train'):
    """
    统一处理特征并调用CD模型的相应方法
    """
    try:
        # 方案1：新模型接口（单一feed_data + 临时特征）
        if hasattr(change_detection, 'feed_data') and len(inspect.signature(change_detection.feed_data).parameters) == 2:
            # 新接口：feed_data只接受一个data参数
            change_detection.feed_data(data)
            
            if phase == 'train':
                # 训练阶段
                if hasattr(change_detection.optimize_parameters, '__code__'):
                    params = change_detection.optimize_parameters.__code__.co_varnames
                    if 'features_A' in params or len(params) > 1:
                        # 新接口：需要特征
                        change_detection.optimize_parameters(features_A, features_B, current_epoch=current_epoch)
                    else:
                        # 特征已经在feed_data中处理
                        change_detection._temp_features_A = features_A
                        change_detection._temp_features_B = features_B
                        change_detection.optimize_parameters()
                else:
                    change_detection._temp_features_A = features_A
                    change_detection._temp_features_B = features_B
                    change_detection.optimize_parameters()
            else:
                # 验证/测试阶段
                if hasattr(change_detection, 'test'):
                    if hasattr(change_detection.test, '__code__'):
                        params = change_detection.test.__code__.co_varnames
                        if 'features_A' in params or len(params) > 1:
                            change_detection.test(features_A, features_B)
                        else:
                            change_detection._temp_features_A = features_A
                            change_detection._temp_features_B = features_B
                            change_detection.test()
                    else:
                        change_detection.test()
        
        # 方案2：旧模型接口（feed_data接受三个参数）
        else:
            change_detection.feed_data(features_A, features_B, data)
            if phase == 'train':
                change_detection.optimize_parameters()
            else:
                change_detection.test()
        
        # 清理临时特征
        if hasattr(change_detection, '_temp_features_A'):
            del change_detection._temp_features_A
        if hasattr(change_detection, '_temp_features_B'):
            del change_detection._temp_features_B
            
    except Exception as e:
        print(f"❌ 特征处理错误 ({phase}): {e}")
        import traceback
        traceback.print_exc()
        raise

def process_batch_efficiently(train_data, diffusion, change_detection, opt, phase="train", current_epoch=None):
    """高效的批量处理 - 完整版"""
    with memory_efficient_context():
        try:
            # 1. 标签验证和适配
            label_validator.validate_and_fix_labels(train_data, phase)
            
            # 2. 物理数据验证（如果启用）
            if opt['datasets'][phase].get('load_physical_data', False):
                label_validator.validate_physical_data(train_data, phase)
            
            # 3. 设备一致性
            device = next(diffusion.netG.parameters()).device
            for key, value in train_data.items():
                if torch.is_tensor(value):
                    train_data[key] = value.to(device)
            
            # 4. Diffusion特征提取
            diffusion.feed_data(train_data)
            
            # 5. 收集多时间步特征
            f_A, f_B = [], []
            for t in opt['model_cd']['t']:
                fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t)
                if opt['model_cd']['feat_type'] == "dec":
                    f_A.append(fd_A_t)
                    f_B.append(fd_B_t)
                    del fe_A_t, fe_B_t
                else:
                    f_A.append(fe_A_t)
                    f_B.append(fe_B_t)
                    del fd_A_t, fd_B_t
            
            # 6. 特征重排
            feat_scales = opt['model_cd']['feat_scales']
            cd_expected_order = sorted(feat_scales, reverse=True)
            
            reordered_f_A = []
            reordered_f_B = []
            
            for fa, fb in zip(f_A, f_B):
                if isinstance(fa, list) and len(fa) > max(feat_scales):
                    timestep_A = [fa[scale] for scale in cd_expected_order]
                    timestep_B = [fb[scale] for scale in cd_expected_order]
                    reordered_f_A.append(timestep_A)
                    reordered_f_B.append(timestep_B)
            
            # 7. 调用CD模型
            process_features_for_cd(change_detection, reordered_f_A, reordered_f_B, 
                                  train_data, current_epoch, phase)
            
            # 8. 清理内存
            del f_A, f_B, reordered_f_A, reordered_f_B
            
        except Exception as e:
            print(f"❌ 批处理失败 ({phase}): {e}")
            import traceback
            traceback.print_exc()
            # 尝试恢复
            torch.cuda.empty_cache()
            raise


# ==================== 优化的日志管理 ====================
def optimized_logging(current_step, current_epoch, n_epoch, loader, change_detection, 
                     logger, opt, phase="train", performance_monitor=None):
    """优化的日志输出 - 增加物理损失信息"""
    # 动态调整日志频率
    if phase == "train":
        base_freq = opt['train'].get('train_print_freq', 10)
        log_freq = max(base_freq, len(loader) // 1000)  # 至少每5%显示一次
    else:
        log_freq = max(1, len(loader) // 500)  # 验证时每10%显示一次
    
    if current_step % log_freq == 0:
        try:
            logs = change_detection.get_current_log()
            
            # 基础信息
            progress = f"[{current_epoch}/{n_epoch-1}]"
            step_info = f"Step {current_step}/{len(loader)}"
            
            # 构建指标信息
            loss_info = f"Loss: {logs.get('l_cd', 0):.5f}"
            
            # 如果有物理损失的详细信息，也显示
            if 'physics_loss' in logs:
                loss_info += f" (CE: {logs.get('ce_loss', 0):.3f}, Phy: {logs.get('physics_loss', 0):.3f})"
            
            metrics = f"{loss_info} mF1: {logs.get('running_acc', 0):.5f}"
            
            # 性能信息
            perf_info = ""
            if performance_monitor:
                perf_info = f" | {performance_monitor.get_stats()}"
            
            message = f"{progress} {step_info} {metrics}{perf_info}"
            print(message)
            
        except Exception as e:
            print(f"日志输出错误: {e}")


# ==================== 错误处理装饰器 ====================
def safe_training_step(func):
    """安全训练步骤装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "device-side assert triggered" in str(e) or "CUDA" in str(e):
                print(f"⚠️  CUDA错误已自动处理: {str(e)[:100]}...")
                torch.cuda.empty_cache()
                return False
            else:
                raise
        except Exception as e:
            print(f"❌ 训练步骤错误: {e}")
            return False
    return wrapper

@safe_training_step
def execute_training_step(change_detection, current_epoch=None):
    """执行训练步骤 - 适配新接口"""
    # 检查是否有临时保存的特征
    if hasattr(change_detection, '_temp_features_A') and hasattr(change_detection, '_temp_features_B'):
        # 新接口：需要传入特征和epoch
        change_detection.optimize_parameters(
            change_detection._temp_features_A,
            change_detection._temp_features_B,
            current_epoch=current_epoch
        )
        # 清理临时特征
        del change_detection._temp_features_A
        del change_detection._temp_features_B
    else:
        # 旧接口：无参数
        change_detection.optimize_parameters()
    
    change_detection._collect_running_batch_states()
    return True

# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ddpm_cd.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # 新增：物理损失相关参数
    parser.add_argument('--use_physics_loss', action='store_true',
                        help='Enable physics-constrained loss')
    parser.add_argument('--physics_config', type=str, default=None,
                        help='Override physics loss config file')
    parser.add_argument('--no_progressive', action='store_true',
                        help='Disable progressive training for physics loss')

    # 解析配置
    args = parser.parse_args()
    opt = Logger.parse(args)

    # 如果命令行指定使用物理损失，覆盖配置文件
    if args.use_physics_loss:
        opt['model_cd']['loss_type'] = 'physics_constrained'
        print("🔧 命令行参数：启用物理约束损失")
    
    # 如果指定了物理损失配置文件，加载并合并
    if args.physics_config:
        import json
        with open(args.physics_config, 'r') as f:
            physics_config = json.load(f)
            if 'physics_loss' in physics_config:
                opt['model_cd']['physics_loss'] = physics_config['physics_loss']
                print(f"🔧 加载物理损失配置: {args.physics_config}")
    
    # 智能选择配置文件
    if args.config == 'config/ddpm_cd.json' and args.use_physics_loss:
        # 如果使用默认配置但指定了物理损失，尝试使用物理损失配置文件
        physics_config_path = 'config/gvlm_cd_physical.json'
        if os.path.exists(physics_config_path):
            args.config = physics_config_path
            print(f"🔧 自动切换到物理损失配置文件: {physics_config_path}")
        else:
            print(f"⚠️  物理损失配置文件不存在: {physics_config_path}")
            print("   将使用默认配置并动态添加物理损失设置")
    
    # 如果禁用渐进式训练
    if args.no_progressive and 'physics_loss' in opt['model_cd']:
        opt['model_cd']['physics_loss']['enable_progressive'] = False
        print("🔧 禁用渐进式训练")
    
    # 验证配置完整性
    if opt['model_cd'].get('loss_type') == 'physics_constrained':
        if 'physics_loss' not in opt['model_cd']:
            # 提供默认物理损失配置
            opt['model_cd']['physics_loss'] = {
                'alpha': 0,
                'beta': 0.05,
                'gamma': 0.03,
                'delta': 0,
                'enable_progressive': True,
                'warmup_epochs': 10,
                'use_progressive': False
            }
            print("⚠️  使用默认物理损失配置")
        
        # 验证数据集配置
        for phase in ['train', 'val']:
            if phase in opt['datasets']:
                if not opt['datasets'][phase].get('load_physical_data', False):
                    print(f"⚠️  警告：{phase}数据集未配置load_physical_data，物理损失可能无法正常工作")
                    opt['datasets'][phase]['load_physical_data'] = True

    opt = Logger.dict_to_nonedict(opt)

    # 设置日志
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    # 新增：打印关键配置摘要
    print("\n" + "="*50)
    print("🔧 关键配置摘要:")
    print("="*50)
    print(f"模型名称: {opt.get('name', 'Unknown')}")
    print(f"训练阶段: {opt['phase']}")
    print(f"损失类型: {opt['model_cd'].get('loss_type', 'ce')}")
    
    if opt['model_cd'].get('loss_type') == 'physics_constrained':
        physics_cfg = opt['model_cd'].get('physics_loss', {})
        print(f"\n物理损失配置:")
        print(f"  - 坡度约束 (α): {physics_cfg.get('alpha', 0)}")
        print(f"  - 空间连续性 (β): {physics_cfg.get('beta', 0)}")
        print(f"  - 尺寸约束 (γ): {physics_cfg.get('gamma', 0)}")
        print(f"  - 地质约束 (δ): {physics_cfg.get('delta', 0)}")
        print(f"  - 渐进式训练: {physics_cfg.get('enable_progressive', True)}")
        print(f"  - 预热轮次: {physics_cfg.get('warmup_epochs', 10)}")
    
    print(f"\n数据配置:")
    for phase in ['train', 'val', 'test']:
        if phase in opt['datasets']:
            ds_cfg = opt['datasets'][phase]
            print(f"  {phase}:")
            print(f"    - 批量大小: {ds_cfg.get('batch_size', 'N/A')}")
            print(f"    - 加载物理数据: {ds_cfg.get('load_physical_data', False)}")
            if ds_cfg.get('load_physical_data'):
                print(f"    - 物理数据路径: {ds_cfg.get('physical_data_path', 'Not specified')}")
    
    print(f"\n训练配置:")
    print(f"  - 总轮次: {opt['train']['n_epoch']}")
    print(f"  - 优化器: {opt['train']['optimizer']['type']}")
    print(f"  - 学习率: {opt['train']['optimizer']['lr']}")
    print(f"  - GPU设备: {opt['gpu_ids']}")
    print("="*50 + "\n")
    
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # 初始化WandbLogger
    if opt['enable_wandb']:
        import wandb
        print("Initializing wandblog.")
        
        # 根据是否使用物理损失更新项目名称
        if opt['model_cd'].get('loss_type') == 'physics_constrained':
            original_project = opt['wandb']['project']
            opt['wandb']['project'] = f"{original_project}-Physics"
            print(f"WandB项目名称更新为: {opt['wandb']['project']}")
        
        wandb_logger = WandbLogger(opt)
        
        # 记录配置信息
        wandb.config.update({
            'loss_type': opt['model_cd'].get('loss_type', 'ce'),
            'physics_loss': opt['model_cd'].get('physics_loss', {}),
            'load_physical_data': any(
                opt['datasets'][p].get('load_physical_data', False) 
                for p in opt['datasets']
            )
        })
        
        wandb.define_metric('epoch')
        wandb.define_metric('training/train_step')
        wandb.define_metric("training/*", step_metric="train_step")
        wandb.define_metric('validation/val_step')
        wandb.define_metric("validation/*", step_metric="val_step")
        
        # 物理损失相关指标
        if opt['model_cd'].get('loss_type') == 'physics_constrained':
            wandb.define_metric("physics/*", step_metric="train_step")
        
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None
        
    # 加载数据集
    print("🔄 加载数据集...")
    for phase, dataset_opt in opt['datasets'].items():
         # 检查是否需要加载物理数据
        if dataset_opt.get('load_physical_data', False):
            print(f"   📊 {phase}数据集将加载物理数据")
            physical_path = dataset_opt.get('physical_data_path', '')
            if not physical_path:
                print(f"   ⚠️  警告：{phase}数据集配置了load_physical_data但未指定physical_data_path")
        if phase == 'train' and args.phase != 'test':
            print("Creating [train] change-detection dataloader.")
            train_set = Data.create_cd_dataset(dataset_opt, phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

            # 验证第一个批次是否包含物理数据
            if dataset_opt.get('load_physical_data', False):
                try:
                    first_batch = next(iter(train_loader))
                    if 'physical_data' in first_batch:
                        print(f"   ✅ 成功加载物理数据，形状: {first_batch['physical_data'].shape}")
                    else:
                        print(f"   ⚠️  数据批次中未找到physical_data字段")
                except Exception as e:
                    print(f"   ❌ 验证物理数据失败: {e}")

        elif phase == 'val' and args.phase != 'test':
            print("Creating [val] change-detection dataloader.")
            val_set = Data.create_cd_dataset(dataset_opt, phase)
            val_loader = Data.create_cd_dataloader(val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)
        
        elif phase == 'test' and args.phase == 'test':
            print("Creating [test] change-detection dataloader.")
            test_set = Data.create_cd_dataset(dataset_opt, phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)
    
    logger.info('Initial Dataset Finished')

    # 加载模型
    print("🔄 加载扩散模型...")
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # 处理DataParallel
    if isinstance(diffusion.netG, nn.DataParallel):
        diffusion.netG = diffusion.netG.module
        print("已解包diffusion模型的DataParallel")

    # 多GPU设置
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        diffusion.netG = diffusion.netG.cuda()
        diffusion.netG = nn.DataParallel(diffusion.netG, device_ids=[0, 1, 2, 3])
        
        # 适度增加batch size
        for phase in opt['datasets']:
            if 'batch_size' in opt['datasets'][phase]:
                original_bs = opt['datasets'][phase]['batch_size']
                # 可以根据GPU数量调整
                # opt['datasets'][phase]['batch_size'] = original_bs * 2
                print(f"{phase} batch_size: {original_bs}")

    # 设置噪声调度
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    # 创建变化检测模型
    print("🔄 加载变化检测模型...")
    # change_detection = Model.create_CD_model(opt)
    # 确保使用新的CD模型类
    try:
        # 创建变化检测模型
        print("🔄 加载变化检测模型...")
        change_detection = Model.create_CD_model(opt)
        
        # 确保兼容性
        change_detection = ensure_cd_model_compatibility(change_detection, opt)
        
        # 验证模型
        print(f"✅ 成功创建CD模型: {change_detection.__class__.__name__}")
        
        # 检查物理损失支持
        if opt['model_cd'].get('loss_type') == 'physics_constrained':
            if hasattr(change_detection, 'criterion'):
                print(f"✅ 物理损失函数已配置: {change_detection.criterion.__class__.__name__}")
            else:
                print("⚠️  警告：模型可能不支持物理损失")
        
    except Exception as e:
        print(f"❌ CD模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 初始化性能指标跟踪器（如果新模型没有）
    if not hasattr(change_detection, 'running_metric'):
        print("🔧 为CD模型添加性能指标跟踪器...")
        from misc.metric_tools import ConfuseMatrixMeter
        change_detection.running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        
        # 添加更新指标的方法
        def _update_metric(self):
            if hasattr(self, 'change_prediction') and hasattr(self, 'label'):
                G_pred = self.change_prediction.detach()
                G_pred = torch.argmax(G_pred, dim=1)
                current_score = self.running_metric.update_cm(
                    pr=G_pred.cpu().numpy(), 
                    gt=self.label.detach().cpu().numpy()
                )
                return current_score
            return 0.0
        
        change_detection._update_metric = lambda: _update_metric(change_detection)
        print("✅ 性能指标跟踪器添加完成")

    # 设置数据集长度信息
    if 'len_train_dataloader' in opt:
        change_detection.len_train_dataloader = opt["len_train_dataloader"]
    if 'len_val_dataloader' in opt:
        change_detection.len_val_dataloader = opt["len_val_dataloader"]

    print("🚀 变化检测模型初始化完成\n")
    
    # 🔧 强制设备一致性设置
    if torch.cuda.is_available():
        target_device = torch.device('cuda:0')
        
        # 确保扩散模型在GPU
        if isinstance(diffusion.netG, nn.DataParallel):
            diffusion.netG.module = diffusion.netG.module.to(target_device)
        else:
            diffusion.netG = diffusion.netG.to(target_device)
        
        # 确保变化检测模型在GPU
        if hasattr(change_detection, 'netCD') and change_detection.netCD is not None:
            if isinstance(change_detection.netCD, nn.DataParallel):
                change_detection.netCD.module = change_detection.netCD.module.to(target_device)
            else:
                change_detection.netCD = change_detection.netCD.to(target_device)
        
        print(f"✅ 强制设备设置完成: {target_device}")
    
    # 处理CD模型的DataParallel
    if hasattr(change_detection, 'netCD') and change_detection.netCD is not None:
        if isinstance(change_detection.netCD, nn.DataParallel):
            change_detection.netCD = change_detection.netCD.module
            print("已解包CD模型的DataParallel")
        
        if torch.cuda.device_count() > 1:
            change_detection.netCD = change_detection.netCD.cuda()

    # 设置训练优化
    use_amp = setup_training_optimization(diffusion, change_detection)
    
    # 创建性能监控器
    performance_monitor = PerformanceMonitor()

    print("🔍 进行最终配置验证...")
    
    # 验证数据加载
    try:
        sample_batch = next(iter(train_loader))
        print(f"✅ 数据加载正常 - 批次大小: {sample_batch['A'].shape[0]}")
        
        required_keys = ['A', 'B', 'L']
        if opt['model_cd'].get('loss_type') == 'physics_constrained':
            required_keys.append('physical_data')
        
        missing_keys = [k for k in required_keys if k not in sample_batch]
        if missing_keys:
            print(f"⚠️  数据批次缺少字段: {missing_keys}")
        else:
            print(f"✅ 数据批次包含所有必需字段")
            
    except Exception as e:
        print(f"❌ 数据验证失败: {e}")
    
    # 验证模型前向传播
    try:
        with torch.no_grad():
            process_batch_efficiently(sample_batch, diffusion, change_detection, 
                                    opt, "test", 0)
        print("✅ 模型前向传播测试通过")
    except Exception as e:
        print(f"⚠️  模型前向传播测试失败: {e}")
        print("   将在实际训练中尝试恢复")
    
    print("🚀 所有设置完成，开始训练...\n")

    #################
    # 训练循环 #
    #################
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0

    if opt['phase'] == 'train':
        # 验证设备设置
        device = next(diffusion.netG.parameters()).device
        print(f"设备检查: 模型在 {device}")
        
        if device.type == 'cpu' and torch.cuda.is_available():
            target_device = torch.device('cuda:0')
            print(f"强制将模型从 {device} 移动到 {target_device}")
            diffusion.netG = diffusion.netG.to(target_device)
            change_detection.netCD = change_detection.netCD.to(target_device)
            device = next(diffusion.netG.parameters()).device
            print(f"移动后验证: 模型现在在 {device}")

        for current_epoch in range(start_epoch, n_epoch):
            epoch_start_time = time.time()
            change_detection._clear_cache()
            
            train_result_path = f'{opt["path"]["results"]}/train/{current_epoch}'
            os.makedirs(train_result_path, exist_ok=True)
            
            ################
            ### 训练阶段 ###
            ################
            print(f"\n🎯 开始训练 Epoch {current_epoch}/{n_epoch-1}")
            message = f'学习率: {change_detection.optCD.param_groups[0]["lr"]:.7f}'
            logger.info(message)
            
            for current_step, train_data in enumerate(train_loader):
                step_start_time = time.time()
                
                # 高效批量处理
                process_batch_efficiently(train_data, diffusion, change_detection, opt, "train")
                
                # 安全的训练步骤
                success = execute_training_step(change_detection, current_epoch)
                
                if not success:
                    print(f"跳过步骤 {current_step}")
                    continue
                
                # 记录性能
                step_time = time.time() - step_start_time
                memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                performance_monitor.log_step(step_time, memory_mb)
                
                # 优化的日志输出
                optimized_logging(current_step, current_epoch, n_epoch, train_loader, 
                                change_detection, logger, opt, "train", performance_monitor)
                
                # 新增：物理约束损失详细信息记录（可选）
                if current_step % opt['train']['train_print_freq'] == 0:
                    if hasattr(change_detection, 'get_physics_constraint_details'):
                        try:
                            physics_details = change_detection.get_physics_constraint_details()
                            if physics_details:
                                print(f"   📊 物理约束损失分解: {physics_details}")
                        except:
                            pass
                
                # 保存可视化结果（减少频率）
                save_freq = max(opt['train']['train_print_freq'] * 2, len(train_loader) // 5)
                if current_step % save_freq == 0:
                    try:
                        visuals = change_detection.get_current_visuals()
                        
                        # 确保设备一致性
                        device = train_data['A'].device
                        visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                        visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                        
                        pred_cm_expanded = visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                        gt_cm_expanded = visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                        
                        grid_img = torch.cat((train_data['A'], train_data['B'], 
                                            pred_cm_expanded, gt_cm_expanded), dim=0)
                        grid_img = Metrics.tensor2img(grid_img)
                        
                        Metrics.save_img(grid_img, 
                            f'{train_result_path}/img_A_B_pred_gt_e{current_epoch}_b{current_step}.png')
                    except Exception as e:
                        print(f"保存可视化失败: {e}")
                
                # 定期内存清理
                if current_step % 50 == 0:
                    torch.cuda.empty_cache()
            
            ### 训练epoch总结 ###
            try:
                change_detection._collect_epoch_states()
                logs = change_detection.get_current_log()
                
                epoch_time = time.time() - epoch_start_time
                message = f'[训练 Epoch {current_epoch}] mF1={logs["epoch_acc"]:.5f}, ' \
                         f'用时={epoch_time/60:.1f}分钟'
                
                print(f"\n✅ {message}")
                logger.info(message)
                
                # 详细指标
                for k, v in logs.items():
                    tb_logger.add_scalar(f'train/{k}', v, current_epoch)
                
                if wandb_logger:
                    wandb_metrics = {
                        'training/mF1': logs['epoch_acc'],
                        'training/mIoU': logs['miou'],
                        'training/OA': logs['acc'],
                        'training/change-F1': logs['F1_1'],
                        'training/no-change-F1': logs['F1_0'],
                        'training/change-IoU': logs['iou_1'],
                        'training/no-change-IoU': logs['iou_0'],
                        'training/train_step': current_epoch,
                        'training/loss': logs.get('l_cd'),
                    }
                    
                    # 新增：记录物理损失分量
                    if 'ce_loss' in logs:
                        wandb_metrics['training/ce_loss'] = logs['ce_loss']
                    if 'slope_constraint' in logs:
                        wandb_metrics['training/slope_constraint'] = logs['slope_constraint']
                    if 'spatial_constraint' in logs:
                        wandb_metrics['training/spatial_constraint'] = logs['spatial_constraint']
                    if 'size_constraint' in logs:
                        wandb_metrics['training/size_constraint'] = logs['size_constraint']
                    if 'geology_constraint' in logs:
                        wandb_metrics['training/geology_constraint'] = logs['geology_constraint']
                    
                    wandb_logger.log_metrics(wandb_metrics)
                    
            except Exception as e:
                print(f"训练指标收集错误: {e}")
            
            change_detection._clear_cache()
            change_detection._update_lr_schedulers()
            
            ##################
            ### 验证阶段 ###
            ##################
            if current_epoch % opt['train']['val_freq'] == 0:
                print(f"\n🔍 开始验证 Epoch {current_epoch}")
                val_start_time = time.time()
                
                val_result_path = f'{opt["path"]["results"]}/val/{current_epoch}'
                os.makedirs(val_result_path, exist_ok=True)

                for current_step, val_data in enumerate(val_loader):
                    with torch.no_grad():  # 验证时不需要梯度
                        process_batch_efficiently(val_data, diffusion, change_detection, opt, "val")
                        # 适配新的test接口
                        if hasattr(change_detection, '_temp_features_A') and hasattr(change_detection, '_temp_features_B'):
                            change_detection.test(
                                change_detection._temp_features_A,
                                change_detection._temp_features_B
                            )
                            # 清理临时特征
                            del change_detection._temp_features_A
                            del change_detection._temp_features_B
                        else:
                            change_detection.test()
                        
                        change_detection._collect_running_batch_states()
                    
                    # 验证日志（减少频率）
                    optimized_logging(current_step, current_epoch, n_epoch, val_loader, 
                                    change_detection, logger, opt, "val")
                    
                    # 验证可视化（更少频率）
                    if current_step % max(1, len(val_loader) // 3) == 0:
                        try:
                            visuals = change_detection.get_current_visuals()
                            device = val_data['A'].device
                            visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                            visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                            
                            pred_cm_expanded = visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                            gt_cm_expanded = visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                            
                            grid_img = torch.cat((val_data['A'], val_data['B'], 
                                                pred_cm_expanded, gt_cm_expanded), dim=0)
                            grid_img = Metrics.tensor2img(grid_img)
                            
                            Metrics.save_img(grid_img, 
                                f'{val_result_path}/img_A_B_pred_gt_e{current_epoch}_b{current_step}.png')
                        except Exception as e:
                            print(f"验证可视化失败: {e}")

                ### 验证总结 ### 
                try:
                    change_detection._collect_epoch_states()
                    logs = change_detection.get_current_log()
                    
                    val_time = time.time() - val_start_time
                    message = f'[验证 Epoch {current_epoch}] mF1={logs["epoch_acc"]:.5f}, ' \
                            f'用时={val_time/60:.1f}分钟'
                    
                    print(f"✅ {message}")
                    logger.info(message)
                    
                    for k, v in logs.items():
                        tb_logger.add_scalar(f'val/{k}', v, current_epoch)

                    # 🔍 详细的WandB调试记录
                    if wandb_logger:
                        try:
                            # 调试：打印所有logs
                            print("\n🔍 === WandB调试信息 ===")
                            print(f"当前epoch: {current_epoch}")
                            print(f"best_mF1: {best_mF1} (类型: {type(best_mF1)})")
                            print("logs内容:")
                            for k, v in logs.items():
                                print(f"  {k}: {v} (类型: {type(v)})")
                            
                            # 安全转换所有指标
                            def safe_convert(value, key):
                                if value is None:
                                    print(f"  ⚠️  {key}: None值")
                                    return None
                                try:
                                    if hasattr(value, 'item'):  # PyTorch tensor
                                        result = float(value.item())
                                    else:
                                        result = float(value)
                                    
                                    # 检查NaN和无穷大
                                    if result != result or result == float('inf') or result == float('-inf'):
                                        print(f"  ❌ {key}: 无效数值 {result}")
                                        return None
                                    
                                    print(f"  ✅ {key}: {value} → {result}")
                                    return result
                                except Exception as e:
                                    print(f"  ❌ {key}: 转换失败 {value} - {e}")
                                    return None
                            
                            # 构建安全的指标字典
                            validation_metrics = {}
                            
                            # 主要指标
                            for wandb_key, log_key in [
                                ('validation/mF1', 'epoch_acc'),
                                ('validation/loss', 'l_cd'),
                                ('validation/mIoU', 'miou'),
                                ('validation/accuracy', 'acc'),
                                ('validation/change_F1', 'F1_1'),
                                ('validation/no_change_F1', 'F1_0'),
                                ('validation/change_IoU', 'iou_1'),
                                ('validation/no_change_IoU', 'iou_0'),
                            ]:
                                converted = safe_convert(logs.get(log_key), wandb_key)
                                if converted is not None:
                                    validation_metrics[wandb_key] = converted
                            
                            # 简化命名的指标
                            for wandb_key, log_key in [
                                ('val_mF1', 'epoch_acc'),
                                ('val_loss', 'l_cd'),
                                ('val_mIoU', 'miou'),
                                ('val_accuracy', 'acc'),
                            ]:
                                converted = safe_convert(logs.get(log_key), wandb_key)
                                if converted is not None:
                                    validation_metrics[wandb_key] = converted
                            
                            # 其他指标
                            validation_metrics['epoch'] = current_epoch
                            validation_metrics['validation_step'] = current_epoch
                            
                            # best_mF1
                            converted_best = safe_convert(best_mF1, 'val_best_mF1')
                            if converted_best is not None:
                                validation_metrics['val_best_mF1'] = converted_best
                                validation_metrics['validation/best_mF1'] = converted_best
                            
                            # 时间
                            validation_metrics['validation/time_minutes'] = val_time / 60
                            
                            print(f"\n📊 将要记录的指标 ({len(validation_metrics)}个):")
                            for k, v in validation_metrics.items():
                                print(f"  {k}: {v}")
                            
                            # 记录到WandB
                            if validation_metrics:
                                wandb_logger.log_metrics(validation_metrics)
                                print(f"\n✅ WandB记录成功: {len(validation_metrics)}个指标")
                            else:
                                print("\n❌ 没有有效指标可记录")
                            
                            print("🔍 === WandB调试信息结束 ===\n")
                            
                        except Exception as e:
                            print(f"❌ WandB记录错误: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # 模型保存逻辑保持不变
                    if logs['epoch_acc'] > best_mF1:
                        is_best_model = True
                        best_mF1 = logs['epoch_acc']
                        print(f"🎉 最佳模型更新! mF1: {best_mF1:.5f}")
                        logger.info('[验证] 最佳模型更新，保存模型和训练状态')
                    else:
                        is_best_model = False
                        logger.info('[验证] 保存当前模型和训练状态')

                    change_detection.save_network(current_epoch, is_best_model=is_best_model)
                    
                except Exception as e:
                    print(f"验证指标收集错误: {e}")
                
                change_detection._clear_cache()
                print(f"--- 进入下一个Epoch ---\n")

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch})
            
            # Epoch结束清理
            torch.cuda.empty_cache()
                
        print("🎉 训练完成!")
        logger.info('训练结束')
        
    else:
        ##################
        ### 测试阶段 ###
        ##################
        logger.info('开始模型评估（测试）')
        print("🔍 开始测试...")
        
        test_result_path = f'{opt["path"]["results"]}/test/'
        os.makedirs(test_result_path, exist_ok=True)
        logger_test = logging.getLogger('test')
        change_detection._clear_cache()

        per_sample_metrics_list = [] # <--- 新增：初始化一个列表来存储每个样本的指标
        
        for current_step, test_data in enumerate(test_loader):
            with torch.no_grad():
                process_batch_efficiently(test_data, diffusion, change_detection, opt, "test")
                
                # 适配新的test接口
                if hasattr(change_detection, '_temp_features_A') and hasattr(change_detection, '_temp_features_B'):
                    change_detection.test(
                        change_detection._temp_features_A,
                        change_detection._temp_features_B
                    )
                    del change_detection._temp_features_A
                    del change_detection._temp_features_B
                else:
                    change_detection.test()
                
                change_detection._collect_running_batch_states()

            # 测试日志
            if current_step % max(1, len(test_loader) // 10) == 0:
                logs = change_detection.get_current_log()
                message = f'[测试] Step {current_step}/{len(test_loader)}, ' \
                         f'mF1: {logs["running_acc"]:.5f}'
                print(message)
                logger_test.info(message)

            # 保存测试结果
            try:
                visuals = change_detection.get_current_visuals()

                # --- 新增：为批次中的每个样本计算并收集指标 ---
                batch_pred_cm = visuals['pred_cm'] # 获取批量的预测图 [N, H, W]
                batch_gt_cm = visuals['gt_cm']   # 获取批量的真值图 [N, H, W]
                batch_paths = test_data.get('A_path', [f'sample_{current_step}_{i}' for i in range(batch_pred_cm.size(0))]) # 获取样本路径或生成ID

                for i in range(batch_pred_cm.size(0)):
                    pred_sample = batch_pred_cm[i]
                    gt_sample = batch_gt_cm[i]
                    
                    # 调用新函数计算指标
                    sample_metrics = calculate_per_sample_metrics(pred_sample, gt_sample)
                    
                    # 添加样本标识符
                    sample_metrics['sample_id'] = os.path.basename(batch_paths[i])
                    
                    # 添加到总列表
                    per_sample_metrics_list.append(sample_metrics)

                visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                
                # 单独保存图像
                img_A = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))
                img_B = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))
                gt_cm = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                          out_type=np.uint8, min_max=(0, 1))
                pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                            out_type=np.uint8, min_max=(0, 1))

                Metrics.save_img(img_A, f'{test_result_path}/img_A_{current_step}.png')
                Metrics.save_img(img_B, f'{test_result_path}/img_B_{current_step}.png')
                Metrics.save_img(pred_cm, f'{test_result_path}/img_pred_cm{current_step}.png')
                Metrics.save_img(gt_cm, f'{test_result_path}/img_gt_cm{current_step}.png')
                
            except Exception as e:
                print(f"测试保存失败: {e}")

        ### 测试总结 ###
        try:
            change_detection._collect_epoch_states()
            logs = change_detection.get_current_log()
            
            message = f'[测试总结] mF1={logs["epoch_acc"]:.5f}\n'
            for k, v in logs.items():
                message += f'{k}: {v:.4e} '
            message += '\n'
            
            print(f"✅ {message}")
            logger_test.info(message)

            # --- 新增：保存单样本指标到CSV文件 ---
            if per_sample_metrics_list:
                metrics_file_path = os.path.join(test_result_path, 'per_sample_metrics.csv')
                # 定义CSV文件的表头，确保 'sample_id' 在第一列
                fieldnames = ['sample_id'] + [key for key in per_sample_metrics_list[0].keys() if key != 'sample_id']
                
                with open(metrics_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(per_sample_metrics_list)
                
                print(f"📊 单样本详细指标已保存到: {metrics_file_path}")
            # --- 新增代码结束 ---

            if wandb_logger:
                wandb_logger.log_metrics({
                    'test/mF1': logs['epoch_acc'],
                    'test/mIoU': logs['miou'],
                    'test/OA': logs['acc'],
                    'test/change-F1': logs['F1_1'],
                    'test/no-change-F1': logs['F1_0'],
                    'test/change-IoU': logs['iou_1'],
                    'test/no-change-IoU': logs['iou_0'],
                })
                
        except Exception as e:
            print(f"测试指标收集错误: {e}")

        print("🎉 测试完成!")
        logger.info('测试结束')
        
