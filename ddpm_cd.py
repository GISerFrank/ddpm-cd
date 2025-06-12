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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 使用4个GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA调用，便于调试

# ==================== 优化版标签验证器 ====================
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
                fixed_labels = labels.clone()
                unique_vals = torch.unique(labels)
                if 255 in unique_vals:
                    fixed_labels[labels == 255] = 1
                data['L'] = torch.clamp(fixed_labels, 0, 1).long()
            return True
        
        # 第一次详细验证
        unique_vals = torch.unique(labels)
        min_val = labels.min().item()
        max_val = labels.max().item()
        
        print(f"\n🔍 [{phase}] 标签验证（仅显示一次）:")
        print(f"   形状: {labels.shape}, 数据类型: {labels.dtype}")
        print(f"   值范围: [{min_val}, {max_val}]")
        print(f"   唯一值: {unique_vals.tolist()}")
        
        # 判断标签类型
        self.is_normalized = (max_val <= 1.0 and min_val >= 0.0)
        
        if self.is_normalized:
            print(f"   🔧 检测到归一化标签，使用阈值二值化（阈值=0.5）")
            fixed_labels = (labels >= 0.5).long()
        else:
            print(f"   🔧 检测到标准标签，映射255→1")
            fixed_labels = labels.clone()
            if 255 in unique_vals:
                fixed_labels[labels == 255] = 1
            fixed_labels = torch.clamp(fixed_labels, 0, 1).long()
        
        # 验证修复结果
        final_unique = torch.unique(fixed_labels)
        zero_count = (fixed_labels == 0).sum().item()
        one_count = (fixed_labels == 1).sum().item()
        total = zero_count + one_count
        
        print(f"   ✅ 修复完成: 唯一值{final_unique.tolist()}")
        print(f"   📊 像素分布: 无变化={100*zero_count/total:.1f}%, 有变化={100*one_count/total:.1f}%")
        print(f"   ✅ 标签验证设置完成，后续批次将快速处理\n")
        
        # 保存统计信息
        self.label_stats = {
            'zero_ratio': zero_count / total,
            'one_ratio': one_count / total,
            'is_normalized': self.is_normalized
        }
        
        data['L'] = fixed_labels
        self.validation_done = True
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

# ==================== 优化版特征重排 ====================
def apply_feature_reordering_optimized(f_A, f_B, train_data, change_detection, opt, phase="train"):
    """
    内存优化的特征重排方案
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
        
        # 使用重排后的特征
        change_detection.feed_data(reordered_f_A, reordered_f_B, train_data)
        
        # 清理重排后的特征
        del reordered_f_A, reordered_f_B
        
        return True
        
    except Exception as e:
        print(f"❌ 特征重排失败: {e}")
        print("🔄 使用回退方案...")
        
        # 简化回退方案
        target_layers = [12, 13, 14]
        corrected_f_A = []
        corrected_f_B = []
        
        for fa, fb in zip(f_A, f_B):
            timestep_A = [fa[i] for i in target_layers if i < len(fa)]
            timestep_B = [fb[i] for i in target_layers if i < len(fb)]
            corrected_f_A.append(timestep_A)
            corrected_f_B.append(timestep_B)
        
        del f_A, f_B
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
def process_batch_efficiently(train_data, diffusion, change_detection, opt, phase="train"):
    """高效的批量处理 - 修复设备问题"""
    with memory_efficient_context():
        # 1. 快速标签验证
        label_validator.validate_and_fix_labels(train_data, phase)
        
        # 2. 🔧 强制设备一致性检查
        device = None
        try:
            device = next(diffusion.netG.parameters()).device
        except:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 确保所有数据在正确设备上
        for key, value in train_data.items():
            if torch.is_tensor(value):
                train_data[key] = value.to(device)
        
        # 3. 确保diffusion模型在正确设备
        if isinstance(diffusion.netG, nn.DataParallel):
            actual_model = diffusion.netG.module
            if next(actual_model.parameters()).device != device:
                actual_model = actual_model.to(device)
                diffusion.netG = nn.DataParallel(actual_model, device_ids=[0, 1, 2, 3])
        else:
            if next(diffusion.netG.parameters()).device != device:
                diffusion.netG = diffusion.netG.to(device)
        
        print(f"🔍 [{phase}] 设备检查 - 模型: {next(diffusion.netG.parameters()).device}, 数据: {train_data['A'].device}")
        
        # 4. 特征提取
        diffusion.feed_data(train_data)
        
        # 5. 收集特征
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
        apply_feature_reordering_optimized(f_A, f_B, train_data, change_detection, opt, phase)

# ==================== 优化的日志管理 ====================
def optimized_logging(current_step, current_epoch, n_epoch, loader, change_detection, 
                     logger, opt, phase="train", performance_monitor=None):
    """优化的日志输出"""
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
            metrics = f"Loss: {logs.get('l_cd', 0):.5f} mF1: {logs.get('running_acc', 0):.5f}"
            
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
def execute_training_step(change_detection):
    """执行训练步骤"""
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

    # 解析配置
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # 设置日志
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # 初始化WandbLogger
    if opt['enable_wandb']:
        import wandb
        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('epoch')
        wandb.define_metric('training/train_step')
        wandb.define_metric("training/*", step_metric="train_step")
        wandb.define_metric('validation/val_step')
        wandb.define_metric("validation/*", step_metric="val_step")
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None

    # 加载数据集
    print("🔄 加载数据集...")
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("Creating [train] change-detection dataloader.")
            train_set = Data.create_cd_dataset(dataset_opt, phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

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
    change_detection = Model.create_CD_model(opt)
    
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
                success = execute_training_step(change_detection)
                
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
                    wandb_logger.log_metrics({
                        'training/mF1': logs['epoch_acc'],
                        'training/mIoU': logs['miou'],
                        'training/OA': logs['acc'],
                        'training/change-F1': logs['F1_1'],
                        'training/no-change-F1': logs['F1_0'],
                        'training/change-IoU': logs['iou_1'],
                        'training/no-change-IoU': logs['iou_0'],
                        'training/train_step': current_epoch,
                        'training/loss': logs.get('l_cd'),
                    })
                    
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
        
        for current_step, test_data in enumerate(test_loader):
            with torch.no_grad():
                process_batch_efficiently(test_data, diffusion, change_detection, opt, "test")
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
        
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist  # 分布式训练
# import torch.multiprocessing as mp  # 多进程
# from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式DataParallel
# from torch.utils.data.distributed import DistributedSampler  # 分布式采样器

# import data as Data
# import model as Model
# import argparse
# import logging
# import core.logger as Logger
# import core.metrics as Metrics
# from core.wandb_logger import WandbLogger
# from tensorboardX import SummaryWriter
# import os
# import numpy as np
# from model.cd_modules.cd_head import cd_head 
# from misc.print_diffuse_feats import print_feats
# import time
# from contextlib import contextmanager

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 使用4个GPU
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA调用，便于调试

# # ==================== 优化版标签验证器 ====================
# class LabelValidator:
#     """高效标签验证器 - 单例模式"""
#     _instance = None
#     _initialized = False
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def __init__(self):
#         if not self._initialized:
#             self.is_normalized = None
#             self.validation_done = False
#             self.label_stats = {}
#             LabelValidator._initialized = True
    
#     def validate_and_fix_labels(self, data, phase="train"):
#         """
#         高效标签验证 - 只在第一次详细检查，后续快速处理
#         """
#         if 'L' not in data:
#             return False
        
#         labels = data['L']
        
#         # 快速通道：如果已经验证过，直接处理
#         if self.validation_done:
#             if self.is_normalized:
#                 data['L'] = (labels >= 0.5).long()
#             else:
#                 fixed_labels = labels.clone()
#                 unique_vals = torch.unique(labels)
#                 if 255 in unique_vals:
#                     fixed_labels[labels == 255] = 1
#                 data['L'] = torch.clamp(fixed_labels, 0, 1).long()
#             return True
        
#         # 第一次详细验证
#         unique_vals = torch.unique(labels)
#         min_val = labels.min().item()
#         max_val = labels.max().item()
        
#         print(f"\n🔍 [{phase}] 标签验证（仅显示一次）:")
#         print(f"   形状: {labels.shape}, 数据类型: {labels.dtype}")
#         print(f"   值范围: [{min_val}, {max_val}]")
#         print(f"   唯一值: {unique_vals.tolist()}")
        
#         # 判断标签类型
#         self.is_normalized = (max_val <= 1.0 and min_val >= 0.0)
        
#         if self.is_normalized:
#             print(f"   🔧 检测到归一化标签，使用阈值二值化（阈值=0.5）")
#             fixed_labels = (labels >= 0.5).long()
#         else:
#             print(f"   🔧 检测到标准标签，映射255→1")
#             fixed_labels = labels.clone()
#             if 255 in unique_vals:
#                 fixed_labels[labels == 255] = 1
#             fixed_labels = torch.clamp(fixed_labels, 0, 1).long()
        
#         # 验证修复结果
#         final_unique = torch.unique(fixed_labels)
#         zero_count = (fixed_labels == 0).sum().item()
#         one_count = (fixed_labels == 1).sum().item()
#         total = zero_count + one_count
        
#         print(f"   ✅ 修复完成: 唯一值{final_unique.tolist()}")
#         print(f"   📊 像素分布: 无变化={100*zero_count/total:.1f}%, 有变化={100*one_count/total:.1f}%")
#         print(f"   ✅ 标签验证设置完成，后续批次将快速处理\n")
        
#         # 保存统计信息
#         self.label_stats = {
#             'zero_ratio': zero_count / total,
#             'one_ratio': one_count / total,
#             'is_normalized': self.is_normalized
#         }
        
#         data['L'] = fixed_labels
#         self.validation_done = True
#         return True

# # 全局标签验证器
# label_validator = LabelValidator()

# # ==================== 内存管理工具 ====================
# @contextmanager
# def memory_efficient_context():
#     """内存管理上下文"""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     try:
#         yield
#     finally:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

# class PerformanceMonitor:
#     """性能监控器"""
#     def __init__(self):
#         self.step_times = []
#         self.memory_usage = []
#         self.start_time = time.time()
    
#     def log_step(self, step_time, memory_mb=None):
#         self.step_times.append(step_time)
#         if memory_mb:
#             self.memory_usage.append(memory_mb)
    
#     def get_stats(self):
#         if not self.step_times:
#             return "无统计数据"
        
#         avg_time = np.mean(self.step_times[-100:])  # 最近100步平均
#         total_time = time.time() - self.start_time
        
#         stats = f"平均步时: {avg_time:.2f}s, 总时间: {total_time/60:.1f}min"
        
#         if self.memory_usage:
#             avg_memory = np.mean(self.memory_usage[-10:])
#             stats += f", 显存: {avg_memory:.1f}MB"
        
#         return stats

# # ==================== 优化版特征重排 ====================
# def apply_feature_reordering_optimized(f_A, f_B, train_data, change_detection, opt, phase="train"):
#     """
#     内存优化的特征重排方案
#     """
#     try:
#         feat_scales = opt['model_cd']['feat_scales']  # [2, 5, 8, 11, 14]
#         cd_expected_order = sorted(feat_scales, reverse=True)  # [14, 11, 8, 5, 2]
        
#         # 只在第一次显示信息
#         if not hasattr(apply_feature_reordering_optimized, '_logged'):
#             print("🎯 使用优化的特征重排方案")
#             print("   保持原始多尺度配置的完整语义")
#             for i, scale in enumerate(cd_expected_order):
#                 print(f"     Block{i}: 使用layer{scale}特征")
#             apply_feature_reordering_optimized._logged = True
        
#         # 高效重排：直接在原地修改
#         reordered_f_A = []
#         reordered_f_B = []
        
#         for fa, fb in zip(f_A, f_B):
#             if isinstance(fa, list) and len(fa) > max(feat_scales):
#                 timestep_A = [fa[scale] for scale in cd_expected_order]
#                 timestep_B = [fb[scale] for scale in cd_expected_order]
#                 reordered_f_A.append(timestep_A)
#                 reordered_f_B.append(timestep_B)
#             else:
#                 raise ValueError(f"特征格式错误: 期望list长度>{max(feat_scales)}, 实际{type(fa)}")
        
#         # 清理原始特征释放内存
#         del f_A, f_B
        
#         # 使用重排后的特征
#         change_detection.feed_data(reordered_f_A, reordered_f_B, train_data)
        
#         # 清理重排后的特征
#         del reordered_f_A, reordered_f_B
        
#         return True
        
#     except Exception as e:
#         print(f"❌ 特征重排失败: {e}")
#         print("🔄 使用回退方案...")
        
#         # 简化回退方案
#         target_layers = [12, 13, 14]
#         corrected_f_A = []
#         corrected_f_B = []
        
#         for fa, fb in zip(f_A, f_B):
#             timestep_A = [fa[i] for i in target_layers if i < len(fa)]
#             timestep_B = [fb[i] for i in target_layers if i < len(fb)]
#             corrected_f_A.append(timestep_A)
#             corrected_f_B.append(timestep_B)
        
#         del f_A, f_B
#         change_detection.feed_data(corrected_f_A, corrected_f_B, train_data)
#         del corrected_f_A, corrected_f_B
        
#         return False

# # ==================== 训练优化设置 ====================
# def setup_training_optimization(diffusion, change_detection):
#     """设置训练优化"""
#     print("🚀 设置训练优化...")
    
#     # 启用CUDA优化
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False
    
#     # 检查混合精度支持
#     use_amp = False
#     if torch.cuda.is_available():
#         try:
#             from torch.cuda.amp import autocast, GradScaler
#             use_amp = True
#             print("   ✅ 支持混合精度训练")
#         except ImportError:
#             print("   ⚠️  不支持混合精度训练")
    
#     # 设置diffusion模型为eval模式（如果不需要训练）
#     if hasattr(diffusion.netG, 'eval'):
#         diffusion.netG.eval()
#         print("   ✅ Diffusion模型设置为eval模式")
    
#     # 检查多GPU设置
#     if torch.cuda.device_count() > 1:
#         print(f"   ✅ 检测到{torch.cuda.device_count()}个GPU")
        
#         # 显示GPU状态
#         for i in range(torch.cuda.device_count()):
#             props = torch.cuda.get_device_properties(i)
#             memory_gb = props.total_memory / 1024**3
#             print(f"     GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    
#     print("🚀 训练优化设置完成\n")
    
#     return use_amp

# # ==================== 批量处理优化 ====================
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

# # ==================== 优化的日志管理 ====================
# def optimized_logging(current_step, current_epoch, n_epoch, loader, change_detection, 
#                      logger, opt, phase="train", performance_monitor=None):
#     """优化的日志输出"""
#     # 动态调整日志频率
#     if phase == "train":
#         base_freq = opt['train'].get('train_print_freq', 10)
#         log_freq = max(base_freq, len(loader) // 1000)  # 至少每5%显示一次
#     else:
#         log_freq = max(1, len(loader) // 500)  # 验证时每10%显示一次
    
#     if current_step % log_freq == 0:
#         try:
#             logs = change_detection.get_current_log()
            
#             # 基础信息
#             progress = f"[{current_epoch}/{n_epoch-1}]"
#             step_info = f"Step {current_step}/{len(loader)}"
#             metrics = f"Loss: {logs.get('l_cd', 0):.5f} mF1: {logs.get('running_acc', 0):.5f}"
            
#             # 性能信息
#             perf_info = ""
#             if performance_monitor:
#                 perf_info = f" | {performance_monitor.get_stats()}"
            
#             message = f"{progress} {step_info} {metrics}{perf_info}"
#             print(message)
            
#         except Exception as e:
#             print(f"日志输出错误: {e}")

# # ==================== 错误处理装饰器 ====================
# def safe_training_step(func):
#     """安全训练步骤装饰器"""
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except RuntimeError as e:
#             if "device-side assert triggered" in str(e) or "CUDA" in str(e):
#                 print(f"⚠️  CUDA错误已自动处理: {str(e)[:100]}...")
#                 torch.cuda.empty_cache()
#                 return False
#             else:
#                 raise
#         except Exception as e:
#             print(f"❌ 训练步骤错误: {e}")
#             return False
#     return wrapper

# @safe_training_step
# def execute_training_step(change_detection):
#     """执行训练步骤"""
#     change_detection.optimize_parameters()
#     change_detection._collect_running_batch_states()
#     return True

# # ==================== checkpoint恢复 ====================
# def load_checkpoint_if_exists(change_detection, opt):
#     """修正版本：优先best模型，修复加载接口"""
    
#     # 🎯 指定的checkpoint路径
#     checkpoint_dir = os.path.expanduser("~/bowen/DDPM-CD/ddpm-cd/checkpoint/gvlm_cd_ddpm")
    
#     if not os.path.exists(checkpoint_dir):
#         print(f"🔍 Checkpoint目录不存在: {checkpoint_dir}")
#         return 0, 0.0
    
#     print(f"🔍 检查checkpoint目录: {checkpoint_dir}")
    
#     import glob
#     import re
    
#     # ========================================
#     # 🥇 第一优先级：检查最佳模型
#     # ========================================
#     best_gen_file = os.path.join(checkpoint_dir, "best_cd_model_gen.pth")
#     best_opt_file = os.path.join(checkpoint_dir, "best_cd_model_opt.pth")
    
#     if os.path.exists(best_gen_file):
#         print("🏆 发现最佳模型，优先使用最佳模型")
#         print(f"   最佳模型文件: {best_gen_file}")
#         print(f"   最佳优化器文件: {best_opt_file}")
        
#         success = load_model_safe(change_detection, best_gen_file, best_opt_file)
        
#         if success:
#             print("✅ 最佳模型加载成功")
#             # 从最佳模型开始，可以设置一个较高的epoch或从0开始
#             return 0, 0.8  # 从epoch 0开始，但best_mF1设置较高值表示这是好模型
#         else:
#             print("❌ 最佳模型加载失败，尝试最新epoch模型")
    
#     # ========================================
#     # 🥈 第二优先级：查找最新epoch模型
#     # ========================================
#     gen_files = glob.glob(os.path.join(checkpoint_dir, "cd_model_E*_gen.pth"))
    
#     if gen_files:
#         print(f"🔍 找到的epoch模型文件: {[os.path.basename(f) for f in gen_files]}")
        
#         # 提取epoch数字并排序
#         epochs = []
#         for f in gen_files:
#             match = re.search(r'cd_model_E(\d+)_gen\.pth', f)
#             if match:
#                 epochs.append(int(match.group(1)))
        
#         if epochs:
#             latest_epoch = max(epochs)
            
#             # 构建文件路径
#             gen_file = os.path.join(checkpoint_dir, f"cd_model_E{latest_epoch}_gen.pth")
#             opt_file = os.path.join(checkpoint_dir, f"cd_model_E{latest_epoch}_opt.pth")
            
#             print(f"🔄 使用最新epoch模型: Epoch {latest_epoch}")
#             print(f"   模型文件: {gen_file}")
#             print(f"   优化器文件: {opt_file}")
            
#             success = load_model_safe(change_detection, gen_file, opt_file)
            
#             if success:
#                 print("✅ 最新epoch模型加载成功")
                
#                 # 检查best_mF1（可以从某个记录文件读取，或设置默认值）
#                 best_mF1 = 0.0
#                 if os.path.exists(best_gen_file):
#                     best_mF1 = 0.5  # 如果有best文件但加载失败，设置一个中等值
                
#                 return latest_epoch + 1, best_mF1
#             else:
#                 print("❌ 最新epoch模型也加载失败")
    
#     print("🆕 没有找到可用的checkpoint，从头开始训练")
#     return 0, 0.0


# def load_model_safe(change_detection, gen_file, opt_file):
#     """安全的模型加载方法 - 尝试多种加载方式"""
    
#     if not os.path.exists(gen_file):
#         print(f"❌ 模型文件不存在: {gen_file}")
#         return False
    
#     print(f"🔄 尝试加载模型: {os.path.basename(gen_file)}")
    
#     # ========================================
#     # 方法1: 直接torch.load + 手动设置state_dict
#     # ========================================
#     try:
#         print("   🔄 方法1: 直接torch.load")
#         checkpoint = torch.load(gen_file, map_location='cpu')
        
#         if hasattr(change_detection, 'netCD') and change_detection.netCD is not None:
#             # 检查checkpoint结构
#             if isinstance(checkpoint, dict):
#                 print(f"   📋 Checkpoint keys: {list(checkpoint.keys())}")
                
#                 # 尝试不同的key
#                 state_dict = None
#                 for key in ['model_state_dict', 'state_dict', 'model', 'netCD']:
#                     if key in checkpoint:
#                         state_dict = checkpoint[key]
#                         print(f"   ✅ 使用key: {key}")
#                         break
                
#                 if state_dict is None:
#                     # 直接作为state_dict
#                     state_dict = checkpoint
#                     print("   ✅ 直接作为state_dict")
#             else:
#                 state_dict = checkpoint
#                 print("   ✅ Checkpoint是state_dict")
            
#             # 加载state_dict
#             change_detection.netCD.load_state_dict(state_dict, strict=False)
#             print("   ✅ 方法1: 模型权重加载成功")
            
#             # 尝试加载优化器
#             load_optimizer_safe(change_detection, opt_file)
            
#             return True
            
#     except Exception as e:
#         print(f"   ❌ 方法1失败: {e}")
    
#     # ========================================
#     # 方法2: 尝试无参数load_network (设置路径)
#     # ========================================
#     try:
#         print("   🔄 方法2: 无参数load_network")
        
#         # 尝试设置路径到opt中
#         if hasattr(change_detection, 'opt'):
#             # 备份原路径
#             original_path = change_detection.opt.get('path', {}).get('resume_state', '')
            
#             # 设置新路径
#             if 'path' not in change_detection.opt:
#                 change_detection.opt['path'] = {}
#             change_detection.opt['path']['resume_state'] = gen_file
            
#             # 尝试加载
#             change_detection.load_network()
            
#             # 恢复原路径
#             if original_path:
#                 change_detection.opt['path']['resume_state'] = original_path
            
#             print("   ✅ 方法2: 加载成功")
#             load_optimizer_safe(change_detection, opt_file)
#             return True
            
#     except Exception as e:
#         print(f"   ❌ 方法2失败: {e}")
    
#     # ========================================
#     # 方法3: 查找其他加载方法
#     # ========================================
#     try:
#         print("   🔄 方法3: 查找其他加载方法")
        
#         # 尝试常见的方法名
#         load_methods = ['load_model', 'load_checkpoint', 'resume_from_checkpoint', 'load_state_dict']
        
#         for method_name in load_methods:
#             if hasattr(change_detection, method_name):
#                 print(f"   🔄 尝试: {method_name}")
#                 method = getattr(change_detection, method_name)
                
#                 try:
#                     # 尝试带参数调用
#                     method(gen_file)
#                     print(f"   ✅ 方法3成功: {method_name}(gen_file)")
#                     load_optimizer_safe(change_detection, opt_file)
#                     return True
#                 except TypeError:
#                     # 尝试无参数调用
#                     try:
#                         method()
#                         print(f"   ✅ 方法3成功: {method_name}()")
#                         load_optimizer_safe(change_detection, opt_file)
#                         return True
#                     except:
#                         continue
#                 except Exception as e:
#                     print(f"   ❌ {method_name}失败: {e}")
#                     continue
        
#     except Exception as e:
#         print(f"   ❌ 方法3失败: {e}")
    
#     print("   ❌ 所有加载方法都失败")
#     return False


# def load_optimizer_safe(change_detection, opt_file):
#     """安全的优化器加载"""
#     if not os.path.exists(opt_file):
#         print("   ⚠️  优化器文件不存在")
#         return False
    
#     try:
#         print(f"   🔄 加载优化器: {os.path.basename(opt_file)}")
#         opt_state = torch.load(opt_file, map_location='cpu')
        
#         # 检查优化器属性
#         optimizer = None
#         for attr_name in ['optCD', 'optimizer', 'opt_CD', 'optim']:
#             if hasattr(change_detection, attr_name):
#                 optimizer = getattr(change_detection, attr_name)
#                 if optimizer is not None:
#                     print(f"   📋 找到优化器属性: {attr_name}")
#                     break
        
#         if optimizer is not None:
#             # 检查opt_state结构
#             if isinstance(opt_state, dict) and 'state_dict' in opt_state:
#                 optimizer.load_state_dict(opt_state['state_dict'])
#             else:
#                 optimizer.load_state_dict(opt_state)
            
#             print("   ✅ 优化器状态加载成功")
#             return True
#         else:
#             print("   ⚠️  没有找到优化器属性")
#             return False
            
#     except Exception as e:
#         print(f"   ❌ 优化器加载失败: {e}")
#         return False


# def debug_change_detection_methods(change_detection):
#     """调试change_detection的方法 - 可选调用"""
#     print("\n🔍 === change_detection 调试信息 ===")
    
#     # 查看类型
#     print(f"对象类型: {type(change_detection)}")
    
#     # 查看load相关方法
#     load_methods = [attr for attr in dir(change_detection) if 'load' in attr.lower() and callable(getattr(change_detection, attr))]
#     print(f"Load方法: {load_methods}")
    
#     # 查看优化器相关属性
#     opt_attrs = [attr for attr in dir(change_detection) if 'opt' in attr.lower()]
#     print(f"优化器相关属性: {opt_attrs}")
    
#     # 查看网络相关属性
#     net_attrs = [attr for attr in dir(change_detection) if 'net' in attr.lower()]
#     print(f"网络相关属性: {net_attrs}")
    
#     print("🔍 === 调试信息结束 ===\n")

# # ==================== 主函数 ====================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, default='config/ddpm_cd.json',
#                         help='JSON file for configuration')
#     parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
#                         help='Run either train(training + validation) or testing', default='train')
#     parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
#     parser.add_argument('-debug', '-d', action='store_true')
#     parser.add_argument('-enable_wandb', action='store_true')
#     parser.add_argument('-log_eval', action='store_true')

#     # 解析配置
#     args = parser.parse_args()
#     opt = Logger.parse(args)
#     opt = Logger.dict_to_nonedict(opt)

#     # 设置日志
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True

#     Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
#     Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
#     logger = logging.getLogger('base')
#     logger.info(Logger.dict2str(opt))
#     tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

#     # 初始化WandbLogger
#     if opt['enable_wandb']:
#         import wandb
#         print("Initializing wandblog.")
#         wandb_logger = WandbLogger(opt)
#         wandb.define_metric('epoch')
#         wandb.define_metric('training/train_step')
#         wandb.define_metric("training/*", step_metric="train_step")
#         wandb.define_metric('validation/val_step')
#         wandb.define_metric("validation/*", step_metric="val_step")
#         train_step = 0
#         val_step = 0
#     else:
#         wandb_logger = None

#     # 加载数据集
#     print("🔄 加载数据集...")
#     for phase, dataset_opt in opt['datasets'].items():
#         if phase == 'train' and args.phase != 'test':
#             print("Creating [train] change-detection dataloader.")
#             train_set = Data.create_cd_dataset(dataset_opt, phase)
#             train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
#             opt['len_train_dataloader'] = len(train_loader)

#         elif phase == 'val' and args.phase != 'test':
#             print("Creating [val] change-detection dataloader.")
#             val_set = Data.create_cd_dataset(dataset_opt, phase)
#             val_loader = Data.create_cd_dataloader(val_set, dataset_opt, phase)
#             opt['len_val_dataloader'] = len(val_loader)
        
#         elif phase == 'test' and args.phase == 'test':
#             print("Creating [test] change-detection dataloader.")
#             test_set = Data.create_cd_dataset(dataset_opt, phase)
#             test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase)
#             opt['len_test_dataloader'] = len(test_loader)
    
#     logger.info('Initial Dataset Finished')

#     # 加载模型
#     print("🔄 加载扩散模型...")
#     diffusion = Model.create_model(opt)
#     logger.info('Initial Diffusion Model Finished')

#     # 处理DataParallel
#     if isinstance(diffusion.netG, nn.DataParallel):
#         diffusion.netG = diffusion.netG.module
#         print("已解包diffusion模型的DataParallel")

#     # 多GPU设置
#     if torch.cuda.device_count() > 1:
#         print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
#         diffusion.netG = diffusion.netG.cuda()
#         diffusion.netG = nn.DataParallel(diffusion.netG, device_ids=[0, 1, 2, 3])
        
#         # 适度增加batch size
#         for phase in opt['datasets']:
#             if 'batch_size' in opt['datasets'][phase]:
#                 original_bs = opt['datasets'][phase]['batch_size']
#                 # 可以根据GPU数量调整
#                 # opt['datasets'][phase]['batch_size'] = original_bs * 2
#                 print(f"{phase} batch_size: {original_bs}")

#     # 设置噪声调度
#     diffusion.set_new_noise_schedule(
#         opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
#     # 创建变化检测模型
#     print("🔄 加载变化检测模型...")
#     change_detection = Model.create_CD_model(opt)
    
#     # 处理CD模型的DataParallel
#     if hasattr(change_detection, 'netCD') and change_detection.netCD is not None:
#         if isinstance(change_detection.netCD, nn.DataParallel):
#             change_detection.netCD = change_detection.netCD.module
#             print("已解包CD模型的DataParallel")
        
#         if torch.cuda.device_count() > 1:
#             change_detection.netCD = change_detection.netCD.cuda()

#     # 设置训练优化
#     use_amp = setup_training_optimization(diffusion, change_detection)
    
#     # 创建性能监控器
#     performance_monitor = PerformanceMonitor()

#     print("🚀 所有设置完成，开始训练...\n")

#     #################
#     # 训练循环 #
#     #################
#     n_epoch = opt['train']['n_epoch']
#     # best_mF1 = 0.0
#     # start_epoch = 0
#     start_epoch, best_mF1 = load_checkpoint_if_exists(change_detection, opt)

#     # 检查是否有保存的模型
#     checkpoint_dir = opt['path'].get('checkpoint', opt['path']['models'])
#     latest_model = os.path.join(checkpoint_dir, 'latest_net_CD.pth')

#     if os.path.exists(latest_model):
#         print(f"🔄 发现保存的模型: {latest_model}")
#         change_detection.load_network(latest_model)
        
#         # 尝试从文件名解析epoch（如果命名规范）
#         try:
#             # 假设文件名包含epoch信息
#             import re
#             match = re.search(r'epoch_(\d+)', latest_model)
#             if match:
#                 start_epoch = int(match.group(1)) + 1
#                 print(f"✅ 从epoch {start_epoch}继续训练")
#         except:
#             print("✅ 从保存的模型继续训练（epoch重置为0）")
#     else:
#         print("🆕 从头开始训练")

#     if opt['phase'] == 'train':
#         # 验证设备设置
#         device = next(diffusion.netG.parameters()).device
#         print(f"设备检查: 模型在 {device}")
        
#         if device.type == 'cpu' and torch.cuda.is_available():
#             target_device = torch.device('cuda:0')
#             print(f"强制将模型从 {device} 移动到 {target_device}")
#             diffusion.netG = diffusion.netG.to(target_device)
#             change_detection.netCD = change_detection.netCD.to(target_device)
#             device = next(diffusion.netG.parameters()).device
#             print(f"移动后验证: 模型现在在 {device}")

#         for current_epoch in range(start_epoch, n_epoch):
#             epoch_start_time = time.time()
#             change_detection._clear_cache()
            
#             train_result_path = f'{opt["path"]["results"]}/train/{current_epoch}'
#             os.makedirs(train_result_path, exist_ok=True)
            
#             ################
#             ### 训练阶段 ###
#             ################
#             print(f"\n🎯 开始训练 Epoch {current_epoch}/{n_epoch-1}")
#             message = f'学习率: {change_detection.optCD.param_groups[0]["lr"]:.7f}'
#             logger.info(message)
            
#             for current_step, train_data in enumerate(train_loader):
#                 step_start_time = time.time()
                
#                 # 高效批量处理
#                 process_batch_efficiently(train_data, diffusion, change_detection, opt, "train")
                
#                 # 安全的训练步骤
#                 success = execute_training_step(change_detection)
                
#                 if not success:
#                     print(f"跳过步骤 {current_step}")
#                     continue
                
#                 # 记录性能
#                 step_time = time.time() - step_start_time
#                 memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
#                 performance_monitor.log_step(step_time, memory_mb)
                
#                 # 优化的日志输出
#                 optimized_logging(current_step, current_epoch, n_epoch, train_loader, 
#                                 change_detection, logger, opt, "train", performance_monitor)
                
#                 # 保存可视化结果（减少频率）
#                 save_freq = max(opt['train']['train_print_freq'] * 2, len(train_loader) // 5)
#                 if current_step % save_freq == 0:
#                     try:
#                         visuals = change_detection.get_current_visuals()
                        
#                         # 确保设备一致性
#                         device = train_data['A'].device
#                         visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
#                         visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                        
#                         pred_cm_expanded = visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
#                         gt_cm_expanded = visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                        
#                         grid_img = torch.cat((train_data['A'], train_data['B'], 
#                                             pred_cm_expanded, gt_cm_expanded), dim=0)
#                         grid_img = Metrics.tensor2img(grid_img)
                        
#                         Metrics.save_img(grid_img, 
#                             f'{train_result_path}/img_A_B_pred_gt_e{current_epoch}_b{current_step}.png')
#                     except Exception as e:
#                         print(f"保存可视化失败: {e}")
                
#                 # 定期内存清理
#                 if current_step % 50 == 0:
#                     torch.cuda.empty_cache()
            
#             ### 训练epoch总结 ###
#             try:
#                 change_detection._collect_epoch_states()
#                 logs = change_detection.get_current_log()
                
#                 epoch_time = time.time() - epoch_start_time
#                 message = f'[训练 Epoch {current_epoch}] mF1={logs["epoch_acc"]:.5f}, ' \
#                          f'用时={epoch_time/60:.1f}分钟'
                
#                 print(f"\n✅ {message}")
#                 logger.info(message)
                
#                 # 详细指标
#                 for k, v in logs.items():
#                     tb_logger.add_scalar(f'train/{k}', v, current_epoch)
                
#                 if wandb_logger:
#                     wandb_logger.log_metrics({
#                         'training/mF1': logs['epoch_acc'],
#                         'training/mIoU': logs['miou'],
#                         'training/OA': logs['acc'],
#                         'training/change-F1': logs['F1_1'],
#                         'training/no-change-F1': logs['F1_0'],
#                         'training/change-IoU': logs['iou_1'],
#                         'training/no-change-IoU': logs['iou_0'],
#                         'training/train_step': current_epoch,
#                         'training/loss': logs.get('l_cd'),
#                     })
                    
#             except Exception as e:
#                 print(f"训练指标收集错误: {e}")
            
#             change_detection._clear_cache()
#             change_detection._update_lr_schedulers()
            
#             ##################
#             ### 验证阶段 ###
#             ##################
#             if current_epoch % opt['train']['val_freq'] == 0:
#                 print(f"\n🔍 开始验证 Epoch {current_epoch}")
#                 val_start_time = time.time()
                
#                 val_result_path = f'{opt["path"]["results"]}/val/{current_epoch}'
#                 os.makedirs(val_result_path, exist_ok=True)

#                 for current_step, val_data in enumerate(val_loader):
#                     with torch.no_grad():  # 验证时不需要梯度
#                         process_batch_efficiently(val_data, diffusion, change_detection, opt, "val")
#                         change_detection.test()
#                         change_detection._collect_running_batch_states()
                    
#                     # 验证日志（减少频率）
#                     optimized_logging(current_step, current_epoch, n_epoch, val_loader, 
#                                     change_detection, logger, opt, "val")
                    
#                     # 验证可视化（更少频率）
#                     if current_step % max(1, len(val_loader) // 3) == 0:
#                         try:
#                             visuals = change_detection.get_current_visuals()
#                             device = val_data['A'].device
#                             visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
#                             visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                            
#                             pred_cm_expanded = visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
#                             gt_cm_expanded = visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                            
#                             grid_img = torch.cat((val_data['A'], val_data['B'], 
#                                                 pred_cm_expanded, gt_cm_expanded), dim=0)
#                             grid_img = Metrics.tensor2img(grid_img)
                            
#                             Metrics.save_img(grid_img, 
#                                 f'{val_result_path}/img_A_B_pred_gt_e{current_epoch}_b{current_step}.png')
#                         except Exception as e:
#                             print(f"验证可视化失败: {e}")

#                 ### 验证总结 ### 
#                 try:
#                     change_detection._collect_epoch_states()
#                     logs = change_detection.get_current_log()
                    
#                     val_time = time.time() - val_start_time
#                     message = f'[验证 Epoch {current_epoch}] mF1={logs["epoch_acc"]:.5f}, ' \
#                             f'用时={val_time/60:.1f}分钟'
                    
#                     print(f"✅ {message}")
#                     logger.info(message)
                    
#                     for k, v in logs.items():
#                         tb_logger.add_scalar(f'val/{k}', v, current_epoch)

#                     # 🔍 详细的WandB调试记录
#                     if wandb_logger:
#                         try:
#                             # 调试：打印所有logs
#                             print("\n🔍 === WandB调试信息 ===")
#                             print(f"当前epoch: {current_epoch}")
#                             print(f"best_mF1: {best_mF1} (类型: {type(best_mF1)})")
#                             print("logs内容:")
#                             for k, v in logs.items():
#                                 print(f"  {k}: {v} (类型: {type(v)})")
                            
#                             # 安全转换所有指标
#                             def safe_convert(value, key):
#                                 if value is None:
#                                     print(f"  ⚠️  {key}: None值")
#                                     return None
#                                 try:
#                                     if hasattr(value, 'item'):  # PyTorch tensor
#                                         result = float(value.item())
#                                     else:
#                                         result = float(value)
                                    
#                                     # 检查NaN和无穷大
#                                     if result != result or result == float('inf') or result == float('-inf'):
#                                         print(f"  ❌ {key}: 无效数值 {result}")
#                                         return None
                                    
#                                     print(f"  ✅ {key}: {value} → {result}")
#                                     return result
#                                 except Exception as e:
#                                     print(f"  ❌ {key}: 转换失败 {value} - {e}")
#                                     return None
                            
#                             # 构建安全的指标字典
#                             validation_metrics = {}
                            
#                             # 主要指标
#                             for wandb_key, log_key in [
#                                 ('validation/mF1', 'epoch_acc'),
#                                 ('validation/loss', 'l_cd'),
#                                 ('validation/mIoU', 'miou'),
#                                 ('validation/accuracy', 'acc'),
#                                 ('validation/change_F1', 'F1_1'),
#                                 ('validation/no_change_F1', 'F1_0'),
#                                 ('validation/change_IoU', 'iou_1'),
#                                 ('validation/no_change_IoU', 'iou_0'),
#                             ]:
#                                 converted = safe_convert(logs.get(log_key), wandb_key)
#                                 if converted is not None:
#                                     validation_metrics[wandb_key] = converted
                            
#                             # 简化命名的指标
#                             for wandb_key, log_key in [
#                                 ('val_mF1', 'epoch_acc'),
#                                 ('val_loss', 'l_cd'),
#                                 ('val_mIoU', 'miou'),
#                                 ('val_accuracy', 'acc'),
#                             ]:
#                                 converted = safe_convert(logs.get(log_key), wandb_key)
#                                 if converted is not None:
#                                     validation_metrics[wandb_key] = converted
                            
#                             # 其他指标
#                             validation_metrics['epoch'] = current_epoch
#                             validation_metrics['validation_step'] = current_epoch
                            
#                             # best_mF1
#                             converted_best = safe_convert(best_mF1, 'val_best_mF1')
#                             if converted_best is not None:
#                                 validation_metrics['val_best_mF1'] = converted_best
#                                 validation_metrics['validation/best_mF1'] = converted_best
                            
#                             # 时间
#                             validation_metrics['validation/time_minutes'] = val_time / 60
                            
#                             print(f"\n📊 将要记录的指标 ({len(validation_metrics)}个):")
#                             for k, v in validation_metrics.items():
#                                 print(f"  {k}: {v}")
                            
#                             # 记录到WandB
#                             if validation_metrics:
#                                 wandb_logger.log_metrics(validation_metrics)
#                                 print(f"\n✅ WandB记录成功: {len(validation_metrics)}个指标")
#                             else:
#                                 print("\n❌ 没有有效指标可记录")
                            
#                             print("🔍 === WandB调试信息结束 ===\n")
                            
#                         except Exception as e:
#                             print(f"❌ WandB记录错误: {e}")
#                             import traceback
#                             traceback.print_exc()
                    
#                     # 模型保存逻辑保持不变
#                     if logs['epoch_acc'] > best_mF1:
#                         is_best_model = True
#                         best_mF1 = logs['epoch_acc']
#                         print(f"🎉 最佳模型更新! mF1: {best_mF1:.5f}")
#                         logger.info('[验证] 最佳模型更新，保存模型和训练状态')
#                     else:
#                         is_best_model = False
#                         logger.info('[验证] 保存当前模型和训练状态')

#                     change_detection.save_network(current_epoch, is_best_model=is_best_model)
                    
#                 except Exception as e:
#                     print(f"验证指标收集错误: {e}")
                
#                 change_detection._clear_cache()
#                 print(f"--- 进入下一个Epoch ---\n")

#             if wandb_logger:
#                 wandb_logger.log_metrics({'epoch': current_epoch})
            
#             # Epoch结束清理
#             torch.cuda.empty_cache()
                
#         print("🎉 训练完成!")
#         logger.info('训练结束')
        
#     else:
#         ##################
#         ### 测试阶段 ###
#         ##################
#         logger.info('开始模型评估（测试）')
#         print("🔍 开始测试...")
        
#         test_result_path = f'{opt["path"]["results"]}/test/'
#         os.makedirs(test_result_path, exist_ok=True)
#         logger_test = logging.getLogger('test')
#         change_detection._clear_cache()
        
#         for current_step, test_data in enumerate(test_loader):
#             with torch.no_grad():
#                 process_batch_efficiently(test_data, diffusion, change_detection, opt, "test")
#                 change_detection.test()
#                 change_detection._collect_running_batch_states()

#             # 测试日志
#             if current_step % max(1, len(test_loader) // 10) == 0:
#                 logs = change_detection.get_current_log()
#                 message = f'[测试] Step {current_step}/{len(test_loader)}, ' \
#                          f'mF1: {logs["running_acc"]:.5f}'
#                 print(message)
#                 logger_test.info(message)

#             # 保存测试结果
#             try:
#                 visuals = change_detection.get_current_visuals()
#                 visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
#                 visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                
#                 # 单独保存图像
#                 img_A = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))
#                 img_B = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))
#                 gt_cm = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
#                                           out_type=np.uint8, min_max=(0, 1))
#                 pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
#                                             out_type=np.uint8, min_max=(0, 1))

#                 Metrics.save_img(img_A, f'{test_result_path}/img_A_{current_step}.png')
#                 Metrics.save_img(img_B, f'{test_result_path}/img_B_{current_step}.png')
#                 Metrics.save_img(pred_cm, f'{test_result_path}/img_pred_cm{current_step}.png')
#                 Metrics.save_img(gt_cm, f'{test_result_path}/img_gt_cm{current_step}.png')
                
#             except Exception as e:
#                 print(f"测试保存失败: {e}")

#         ### 测试总结 ###
#         try:
#             change_detection._collect_epoch_states()
#             logs = change_detection.get_current_log()
            
#             message = f'[测试总结] mF1={logs["epoch_acc"]:.5f}\n'
#             for k, v in logs.items():
#                 message += f'{k}: {v:.4e} '
#             message += '\n'
            
#             print(f"✅ {message}")
#             logger_test.info(message)

#             if wandb_logger:
#                 wandb_logger.log_metrics({
#                     'test/mF1': logs['epoch_acc'],
#                     'test/mIoU': logs['miou'],
#                     'test/OA': logs['acc'],
#                     'test/change-F1': logs['F1_1'],
#                     'test/no-change-F1': logs['F1_0'],
#                     'test/change-IoU': logs['iou_1'],
#                     'test/no-change-IoU': logs['iou_0'],
#                 })
                
#         except Exception as e:
#             print(f"测试指标收集错误: {e}")

#         print("🎉 测试完成!")
#         logger.info('测试结束')

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist  # 分布式训练
# import torch.multiprocessing as mp  # 多进程
# from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式DataParallel
# from torch.utils.data.distributed import DistributedSampler  # 分布式采样器

# import data as Data
# import model as Model
# import argparse
# import logging
# import core.logger as Logger
# import core.metrics as Metrics
# from core.wandb_logger import WandbLogger
# from tensorboardX import SummaryWriter
# import os
# import numpy as np
# from model.cd_modules.cd_head import cd_head 
# from misc.print_diffuse_feats import print_feats
# import time
# import datetime
# from contextlib import contextmanager

# # ==================== DDP兼容性工具 ====================
# def safe_model_call(model, method_name, *args, **kwargs):
#     """安全调用被DDP/DP包装的模型方法"""
#     # 获取实际的模型
#     actual_model = model
#     if isinstance(model, (DDP, nn.DataParallel)):
#         actual_model = model.module
    
#     # 检查方法是否存在
#     if hasattr(actual_model, method_name):
#         method = getattr(actual_model, method_name)
#         return method(*args, **kwargs)
#     else:
#         raise AttributeError(f"模型没有方法 '{method_name}'")

# def get_actual_model(model):
#     """获取被DDP/DP包装的实际模型"""
#     if isinstance(model, (DDP, nn.DataParallel)):
#         return model.module
#     return model

# # ==================== 多GPU配置类 ====================
# class MultiGPUConfig:
#     """多GPU训练配置管理"""
#     def __init__(self, args, opt):
#         self.args = args
#         self.opt = opt
#         self.world_size = torch.cuda.device_count()
#         self.use_ddp = args.use_ddp if hasattr(args, 'use_ddp') else False
#         self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
#         self.rank = int(os.environ.get('RANK', 0))
        
#         # 根据GPU数量选择策略
#         if self.world_size <= 1:
#             self.strategy = 'single'
#         elif self.use_ddp:
#             self.strategy = 'ddp'
#         else:
#             self.strategy = 'dp'
        
#         print(f"🔧 多GPU配置: {self.strategy}, GPU数量: {self.world_size}")

# # ==================== 改进的分布式训练初始化 ====================
# def setup_distributed_training(rank, world_size, port='29500'):
#     """初始化分布式训练 - 改进版本"""
#     print(f"🔄 [GPU {rank}] 开始初始化分布式训练...")
    
#     try:
#         # 设置环境变量
#         os.environ['MASTER_ADDR'] = '127.0.0.1'  # 使用127.0.0.1而不是localhost
#         os.environ['MASTER_PORT'] = str(port)
#         os.environ['WORLD_SIZE'] = str(world_size)
#         os.environ['RANK'] = str(rank)
        
#         print(f"   📋 [GPU {rank}] 环境变量设置完成")
#         print(f"   📋 [GPU {rank}] MASTER_ADDR: {os.environ['MASTER_ADDR']}")
#         print(f"   📋 [GPU {rank}] MASTER_PORT: {os.environ['MASTER_PORT']}")
#         print(f"   📋 [GPU {rank}] WORLD_SIZE: {world_size}, RANK: {rank}")
        
#         # 设置CUDA设备
#         print(f"   🔄 [GPU {rank}] 设置CUDA设备...")
#         torch.cuda.set_device(rank)
#         print(f"   ✅ [GPU {rank}] CUDA设备设置为: {torch.cuda.current_device()}")
        
#         # 初始化进程组 - 添加更多错误处理
#         print(f"   🔄 [GPU {rank}] 初始化进程组...")
        
#         # 尝试不同的初始化方法
#         init_methods = [
#             f'tcp://127.0.0.1:{port}',
#             f'tcp://localhost:{port}',
#             'env://'
#         ]
        
#         success = False
#         for init_method in init_methods:
#             try:
#                 print(f"   🔄 [GPU {rank}] 尝试初始化方法: {init_method}")
                
#                 dist.init_process_group(
#                     backend='nccl',
#                     init_method=init_method,
#                     world_size=world_size,
#                     rank=rank,
#                     timeout=datetime.timedelta(seconds=30)  # 设置超时
#                 )
                
#                 print(f"   ✅ [GPU {rank}] 进程组初始化成功，使用方法: {init_method}")
#                 success = True
#                 break
                
#             except Exception as e:
#                 print(f"   ❌ [GPU {rank}] 初始化方法 {init_method} 失败: {e}")
#                 continue
        
#         if not success:
#             print(f"   ❌ [GPU {rank}] 所有初始化方法都失败")
#             return False
        
#         # 验证分布式设置
#         print(f"   🔄 [GPU {rank}] 验证分布式设置...")
#         if dist.is_initialized():
#             print(f"   ✅ [GPU {rank}] 分布式已初始化")
#             print(f"   📋 [GPU {rank}] World size: {dist.get_world_size()}")
#             print(f"   📋 [GPU {rank}] Rank: {dist.get_rank()}")
            
#             # 简单的通信测试
#             print(f"   🔄 [GPU {rank}] 进行通信测试...")
#             test_tensor = torch.tensor([rank], device=f'cuda:{rank}')
#             dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
#             expected_sum = sum(range(world_size))
            
#             if test_tensor.item() == expected_sum:
#                 print(f"   ✅ [GPU {rank}] 通信测试成功: {test_tensor.item()} == {expected_sum}")
#             else:
#                 print(f"   ❌ [GPU {rank}] 通信测试失败: {test_tensor.item()} != {expected_sum}")
#                 return False
#         else:
#             print(f"   ❌ [GPU {rank}] 分布式未正确初始化")
#             return False
        
#         print(f"✅ [GPU {rank}] 分布式训练初始化完成")
#         return True
        
#     except Exception as e:
#         print(f"❌ [GPU {rank}] 分布式训练初始化失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
    
# # ==================== 调试用的分布式状态检查 ====================
# def debug_distributed_state(rank):
#     """调试分布式状态"""
#     print(f"\n🔍 [GPU {rank}] === 分布式状态调试 ===")
    
#     # 检查环境变量
#     env_vars = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK']
#     for var in env_vars:
#         value = os.environ.get(var, 'Not set')
#         print(f"   {var}: {value}")
    
#     # 检查CUDA状态
#     print(f"   CUDA available: {torch.cuda.is_available()}")
#     print(f"   CUDA device count: {torch.cuda.device_count()}")
#     print(f"   Current CUDA device: {torch.cuda.current_device()}")
    
#     # 检查分布式状态
#     print(f"   Distributed initialized: {dist.is_initialized()}")
#     if dist.is_initialized():
#         print(f"   Backend: {dist.get_backend()}")
#         print(f"   World size: {dist.get_world_size()}")
#         print(f"   Rank: {dist.get_rank()}")
    
#     print(f"🔍 [GPU {rank}] === 调试信息结束 ===\n")

# # ==================== 带超时的安全DDP包装 ====================
# def safe_ddp_wrap_with_timeout(model, device_ids, output_device, timeout_seconds=60):
#     """带超时的安全DDP包装"""
#     import signal
#     import threading
    
#     result = [None]
#     exception = [None]
    
#     def wrap_model():
#         try:
#             wrapped_model = DDP(
#                 model,
#                 device_ids=device_ids,
#                 output_device=output_device,
#                 find_unused_parameters=False
#             )
#             result[0] = wrapped_model
#         except Exception as e:
#             exception[0] = e
    
#     # 在新线程中执行包装
#     thread = threading.Thread(target=wrap_model)
#     thread.start()
#     thread.join(timeout=timeout_seconds)
    
#     if thread.is_alive():
#         print(f"   ⚠️  DDP包装超时（{timeout_seconds}秒），可能存在死锁")
#         return None
    
#     if exception[0]:
#         print(f"   ❌ DDP包装异常: {exception[0]}")
#         return None
    
#     return result[0]

# def cleanup_distributed():
#     """清理分布式训练"""
#     if dist.is_initialized():
#         dist.destroy_process_group()

# # ==================== 多GPU模型包装器 ====================
# class MultiGPUModelWrapper:
#     """多GPU模型包装器"""
    
#     def __init__(self, config):
#         self.config = config
#         self.is_main_process = (config.rank == 0) if config.use_ddp else True
    
#     def wrap_diffusion_model(self, diffusion):
#         """包装扩散模型"""
#         print(f"🔧 包装扩散模型: {self.config.strategy}")
        
#         # 确保模型在正确的设备上
#         if torch.cuda.is_available():
#             if self.config.strategy == 'ddp':
#                 device = torch.device(f'cuda:{self.config.local_rank}')
#                 diffusion.netG = diffusion.netG.to(device)
                
#                 # 包装为DDP
#                 diffusion.netG = DDP(
#                     diffusion.netG, 
#                     device_ids=[self.config.local_rank],
#                     output_device=self.config.local_rank,
#                     find_unused_parameters=True
#                 )
#                 print(f"✅ [GPU {self.config.rank}] 扩散模型DDP包装完成")
                
#             elif self.config.strategy == 'dp' and self.config.world_size > 1:
#                 # 解包现有的DataParallel（如果存在）
#                 if isinstance(diffusion.netG, nn.DataParallel):
#                     diffusion.netG = diffusion.netG.module
                
#                 device_ids = list(range(self.config.world_size))
#                 diffusion.netG = diffusion.netG.cuda()
#                 diffusion.netG = nn.DataParallel(diffusion.netG, device_ids=device_ids)
#                 print(f"✅ 扩散模型DataParallel包装完成，使用GPU: {device_ids}")
                
#             else:
#                 # 单GPU
#                 diffusion.netG = diffusion.netG.cuda()
#                 print("✅ 扩散模型单GPU设置完成")
        
#         return diffusion
    
#     def wrap_change_detection_model(self, change_detection):
#         """包装变化检测模型 - 修复版本"""
#         print(f"🔧 包装变化检测模型: {self.config.strategy}")
        
#         if not hasattr(change_detection, 'netCD') or change_detection.netCD is None:
#             print("⚠️  变化检测模型不存在netCD属性")
#             return change_detection
        
#         if torch.cuda.is_available():
#             if self.config.strategy == 'ddp':
#                 device = torch.device(f'cuda:{self.config.local_rank}')
#                 print(f"   📍 [GPU {self.config.rank}] 目标设备: {device}")
                
#                 # 1. 首先移动模型到设备
#                 print(f"   🔄 [GPU {self.config.rank}] 移动模型到设备...")
#                 try:
#                     change_detection.netCD = change_detection.netCD.to(device)
#                     print(f"   ✅ [GPU {self.config.rank}] 模型移动完成")
#                 except Exception as e:
#                     print(f"   ❌ [GPU {self.config.rank}] 模型移动失败: {e}")
#                     return change_detection
                
#                 # 2. 处理优化器（如果存在且已有状态）
#                 print(f"   🔄 [GPU {self.config.rank}] 检查优化器...")
#                 if hasattr(change_detection, 'optCD') and change_detection.optCD is not None:
#                     # 只有当优化器有状态时才移动
#                     if len(change_detection.optCD.state) > 0:
#                         print(f"   🔄 [GPU {self.config.rank}] 移动优化器状态...")
#                         try:
#                             for state in change_detection.optCD.state.values():
#                                 for k, v in state.items():
#                                     if torch.is_tensor(v):
#                                         state[k] = v.to(device)
#                             print(f"   ✅ [GPU {self.config.rank}] 优化器状态移动完成")
#                         except Exception as e:
#                             print(f"   ⚠️  [GPU {self.config.rank}] 优化器状态移动失败: {e}")
#                     else:
#                         print(f"   ℹ️  [GPU {self.config.rank}] 优化器无状态，跳过移动")
                
#                 # 3. 添加同步点，确保所有进程都完成了模型移动
#                 print(f"   🔄 [GPU {self.config.rank}] 等待所有进程完成模型移动...")
#                 try:
#                     if dist.is_initialized():
#                         dist.barrier()
#                         print(f"   ✅ [GPU {self.config.rank}] 同步完成")
#                 except Exception as e:
#                     print(f"   ⚠️  [GPU {self.config.rank}] 同步失败: {e}")
                
#                 # 4. 包装为DDP - 使用更保守的参数
#                 print(f"   🔄 [GPU {self.config.rank}] 创建DDP包装...")
#                 try:
#                     change_detection.netCD = DDP(
#                         change_detection.netCD,
#                         device_ids=[self.config.local_rank],
#                         output_device=self.config.local_rank,
#                         find_unused_parameters=False,  # 改为False，避免额外检查
#                         broadcast_buffers=False,
#                         gradient_as_bucket_view=False,  # 优化内存使用
#                         static_graph = True
#                     )
#                     print(f"   ✅ [GPU {self.config.rank}] DDP包装完成")
#                 except Exception as e:
#                     print(f"   ❌ [GPU {self.config.rank}] DDP包装失败: {e}")
#                     print(f"   🔄 [GPU {self.config.rank}] 尝试基本DDP配置...")
#                     try:
#                         change_detection.netCD = DDP(
#                             change_detection.netCD,
#                             device_ids=[self.config.local_rank],
#                             output_device=self.config.local_rank
#                         )
#                         print(f"   ✅ [GPU {self.config.rank}] 基本DDP包装完成")
#                     except Exception as e2:
#                         print(f"   ❌ [GPU {self.config.rank}] 基本DDP也失败: {e2}")
#                         print(f"   🔄 [GPU {self.config.rank}] 回退到单GPU模式...")
#                         # 回退到单GPU
#                         return change_detection
                
#                 print(f"✅ [GPU {self.config.rank}] 变化检测模型DDP包装完成")
                
#             elif self.config.strategy == 'dp' and self.config.world_size > 1:
#                 # DataParallel包装
#                 if isinstance(change_detection.netCD, nn.DataParallel):
#                     change_detection.netCD = change_detection.netCD.module
                
#                 device_ids = list(range(self.config.world_size))
#                 change_detection.netCD = change_detection.netCD.cuda()
#                 change_detection.netCD = nn.DataParallel(change_detection.netCD, device_ids=device_ids)
#                 print(f"✅ 变化检测模型DataParallel包装完成，使用GPU: {device_ids}")
                
#             else:
#                 # 单GPU
#                 change_detection.netCD = change_detection.netCD.cuda()
#                 print("✅ 变化检测模型单GPU设置完成")
        
#         return change_detection

# # ==================== 多GPU数据加载器 ====================
# def create_multi_gpu_dataloader(dataset, dataset_opt, phase, config):
#     """创建多GPU数据加载器"""
    
#     if config.strategy == 'ddp':
#         # DDP需要分布式采样器
#         sampler = DistributedSampler(
#             dataset,
#             num_replicas=config.world_size,
#             rank=config.rank,
#             shuffle=(phase == 'train')
#         )
        
#         # 调整batch size（每个GPU的batch size）
#         batch_size = dataset_opt['batch_size'] // config.world_size
#         if batch_size == 0:
#             batch_size = 1
#             print(f"⚠️  batch_size太小，每个GPU使用batch_size=1")
        
#         dataloader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             sampler=sampler,
#             num_workers=dataset_opt.get('num_workers', 4),
#             pin_memory=True,
#             drop_last=(phase == 'train')
#         )
        
#         print(f"✅ [GPU {config.rank}] DDP数据加载器创建完成，batch_size={batch_size}")
        
#     else:
#         # DataParallel或单GPU使用标准数据加载器
#         if config.strategy == 'dp' and config.world_size > 1:
#             # DataParallel可以增加batch size
#             batch_size = dataset_opt['batch_size'] * config.world_size
#             print(f"✅ DataParallel增加batch_size: {dataset_opt['batch_size']} → {batch_size}")
#         else:
#             batch_size = dataset_opt['batch_size']
        
#         dataloader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=(phase == 'train'),
#             num_workers=dataset_opt.get('num_workers', 4),
#             pin_memory=True,
#             drop_last=(phase == 'train')
#         )
    
#     return dataloader, sampler if config.strategy == 'ddp' else None

# # ==================== 多GPU性能监控 ====================
# class MultiGPUPerformanceMonitor:
#     """多GPU性能监控"""
    
#     def __init__(self, config):
#         self.config = config
#         self.step_times = []
#         self.memory_usage = []
#         self.start_time = time.time()
#         self.is_main_process = (config.rank == 0) if config.use_ddp else True
    
#     def log_step(self, step_time, memory_mb=None):
#         if self.is_main_process:
#             self.step_times.append(step_time)
#             if memory_mb:
#                 self.memory_usage.append(memory_mb)
    
#     def get_stats(self):
#         if not self.is_main_process or not self.step_times:
#             return ""
        
#         avg_time = np.mean(self.step_times[-100:])
#         total_time = time.time() - self.start_time
        
#         stats = f"平均步时: {avg_time:.2f}s, 总时间: {total_time/60:.1f}min"
        
#         if self.memory_usage:
#             avg_memory = np.mean(self.memory_usage[-10:])
#             stats += f", GPU{self.config.local_rank}显存: {avg_memory:.1f}MB"
        
#         # 添加多GPU信息
#         if self.config.world_size > 1:
#             stats += f" [策略: {self.config.strategy.upper()}, {self.config.world_size}GPU]"
        
#         return stats

# # ==================== 多GPU安全训练步骤 ====================
# def safe_multi_gpu_training_step(change_detection, config):
#     """多GPU安全训练步骤"""
#     try:
#         change_detection.optimize_parameters()
#         change_detection._collect_running_batch_states()
        
#         # DDP需要同步
#         if config.strategy == 'ddp':
#             dist.barrier()
        
#         return True
        
#     except RuntimeError as e:
#         if "device-side assert triggered" in str(e) or "CUDA" in str(e):
#             if config.rank == 0 or not config.use_ddp:
#                 print(f"⚠️  [GPU {config.local_rank}] CUDA错误已自动处理: {str(e)[:100]}...")
#             torch.cuda.empty_cache()
#             return False
#         else:
#             raise
#     except Exception as e:
#         if config.rank == 0 or not config.use_ddp:
#             print(f"❌ [GPU {config.local_rank}] 训练步骤错误: {e}")
#         return False

# # ==================== 多GPU模型保存和加载 ====================
# class MultiGPUCheckpointManager:
#     """多GPU checkpoint管理器"""
    
#     def __init__(self, config):
#         self.config = config
#         self.is_main_process = (config.rank == 0) if config.use_ddp else True
    
#     def save_checkpoint(self, change_detection, epoch, is_best_model=False):
#         """保存checkpoint"""
#         if not self.is_main_process:
#             return
        
#         try:
#             # 获取原始模型（去除DDP/DP包装）
#             if hasattr(change_detection, 'netCD'):
#                 if isinstance(change_detection.netCD, (DDP, nn.DataParallel)):
#                     model_state_dict = change_detection.netCD.module.state_dict()
#                 else:
#                     model_state_dict = change_detection.netCD.state_dict()
#             else:
#                 print("⚠️  无法找到netCD模型")
#                 return
            
#             # 保存路径
#             checkpoint_dir = os.path.expanduser("~/bowen/DDPM-CD/ddpm-cd/checkpoint/gvlm_cd_ddpm")
#             os.makedirs(checkpoint_dir, exist_ok=True)
            
#             # 保存模型
#             gen_file = os.path.join(checkpoint_dir, f"cd_model_E{epoch}_gen.pth")
#             torch.save(model_state_dict, gen_file)
            
#             # 保存优化器
#             if hasattr(change_detection, 'optCD') and change_detection.optCD is not None:
#                 opt_file = os.path.join(checkpoint_dir, f"cd_model_E{epoch}_opt.pth")
#                 torch.save(change_detection.optCD.state_dict(), opt_file)
            
#             # 保存最佳模型
#             if is_best_model:
#                 best_gen_file = os.path.join(checkpoint_dir, "best_cd_model_gen.pth")
#                 torch.save(model_state_dict, best_gen_file)
                
#                 if hasattr(change_detection, 'optCD') and change_detection.optCD is not None:
#                     best_opt_file = os.path.join(checkpoint_dir, "best_cd_model_opt.pth")
#                     torch.save(change_detection.optCD.state_dict(), best_opt_file)
                
#                 print(f"🏆 [主进程] 最佳模型已保存")
            
#             print(f"💾 [主进程] Checkpoint已保存: Epoch {epoch}")
            
#         except Exception as e:
#             print(f"❌ [主进程] 保存checkpoint失败: {e}")
    
#     def load_checkpoint(self, change_detection, opt):
#         """加载checkpoint"""
#         if self.is_main_process:
#             # 主进程加载
#             start_epoch, best_mF1 = load_checkpoint_if_exists(change_detection, opt)
#         else:
#             # 非主进程等待
#             start_epoch, best_mF1 = 0, 0.0
        
#         # DDP需要同步加载状态
#         if self.config.strategy == 'ddp':
#             # 等待所有进程
#             dist.barrier()
            
#             # 广播加载状态（从主进程到所有进程）
#             state = torch.tensor([start_epoch, best_mF1], device=f'cuda:{self.config.local_rank}')
#             dist.broadcast(state, src=0)
#             start_epoch, best_mF1 = int(state[0].item()), float(state[1].item())
            
#             print(f"📡 [GPU {self.config.rank}] 同步checkpoint状态: epoch={start_epoch}, best_mF1={best_mF1:.4f}")
        
#         return start_epoch, best_mF1

# # ==================== 多GPU训练主函数 ====================
# def train_multi_gpu(rank, world_size, args, opt):
#     """多GPU训练主函数"""
#     # 1. 添加调试信息
#     print(f"\n🚀 [GPU {rank}] 开始多GPU训练，世界大小: {world_size}")
    
#     # 配置多GPU
#     config = MultiGPUConfig(args, opt)
#     config.rank = rank
#     config.local_rank = rank
#     config.world_size = world_size
    
#     # 2. 调试分布式状态
#     debug_distributed_state(rank)
    
#     # 3. 初始化分布式训练
#     if config.strategy == 'ddp':
#         print(f"🔄 [GPU {rank}] 初始化DDP...")
#         if not setup_distributed_training(rank, world_size):
#             print(f"❌ [GPU {rank}] DDP初始化失败，退出")
#             return
        
#         # 再次调试状态
#         debug_distributed_state(rank)
    
#     try:
#         # 设置日志（只在主进程）
#         if config.rank == 0:
#             Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
#             Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
#             logger = logging.getLogger('base')
#             logger.info(Logger.dict2str(opt))
#             tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
#         else:
#             logger = None
#             tb_logger = None
        
#         # 初始化WandbLogger（只在主进程）
#         wandb_logger = None
#         if opt['enable_wandb'] and config.rank == 0:
#             import wandb
#             print("Initializing wandblog.")
#             wandb_logger = WandbLogger(opt)
#             wandb.define_metric('epoch')
#             wandb.define_metric('training/train_step')
#             wandb.define_metric("training/*", step_metric="train_step")
#             wandb.define_metric('validation/val_step')
#             wandb.define_metric("validation/*", step_metric="val_step")
        
#         # 创建数据集和数据加载器
#         print(f"🔄 [GPU {rank}] 加载数据集...")
#         train_loader = val_loader = test_loader = None
#         train_sampler = val_sampler = test_sampler = None
        
#         for phase, dataset_opt in opt['datasets'].items():
#             if phase == 'train' and args.phase != 'test':
#                 print(f"[GPU {rank}] Creating [train] change-detection dataloader.")
#                 train_set = Data.create_cd_dataset(dataset_opt, phase)
#                 train_loader, train_sampler = create_multi_gpu_dataloader(train_set, dataset_opt, phase, config)
#                 opt['len_train_dataloader'] = len(train_loader)

#             elif phase == 'val' and args.phase != 'test':
#                 print(f"[GPU {rank}] Creating [val] change-detection dataloader.")
#                 val_set = Data.create_cd_dataset(dataset_opt, phase)
#                 val_loader, val_sampler = create_multi_gpu_dataloader(val_set, dataset_opt, phase, config)
#                 opt['len_val_dataloader'] = len(val_loader)
            
#             elif phase == 'test' and args.phase == 'test':
#                 print(f"[GPU {rank}] Creating [test] change-detection dataloader.")
#                 test_set = Data.create_cd_dataset(dataset_opt, phase)
#                 test_loader, test_sampler = create_multi_gpu_dataloader(test_set, dataset_opt, phase, config)
#                 opt['len_test_dataloader'] = len(test_loader)
        
#         if config.rank == 0:
#             logger.info('Initial Dataset Finished')
        
#         # 加载扩散模型
#         print(f"🔄 [GPU {rank}] 加载扩散模型...")
#         diffusion = Model.create_model(opt)
        
#         # ⚠️ 重要：先设置噪声调度，再进行多GPU包装
#         print(f"🔄 [GPU {rank}] 设置噪声调度...")
#         diffusion.set_new_noise_schedule(
#             opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
        
#         # 多GPU包装器
#         model_wrapper = MultiGPUModelWrapper(config)
#         diffusion = model_wrapper.wrap_diffusion_model(diffusion)
        
#         if config.rank == 0:
#             logger.info('Initial Diffusion Model Finished')
        
#         # 创建变化检测模型
#         print(f"🔄 [GPU {rank}] 加载变化检测模型...")
#         change_detection = Model.create_CD_model(opt)
#         change_detection = model_wrapper.wrap_change_detection_model(change_detection)
        
#         # 设置训练优化
#         use_amp = setup_training_optimization(diffusion, change_detection)
        
#         # 创建多GPU性能监控器
#         performance_monitor = MultiGPUPerformanceMonitor(config)
        
#         # 创建checkpoint管理器
#         checkpoint_manager = MultiGPUCheckpointManager(config)
        
#         if config.rank == 0:
#             print("🚀 所有设置完成，开始多GPU训练...\n")
        
#         # 训练循环
#         n_epoch = opt['train']['n_epoch']
#         start_epoch, best_mF1 = checkpoint_manager.load_checkpoint(change_detection, opt)
        
#         if opt['phase'] == 'train':
#             for current_epoch in range(start_epoch, n_epoch):
#                 epoch_start_time = time.time()
                
#                 # 设置epoch（用于DDP采样器）
#                 if config.strategy == 'ddp' and train_sampler is not None:
#                     train_sampler.set_epoch(current_epoch)
                
#                 change_detection._clear_cache()
                
#                 if config.rank == 0:
#                     train_result_path = f'{opt["path"]["results"]}/train/{current_epoch}'
#                     os.makedirs(train_result_path, exist_ok=True)
                
#                 ################
#                 ### 训练阶段 ###
#                 ################
#                 if config.rank == 0:
#                     print(f"\n🎯 开始训练 Epoch {current_epoch}/{n_epoch-1}")
#                     message = f'学习率: {change_detection.optCD.param_groups[0]["lr"]:.7f}'
#                     logger.info(message)
                
#                 for current_step, train_data in enumerate(train_loader):
#                     step_start_time = time.time()
                    
#                     # 高效批量处理
#                     process_batch_efficiently(train_data, diffusion, change_detection, opt, "train")
                    
#                     # 多GPU安全训练步骤
#                     success = safe_multi_gpu_training_step(change_detection, config)
                    
#                     if not success:
#                         if config.rank == 0:
#                             print(f"跳过步骤 {current_step}")
#                         continue
                    
#                     # 记录性能
#                     step_time = time.time() - step_start_time
#                     memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
#                     performance_monitor.log_step(step_time, memory_mb)
                    
#                     # 优化的日志输出（只在主进程）
#                     if config.rank == 0:
#                         optimized_logging(current_step, current_epoch, n_epoch, train_loader, 
#                                         change_detection, logger, opt, "train", performance_monitor)
                    
#                     # 保存可视化结果（只在主进程）
#                     if config.rank == 0:
#                         save_freq = max(opt['train']['train_print_freq'] * 2, len(train_loader) // 5)
#                         if current_step % save_freq == 0:
#                             try:
#                                 visuals = change_detection.get_current_visuals()
                                
#                                 device = train_data['A'].device
#                                 visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
#                                 visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                                
#                                 pred_cm_expanded = visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
#                                 gt_cm_expanded = visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                                
#                                 grid_img = torch.cat((train_data['A'], train_data['B'], 
#                                                     pred_cm_expanded, gt_cm_expanded), dim=0)
#                                 grid_img = Metrics.tensor2img(grid_img)
                                
#                                 Metrics.save_img(grid_img, 
#                                     f'{train_result_path}/img_A_B_pred_gt_e{current_epoch}_b{current_step}.png')
#                             except Exception as e:
#                                 print(f"保存可视化失败: {e}")
                    
#                     # 定期内存清理
#                     if current_step % 50 == 0:
#                         torch.cuda.empty_cache()
                
#                 # DDP同步
#                 if config.strategy == 'ddp':
#                     dist.barrier()
                
#                 ### 训练epoch总结 ###
#                 if config.rank == 0:
#                     try:
#                         change_detection._collect_epoch_states()
#                         logs = change_detection.get_current_log()
                        
#                         epoch_time = time.time() - epoch_start_time
#                         message = f'[训练 Epoch {current_epoch}] mF1={logs["epoch_acc"]:.5f}, ' \
#                                  f'用时={epoch_time/60:.1f}分钟'
                        
#                         print(f"\n✅ {message}")
#                         logger.info(message)
                        
#                         # 详细指标
#                         for k, v in logs.items():
#                             tb_logger.add_scalar(f'train/{k}', v, current_epoch)
                        
#                         if wandb_logger:
#                             wandb_logger.log_metrics({
#                                 'training/mF1': logs['epoch_acc'],
#                                 'training/mIoU': logs['miou'],
#                                 'training/OA': logs['acc'],
#                                 'training/change-F1': logs['F1_1'],
#                                 'training/no-change-F1': logs['F1_0'],
#                                 'training/change-IoU': logs['iou_1'],
#                                 'training/no-change-IoU': logs['iou_0'],
#                                 'training/train_step': current_epoch,
#                                 'training/loss': logs.get('l_cd'),
#                             })
                            
#                     except Exception as e:
#                         print(f"训练指标收集错误: {e}")
                
#                 change_detection._clear_cache()
#                 change_detection._update_lr_schedulers()
                
#                 ##################
#                 ### 验证阶段 ###
#                 ##################
#                 if current_epoch % opt['train']['val_freq'] == 0:
#                     if config.rank == 0:
#                         print(f"\n🔍 开始验证 Epoch {current_epoch}")
#                         val_result_path = f'{opt["path"]["results"]}/val/{current_epoch}'
#                         os.makedirs(val_result_path, exist_ok=True)
                    
#                     val_start_time = time.time()

#                     for current_step, val_data in enumerate(val_loader):
#                         with torch.no_grad():
#                             process_batch_efficiently(val_data, diffusion, change_detection, opt, "val")
#                             change_detection.test()
#                             change_detection._collect_running_batch_states()
                        
#                         # 验证日志（只在主进程）
#                         if config.rank == 0:
#                             optimized_logging(current_step, current_epoch, n_epoch, val_loader, 
#                                             change_detection, logger, opt, "val")
                    
#                     # DDP同步
#                     if config.strategy == 'ddp':
#                         dist.barrier()
                    
#                     ### 验证总结（只在主进程）###
#                     if config.rank == 0:
#                         try:
#                             change_detection._collect_epoch_states()
#                             logs = change_detection.get_current_log()
                            
#                             val_time = time.time() - val_start_time
#                             message = f'[验证 Epoch {current_epoch}] mF1={logs["epoch_acc"]:.5f}, ' \
#                                     f'用时={val_time/60:.1f}分钟'
                            
#                             print(f"✅ {message}")
#                             logger.info(message)
                            
#                             for k, v in logs.items():
#                                 tb_logger.add_scalar(f'val/{k}', v, current_epoch)

#                             if wandb_logger:
#                                 wandb_logger.log_metrics({
#                                     'validation/mF1': logs['epoch_acc'],
#                                     'validation/mIoU': logs['miou'],
#                                     'validation/OA': logs['acc'],
#                                     'validation/change-F1': logs['F1_1'],
#                                     'validation/no-change-F1': logs['F1_0'],
#                                     'validation/change-IoU': logs['iou_1'],
#                                     'validation/no-change-IoU': logs['iou_0'],
#                                     'epoch': current_epoch,
#                                 })
                            
#                             # 模型保存逻辑
#                             if logs['epoch_acc'] > best_mF1:
#                                 is_best_model = True
#                                 best_mF1 = logs['epoch_acc']
#                                 print(f"🎉 最佳模型更新! mF1: {best_mF1:.5f}")
#                                 logger.info('[验证] 最佳模型更新，保存模型和训练状态')
#                             else:
#                                 is_best_model = False
#                                 logger.info('[验证] 保存当前模型和训练状态')

#                             # 保存checkpoint
#                             checkpoint_manager.save_checkpoint(change_detection, current_epoch, is_best_model)
                            
#                         except Exception as e:
#                             print(f"验证指标收集错误: {e}")
                    
#                     change_detection._clear_cache()
#                     if config.rank == 0:
#                         print(f"--- 进入下一个Epoch ---\n")

#                 if wandb_logger and config.rank == 0:
#                     wandb_logger.log_metrics({'epoch': current_epoch})
                
#                 # Epoch结束清理
#                 torch.cuda.empty_cache()
            
#             if config.rank == 0:
#                 print("🎉 多GPU训练完成!")
#                 logger.info('多GPU训练结束')
    
#     finally:
#         # 清理分布式训练
#         if config.strategy == 'ddp':
#             cleanup_distributed()

# # ==================== 原有函数保持不变 ====================

# # ==================== 优化版标签验证器 ====================
# class LabelValidator:
#     """高效标签验证器 - 单例模式"""
#     _instance = None
#     _initialized = False
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def __init__(self):
#         if not self._initialized:
#             self.is_normalized = None
#             self.validation_done = False
#             self.label_stats = {}
#             LabelValidator._initialized = True
    
#     def validate_and_fix_labels(self, data, phase="train"):
#         """
#         高效标签验证 - 只在第一次详细检查，后续快速处理
#         """
#         if 'L' not in data:
#             return False
        
#         labels = data['L']
        
#         # 快速通道：如果已经验证过，直接处理
#         if self.validation_done:
#             if self.is_normalized:
#                 data['L'] = (labels >= 0.5).long()
#             else:
#                 fixed_labels = labels.clone()
#                 unique_vals = torch.unique(labels)
#                 if 255 in unique_vals:
#                     fixed_labels[labels == 255] = 1
#                 data['L'] = torch.clamp(fixed_labels, 0, 1).long()
#             return True
        
#         # 第一次详细验证
#         unique_vals = torch.unique(labels)
#         min_val = labels.min().item()
#         max_val = labels.max().item()
        
#         print(f"\n🔍 [{phase}] 标签验证（仅显示一次）:")
#         print(f"   形状: {labels.shape}, 数据类型: {labels.dtype}")
#         print(f"   值范围: [{min_val}, {max_val}]")
#         print(f"   唯一值: {unique_vals.tolist()}")
        
#         # 判断标签类型
#         self.is_normalized = (max_val <= 1.0 and min_val >= 0.0)
        
#         if self.is_normalized:
#             print(f"   🔧 检测到归一化标签，使用阈值二值化（阈值=0.5）")
#             fixed_labels = (labels >= 0.5).long()
#         else:
#             print(f"   🔧 检测到标准标签，映射255→1")
#             fixed_labels = labels.clone()
#             if 255 in unique_vals:
#                 fixed_labels[labels == 255] = 1
#             fixed_labels = torch.clamp(fixed_labels, 0, 1).long()
        
#         # 验证修复结果
#         final_unique = torch.unique(fixed_labels)
#         zero_count = (fixed_labels == 0).sum().item()
#         one_count = (fixed_labels == 1).sum().item()
#         total = zero_count + one_count
        
#         print(f"   ✅ 修复完成: 唯一值{final_unique.tolist()}")
#         print(f"   📊 像素分布: 无变化={100*zero_count/total:.1f}%, 有变化={100*one_count/total:.1f}%")
#         print(f"   ✅ 标签验证设置完成，后续批次将快速处理\n")
        
#         # 保存统计信息
#         self.label_stats = {
#             'zero_ratio': zero_count / total,
#             'one_ratio': one_count / total,
#             'is_normalized': self.is_normalized
#         }
        
#         data['L'] = fixed_labels
#         self.validation_done = True
#         return True

# # 全局标签验证器
# label_validator = LabelValidator()

# # ==================== 内存管理工具 ====================
# @contextmanager
# def memory_efficient_context():
#     """内存管理上下文"""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     try:
#         yield
#     finally:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

# # ==================== 优化版特征重排 ====================
# def apply_feature_reordering_optimized(f_A, f_B, train_data, change_detection, opt, phase="train"):
#     """
#     内存优化的特征重排方案
#     """
#     try:
#         feat_scales = opt['model_cd']['feat_scales']  # [2, 5, 8, 11, 14]
#         cd_expected_order = sorted(feat_scales, reverse=True)  # [14, 11, 8, 5, 2]
        
#         # 只在第一次显示信息
#         if not hasattr(apply_feature_reordering_optimized, '_logged'):
#             print("🎯 使用优化的特征重排方案")
#             print("   保持原始多尺度配置的完整语义")
#             for i, scale in enumerate(cd_expected_order):
#                 print(f"     Block{i}: 使用layer{scale}特征")
#             apply_feature_reordering_optimized._logged = True
        
#         # 高效重排：直接在原地修改
#         reordered_f_A = []
#         reordered_f_B = []
        
#         for fa, fb in zip(f_A, f_B):
#             if isinstance(fa, list) and len(fa) > max(feat_scales):
#                 timestep_A = [fa[scale] for scale in cd_expected_order]
#                 timestep_B = [fb[scale] for scale in cd_expected_order]
#                 reordered_f_A.append(timestep_A)
#                 reordered_f_B.append(timestep_B)
#             else:
#                 raise ValueError(f"特征格式错误: 期望list长度>{max(feat_scales)}, 实际{type(fa)}")
        
#         # 清理原始特征释放内存
#         del f_A, f_B
        
#         # 使用重排后的特征
#         change_detection.feed_data(reordered_f_A, reordered_f_B, train_data)
        
#         # 清理重排后的特征
#         del reordered_f_A, reordered_f_B
        
#         return True
        
#     except Exception as e:
#         print(f"❌ 特征重排失败: {e}")
#         print("🔄 使用回退方案...")
        
#         # 简化回退方案
#         target_layers = [12, 13, 14]
#         corrected_f_A = []
#         corrected_f_B = []
        
#         for fa, fb in zip(f_A, f_B):
#             timestep_A = [fa[i] for i in target_layers if i < len(fa)]
#             timestep_B = [fb[i] for i in target_layers if i < len(fb)]
#             corrected_f_A.append(timestep_A)
#             corrected_f_B.append(timestep_B)
        
#         del f_A, f_B
#         change_detection.feed_data(corrected_f_A, corrected_f_B, train_data)
#         del corrected_f_A, corrected_f_B
        
#         return False

# # ==================== 训练优化设置 ====================
# def setup_training_optimization(diffusion, change_detection):
#     """设置训练优化"""
#     print("🚀 设置训练优化...")
    
#     # 启用CUDA优化
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False
    
#     # 检查混合精度支持
#     use_amp = False
#     if torch.cuda.is_available():
#         try:
#             from torch.cuda.amp import autocast, GradScaler
#             use_amp = True
#             print("   ✅ 支持混合精度训练")
#         except ImportError:
#             print("   ⚠️  不支持混合精度训练")
    
#     # 设置diffusion模型为eval模式（如果不需要训练）
#     if hasattr(diffusion.netG, 'eval'):
#         diffusion.netG.eval()
#         print("   ✅ Diffusion模型设置为eval模式")
    
#     # 检查多GPU设置
#     if torch.cuda.device_count() > 1:
#         print(f"   ✅ 检测到{torch.cuda.device_count()}个GPU")
        
#         # 显示GPU状态
#         for i in range(torch.cuda.device_count()):
#             props = torch.cuda.get_device_properties(i)
#             memory_gb = props.total_memory / 1024**3
#             print(f"     GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    
#     print("🚀 训练优化设置完成\n")
    
#     return use_amp

# # ==================== 批量处理优化 ====================
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

# # ==================== 优化的日志管理 ====================
# def optimized_logging(current_step, current_epoch, n_epoch, loader, change_detection, 
#                      logger, opt, phase="train", performance_monitor=None):
#     """优化的日志输出"""
#     # 动态调整日志频率
#     if phase == "train":
#         base_freq = opt['train'].get('train_print_freq', 10)
#         log_freq = max(base_freq, len(loader) // 1000)  # 至少每0.1%显示一次
#     else:
#         log_freq = max(1, len(loader) // 1000)  # 测试时每0.1%显示一次
    
#     if current_step % log_freq == 0:
#         try:
#             logs = change_detection.get_current_log()
            
#             # 基础信息
#             progress = f"[{current_epoch}/{n_epoch-1}]"
#             step_info = f"Step {current_step}/{len(loader)}"
#             metrics = f"Loss: {logs.get('l_cd', 0):.5f} mF1: {logs.get('running_acc', 0):.5f}"
            
#             # 性能信息
#             perf_info = ""
#             if performance_monitor:
#                 perf_info = f" | {performance_monitor.get_stats()}"
            
#             message = f"{progress} {step_info} {metrics}{perf_info}"
#             print(message)
            
#         except Exception as e:
#             print(f"日志输出错误: {e}")

# # ==================== checkpoint恢复 ====================
# def load_checkpoint_if_exists(change_detection, opt):
#     """修正版本：优先best模型，修复加载接口"""
    
#     # 🎯 指定的checkpoint路径
#     checkpoint_dir = os.path.expanduser("~/bowen/DDPM-CD/ddpm-cd/checkpoint/gvlm_cd_ddpm")
    
#     if not os.path.exists(checkpoint_dir):
#         print(f"🔍 Checkpoint目录不存在: {checkpoint_dir}")
#         return 0, 0.0
    
#     print(f"🔍 检查checkpoint目录: {checkpoint_dir}")
    
#     import glob
#     import re
    
#     # ========================================
#     # 🥇 第一优先级：检查最佳模型
#     # ========================================
#     best_gen_file = os.path.join(checkpoint_dir, "best_cd_model_gen.pth")
#     best_opt_file = os.path.join(checkpoint_dir, "best_cd_model_opt.pth")
    
#     if os.path.exists(best_gen_file):
#         print("🏆 发现最佳模型，优先使用最佳模型")
#         print(f"   最佳模型文件: {best_gen_file}")
#         print(f"   最佳优化器文件: {best_opt_file}")
        
#         success = load_model_safe(change_detection, best_gen_file, best_opt_file)
        
#         if success:
#             print("✅ 最佳模型加载成功")
#             # 从最佳模型开始，可以设置一个较高的epoch或从0开始
#             return 0, 0.8  # 从epoch 0开始，但best_mF1设置较高值表示这是好模型
#         else:
#             print("❌ 最佳模型加载失败，尝试最新epoch模型")
    
#     # ========================================
#     # 🥈 第二优先级：查找最新epoch模型
#     # ========================================
#     gen_files = glob.glob(os.path.join(checkpoint_dir, "cd_model_E*_gen.pth"))
    
#     if gen_files:
#         print(f"🔍 找到的epoch模型文件: {[os.path.basename(f) for f in gen_files]}")
        
#         # 提取epoch数字并排序
#         epochs = []
#         for f in gen_files:
#             match = re.search(r'cd_model_E(\d+)_gen\.pth', f)
#             if match:
#                 epochs.append(int(match.group(1)))
        
#         if epochs:
#             latest_epoch = max(epochs)
            
#             # 构建文件路径
#             gen_file = os.path.join(checkpoint_dir, f"cd_model_E{latest_epoch}_gen.pth")
#             opt_file = os.path.join(checkpoint_dir, f"cd_model_E{latest_epoch}_opt.pth")
            
#             print(f"🔄 使用最新epoch模型: Epoch {latest_epoch}")
#             print(f"   模型文件: {gen_file}")
#             print(f"   优化器文件: {opt_file}")
            
#             success = load_model_safe(change_detection, gen_file, opt_file)
            
#             if success:
#                 print("✅ 最新epoch模型加载成功")
                
#                 # 检查best_mF1（可以从某个记录文件读取，或设置默认值）
#                 best_mF1 = 0.0
#                 if os.path.exists(best_gen_file):
#                     best_mF1 = 0.5  # 如果有best文件但加载失败，设置一个中等值
                
#                 return latest_epoch + 1, best_mF1
#             else:
#                 print("❌ 最新epoch模型也加载失败")
    
#     print("🆕 没有找到可用的checkpoint，从头开始训练")
#     return 0, 0.0


# def load_model_safe(change_detection, gen_file, opt_file):
#     """安全的模型加载方法 - 尝试多种加载方式"""
    
#     if not os.path.exists(gen_file):
#         print(f"❌ 模型文件不存在: {gen_file}")
#         return False
    
#     print(f"🔄 尝试加载模型: {os.path.basename(gen_file)}")
    
#     # ========================================
#     # 方法1: 直接torch.load + 手动设置state_dict
#     # ========================================
#     try:
#         print("   🔄 方法1: 直接torch.load")
#         checkpoint = torch.load(gen_file, map_location='cpu')
        
#         # 获取实际的模型（处理DDP/DP包装）
#         model = change_detection.netCD
#         if isinstance(model, (DDP, nn.DataParallel)):
#             model = model.module
        
#         if model is not None:
#             # 检查checkpoint结构
#             if isinstance(checkpoint, dict):
#                 print(f"   📋 Checkpoint keys: {list(checkpoint.keys())}")
                
#                 # 尝试不同的key
#                 state_dict = None
#                 for key in ['model_state_dict', 'state_dict', 'model', 'netCD']:
#                     if key in checkpoint:
#                         state_dict = checkpoint[key]
#                         print(f"   ✅ 使用key: {key}")
#                         break
                
#                 if state_dict is None:
#                     # 直接作为state_dict
#                     state_dict = checkpoint
#                     print("   ✅ 直接作为state_dict")
#             else:
#                 state_dict = checkpoint
#                 print("   ✅ Checkpoint是state_dict")
            
#             # 加载state_dict
#             model.load_state_dict(state_dict, strict=False)
#             print("   ✅ 方法1: 模型权重加载成功")
            
#             # 尝试加载优化器
#             load_optimizer_safe(change_detection, opt_file)
            
#             return True
            
#     except Exception as e:
#         print(f"   ❌ 方法1失败: {e}")
    
#     # ========================================
#     # 方法2: 尝试无参数load_network (设置路径)
#     # ========================================
#     try:
#         print("   🔄 方法2: 无参数load_network")
        
#         # 尝试设置路径到opt中
#         if hasattr(change_detection, 'opt'):
#             # 备份原路径
#             original_path = change_detection.opt.get('path', {}).get('resume_state', '')
            
#             # 设置新路径
#             if 'path' not in change_detection.opt:
#                 change_detection.opt['path'] = {}
#             change_detection.opt['path']['resume_state'] = gen_file
            
#             # 尝试加载
#             change_detection.load_network()
            
#             # 恢复原路径
#             if original_path:
#                 change_detection.opt['path']['resume_state'] = original_path
            
#             print("   ✅ 方法2: 加载成功")
#             load_optimizer_safe(change_detection, opt_file)
#             return True
            
#     except Exception as e:
#         print(f"   ❌ 方法2失败: {e}")
    
#     print("   ❌ 所有加载方法都失败")
#     return False


# def load_optimizer_safe(change_detection, opt_file):
#     """安全的优化器加载"""
#     if not os.path.exists(opt_file):
#         print("   ⚠️  优化器文件不存在")
#         return False

#     try:
#         print(f"   🔄 加载优化器: {os.path.basename(opt_file)}")
#         opt_state = torch.load(opt_file, map_location='cpu')

#         optimizer = None
#         for attr_name in ['optCD', 'optimizer', 'opt_CD', 'optim']:
#             if hasattr(change_detection, attr_name):
#                 optimizer = getattr(change_detection, attr_name)
#                 if optimizer is not None:
#                     print(f"   📋 找到优化器属性: {attr_name}")
#                     break

#         if optimizer is not None:
#             print(f"   ℹ️  opt_state type: {type(opt_state)}")
#             actual_state_to_load = None
#             if isinstance(opt_state, dict):
#                 print(f"   ℹ️  opt_state keys: {list(opt_state.keys())}")
#                 if 'optimizer' in opt_state: # Primary case based on your log
#                     actual_state_to_load = opt_state['optimizer']
#                     print(f"   ℹ️  使用 opt_state['optimizer'] 进行加载")
#                 elif 'state_dict' in opt_state: # Fallback for another common pattern
#                     actual_state_to_load = opt_state['state_dict']
#                     print(f"   ℹ️  使用 opt_state['state_dict'] 进行加载")
#                 else: # Fallback: opt_state itself might be the state_dict
#                     actual_state_to_load = opt_state
#                     print(f"   ℹ️  直接使用 opt_state 进行加载")
#             else: # opt_state is not a dict, assume it's the state_dict
#                 actual_state_to_load = opt_state
#                 print(f"   ℹ️  opt_state 不是字典，直接使用 opt_state 进行加载")

#             if actual_state_to_load is not None:
#                 optimizer.load_state_dict(actual_state_to_load)
#                 print("   ✅ 优化器状态加载成功")
#                 return True
#             else:
#                 print("   ❌ 未能从 opt_state 中确定要加载的优化器状态字典")
#                 return False
#         else:
#             if opt['phase'] == 'test' and optimizer is None: # 如果是测试且没找到属性
#                 print("   ℹ️  测试阶段，未找到优化器属性，这通常是正常的。")
#             elif opt['phase'] == 'test' and optimizer is not None: # 找到了属性但不是优化器实例
#                 print(f"   ℹ️  测试阶段，找到属性 {attr_name} 但它不是一个有效的优化器实例。")
#             else: # 其他情况，包括训练时没找到
#                 print("   ⚠️  没有找到有效的优化器属性或实例。")
#             return True # 或者 True，取决于你是否认为这是个错误

#     except Exception as e:
#         print(f"   ❌ 优化器加载失败: {e}")
#         return False

# # ==================== 更新主函数 ====================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, default='config/ddpm_cd.json',
#                         help='JSON file for configuration')
#     parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
#                         help='Run either train(training + validation) or testing', default='train')
#     parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
#     parser.add_argument('-debug', '-d', action='store_true')
#     parser.add_argument('-enable_wandb', action='store_true')
#     parser.add_argument('-log_eval', action='store_true')
    
#     # 新增多GPU选项
#     parser.add_argument('--use_ddp', action='store_true', 
#                         help='使用DistributedDataParallel (推荐)')
#     parser.add_argument('--port', type=str, default='29500',
#                         help='分布式训练端口')
#     parser.add_argument('--multi_gpu_strategy', type=str, 
#                         choices=['auto', 'dp', 'ddp'], default='auto',
#                         help='多GPU策略: auto(自动选择), dp(DataParallel), ddp(DistributedDataParallel)')

#     args = parser.parse_args()
    
#     # 解析配置
#     opt = Logger.parse(args)
#     opt = Logger.dict_to_nonedict(opt)
    
#     # 设置多GPU环境
#     if torch.cuda.is_available():
#         world_size = torch.cuda.device_count()
#         print(f"🔍 检测到 {world_size} 个GPU")
        
#         # 根据参数决定训练策略
#         if args.multi_gpu_strategy == 'ddp' or args.use_ddp:
#             args.use_ddp = True
#         elif args.multi_gpu_strategy == 'dp':
#             args.use_ddp = False
#         else:  # auto
#             # 自动选择：GPU数量>1时优先使用DDP
#             args.use_ddp = (world_size > 1)
        
#         if world_size > 1:
#             if args.use_ddp:
#                 print(f"🚀 启动多GPU训练: DistributedDataParallel ({world_size} GPUs)")
#                 # 启动多进程分布式训练
#                 # mp.spawn(train_multi_gpu, 
#                 #         args=(world_size, args, opt), 
#                 #         nprocs=world_size, 
#                 #         join=True)
#                 # 强制使用DataParallel
#                 args.use_ddp = False
#                 args.multi_gpu_strategy = 'dp'
#                 config = MultiGPUConfig(args, opt)
#                 train_multi_gpu(0, world_size, args, opt)
#             else:
#                 print(f"🚀 启动多GPU训练: DataParallel ({world_size} GPUs)")
#                 # 单进程DataParallel训练
#                 config = MultiGPUConfig(args, opt)
#                 train_multi_gpu(0, world_size, args, opt)
#         else:
#             print("🚀 启动单GPU训练")
#             # 单GPU训练
#             config = MultiGPUConfig(args, opt)
#             train_multi_gpu(0, 1, args, opt)
#     else:
#         print("❌ 未检测到GPU，退出训练")
#         exit(1)