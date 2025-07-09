import os
import time
import argparse
import json
import warnings
import torch
import torch.nn as nn
import contextlib
import gc
import sys
import inspect
from collections import OrderedDict

# 必要的导入
import utils
import Data
import Model
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# 设置警告和异常处理
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

# ==================== 内存管理工具 ====================
@contextlib.contextmanager
def memory_efficient_context():
    """内存高效的上下文管理器"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ==================== 标签验证器 ====================
class LabelValidator:
    """统一的标签验证和适配器"""
    def __init__(self):
        self.logged_info = set()
    
    def validate_and_fix_labels(self, data, phase="train"):
        """验证并修复标签格式"""
        if 'L' not in data:
            return
        
        label = data['L']
        original_shape = label.shape
        original_dtype = label.dtype
        unique_before = torch.unique(label)
        
        # 确保标签是正确的形状
        if len(label.shape) == 4 and label.shape[1] == 1:
            label = label.squeeze(1)
            data['L'] = label
        
        # 确保标签是long类型
        if label.dtype != torch.long:
            data['L'] = label.long()
        
        # 记录验证信息（只记录一次）
        info_key = f"{phase}_label_validation"
        if info_key not in self.logged_info:
            unique_after = torch.unique(data['L'])
            print(f"📊 标签验证 ({phase}):")
            print(f"   原始形状: {original_shape} -> {data['L'].shape}")
            print(f"   数据类型: {original_dtype} -> {data['L'].dtype}")
            print(f"   唯一值: {unique_before.tolist()} -> {unique_after.tolist()}")
            self.logged_info.add(info_key)
    
    def validate_physical_data(self, data, phase="train"):
        """验证物理数据（如果存在）"""
        if 'physical_data' not in data:
            return
        
        info_key = f"{phase}_physical_validation"
        if info_key not in self.logged_info:
            phys_data = data['physical_data']
            print(f"🌡️ 物理数据验证 ({phase}):")
            print(f"   形状: {phys_data.shape}")
            print(f"   数据类型: {phys_data.dtype}")
            print(f"   范围: [{phys_data.min():.3f}, {phys_data.max():.3f}]")
            self.logged_info.add(info_key)

# 创建全局验证器实例
label_validator = LabelValidator()

# ==================== 性能监控 ====================
class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.step_times = []
        self.gpu_memory_usage = []
        self.last_report_step = 0
    
    def record_step(self, step_time):
        self.step_times.append(step_time)
        if torch.cuda.is_available():
            self.gpu_memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
    
    def report(self, current_step, interval=100):
        if current_step - self.last_report_step >= interval and len(self.step_times) > 0:
            avg_time = sum(self.step_times[-interval:]) / len(self.step_times[-interval:])
            if self.gpu_memory_usage:
                avg_memory = sum(self.gpu_memory_usage[-interval:]) / len(self.gpu_memory_usage[-interval:])
                print(f"⚡ 性能统计: 平均步骤时间={avg_time:.3f}s, GPU内存={avg_memory:.2f}GB")
            else:
                print(f"⚡ 性能统计: 平均步骤时间={avg_time:.3f}s")
            self.last_report_step = current_step

# ==================== 模型兼容性确保 ====================
def ensure_cd_model_compatibility(cd_model, opt):
    """确保CD模型具有所需的方法和属性"""
    # 检查并添加缺失的方法
    if not hasattr(cd_model, '_collect_running_batch_states'):
        def _collect_running_batch_states(self):
            self.running_acc = self._update_metric()
            
            m = len(self.log_dict) 
            if m == self.len_train_dataloader:
                for key in self.log_dict.keys():
                    self.log_dict[key] /= m
                    
                message = '[Training CD]. epoch: [%d/%d]. Aver_running_acc:%.5f\n' % \
                        (self.epoch, self.opt['n_epoch']-1, self.running_acc/m)
                for k, v in self.log_dict.items():
                    message += '%s: %.4e ' % (k, v)
                    self.tb_logger.add_scalar(k, v, self.global_step)
                    
                self.logger.info(message)
                self.log_dict = OrderedDict()
        
        cd_model._collect_running_batch_states = lambda: _collect_running_batch_states(cd_model)
    
    if not hasattr(cd_model, '_clear_cache'):
        def _clear_cache(self):
            self.running_metric.clear()
        
        cd_model._clear_cache = lambda: _clear_cache(cd_model)
    
    if not hasattr(cd_model, '_update_metric'):
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
        
        cd_model._update_metric = lambda: _update_metric(cd_model)
    
    return cd_model

# ==================== 特征处理和CD模型调用 ====================
def process_features_for_cd(change_detection, features_A, features_B, data, current_epoch=None, phase='train'):
    """
    统一处理特征并调用CD模型的相应方法
    """
    try:
        # 检查模型接口版本
        feed_data_params = inspect.signature(change_detection.feed_data).parameters
        
        if len(feed_data_params) == 2:  # 新接口: feed_data(self, data)
            change_detection.feed_data(data)
            
            if phase == 'train':
                # 检查optimize_parameters的参数
                if hasattr(change_detection.optimize_parameters, '__code__'):
                    opt_params = change_detection.optimize_parameters.__code__.co_varnames
                    if 'features_A' in opt_params:
                        change_detection.optimize_parameters(features_A, features_B, current_epoch=current_epoch)
                    else:
                        # 通过临时属性传递特征
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
                    test_params = inspect.signature(change_detection.test).parameters if hasattr(change_detection.test, '__call__') else []
                    if len(test_params) > 1:
                        change_detection.test(features_A, features_B)
                    else:
                        change_detection._temp_features_A = features_A
                        change_detection._temp_features_B = features_B
                        change_detection.test()
        
        else:  # 旧接口: feed_data(self, features_A, features_B, data)
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

# ==================== 高效批量处理 ====================
def process_batch_efficiently(train_data, diffusion, change_detection, opt, phase="train", current_epoch=None):
    """高效的批量处理 - 支持cd_head_v8的直接调用"""
    with memory_efficient_context():
        try:
            # 1. 标签验证
            label_validator.validate_and_fix_labels(train_data, phase)
            
            # 2. 物理数据验证和准备
            physical_data = None
            if opt['model_cd'].get('physics_attention', {}).get('enabled', False):
                if 'physical_data' in train_data:
                    physical_data = train_data['physical_data']
                    label_validator.validate_physical_data(train_data, phase)
                else:
                    print("⚠️ 物理注意力已启用但未找到物理数据")
            
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
            
            # 6. 特征重排（为cd_head_v8准备正确格式）
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
            
            # 7. 检查cd_head_v8的forward方法并直接调用
            if hasattr(change_detection.netCD, 'forward') and opt['model_cd'].get('version') == 'v8':
                # cd_head_v8可以直接调用forward
                change_detection.feed_data(train_data)
                
                if phase == 'train':
                    # 训练时调用netCD.forward并计算损失
                    cm = change_detection.netCD(reordered_f_A, reordered_f_B, physical_data)
                    change_detection.change_prediction = cm
                    
                    # 计算损失和优化
                    change_detection.cal_loss()
                    change_detection.optCD.zero_grad()
                    change_detection.loss_v.backward()
                    change_detection.optCD.step()
                    
                    # 更新指标
                    change_detection._update_metric()
                else:
                    # 测试/验证阶段
                    with torch.no_grad():
                        cm = change_detection.netCD(reordered_f_A, reordered_f_B, physical_data)
                        change_detection.change_prediction = cm
                        change_detection._update_metric()
                        
            else:
                # 旧版本模型的处理
                process_features_for_cd(change_detection, reordered_f_A, reordered_f_B, 
                                      train_data, current_epoch, phase)
            
            # 8. 清理内存
            del f_A, f_B, reordered_f_A, reordered_f_B
            
        except Exception as e:
            print(f"❌ 批处理错误: {e}")
            import traceback
            traceback.print_exc()
            raise

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
    
    # 设置diffusion模型为eval模式
    if hasattr(diffusion.netG, 'eval'):
        diffusion.netG.eval()
        print("   ✅ Diffusion模型设置为eval模式")
    
    # 显示GPU信息
    if torch.cuda.device_count() > 1:
        print(f"   ✅ 检测到{torch.cuda.device_count()}个GPU")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"     GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    
    print("🚀 训练优化设置完成\n")
    
    return use_amp

# ==================== 验证和测试功能 ====================
def validation(val_loader, epoch, diffusion, change_detection, opt, tb_logger):
    """执行验证"""
    avg_loss = 0.0
    diffusion.eval()
    change_detection._clear_cache()
    
    with torch.no_grad():
        for _, val_data in enumerate(val_loader):
            process_batch_efficiently(val_data, diffusion, change_detection, opt, phase="val")
            avg_loss += change_detection.loss_v
    
    avg_loss = avg_loss / len(val_loader)
    
    # 获取性能指标
    scores_dict = change_detection.running_metric.get_cm()
    
    # 使用提供的计算函数
    val_metrics = compute_metrics(scores_dict)
    
    # 记录到tensorboard
    tb_logger.add_scalar('[Val] Loss_epoch', avg_loss, epoch)
    print(f"\n[Val] epoch: {epoch}. loss: {avg_loss:.5f}. "
          f"F1: {val_metrics['mF1']:.5f}, IoU: {val_metrics['mIoU']:.5f}")
    
    return val_metrics

def compute_metrics(conf_mat_dict):
    """计算评估指标"""
    tn = conf_mat_dict['tn']
    fp = conf_mat_dict['fp']
    fn = conf_mat_dict['fn'] 
    tp = conf_mat_dict['tp']
    
    # 计算每个类别的指标
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    iou_0 = tn / (tn + fn + fp) if (tn + fn + fp) > 0 else 0
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    iou_1 = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    
    # 计算平均指标
    mf1 = (f1_0 + f1_1) / 2
    miou = (iou_0 + iou_1) / 2
    
    return {
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
    
    # 物理损失相关参数
    parser.add_argument('--use_physics_loss', action='store_true',
                        help='Enable physics-constrained loss')
    parser.add_argument('--physics_config', type=str, default=None,
                        help='Override physics loss config file')

    # 解析参数
    args = parser.parse_args()
    opt = utils.parse(args)
    opt = utils.dict_to_nonedict(opt)

    # 设置日志
    utils.mkdirs(
        (path for key, path in opt['path'].items() if not key == 'experiments_root'))
    utils.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    utils.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(utils.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # 设置随机种子
    torch.manual_seed(12345)
    
    # ==================== 数据集准备 ====================
    print("\n" + "="*50)
    print("🔧 准备数据集...")
    print("="*50)
    
    for phase, dataset in opt['datasets'].items():
        if phase == 'train' and args.phase == 'train':
            print("Creating [train] change-detection dataloader.")
            dataset_opt = dataset
            train_set = Data.create_cd_dataset(dataset_opt, phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)
            
        elif phase == 'val' and args.phase == 'train':
            print("Creating [val] change-detection dataloader.")
            dataset_opt = dataset
            val_set = Data.create_cd_dataset(dataset_opt, phase)
            val_loader = Data.create_cd_dataloader(val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)
        
        elif phase == 'test' and args.phase == 'test':
            print("Creating [test] change-detection dataloader.")
            test_set = Data.create_cd_dataset(dataset_opt, phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)
    
    logger.info('Initial Dataset Finished')

    # ==================== 模型加载 ====================
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
        diffusion.netG = nn.DataParallel(diffusion.netG)

    # 设置噪声调度
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    # 创建变化检测模型
    print("🔄 加载变化检测模型...")
    try:
        change_detection = Model.create_CD_model(opt)
        
        # 确保兼容性
        change_detection = ensure_cd_model_compatibility(change_detection, opt)
        
        print(f"✅ 成功创建CD模型: {change_detection.__class__.__name__}")
        
        # 特别处理cd_head_v8
        if opt['model_cd'].get('version') == 'v8':
            print("🔧 检测到cd_head_v8模型")
            
            # 检查各个模块的启用状态
            if hasattr(change_detection, 'netCD'):
                netCD = change_detection.netCD
                print(f"   物理注意力: {'✓' if netCD.use_physics else '✗'}")
                print(f"   交叉注意力: {'✓' if netCD.use_cross_attention else '✗'}")
                print(f"   Mamba全局分析: {'✓' if netCD.use_mamba else '✗'}")
                print(f"   物理聚焦: {'✓' if netCD.use_physics_focus else '✗'}")
                print(f"   MoE专家决策: {'✓' if netCD.use_moe else '✗'}")
        
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

    # 初始化性能指标跟踪器（如果需要）
    if not hasattr(change_detection, 'running_metric'):
        print("🔧 为CD模型添加性能指标跟踪器...")
        from misc.metric_tools import ConfuseMatrixMeter
        change_detection.running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])

    # 设置数据集长度信息
    if 'len_train_dataloader' in opt:
        change_detection.len_train_dataloader = opt["len_train_dataloader"]
    if 'len_val_dataloader' in opt:
        change_detection.len_val_dataloader = opt["len_val_dataloader"]

    print("🚀 变化检测模型初始化完成\n")
    
    # 设备一致性设置
    if torch.cuda.is_available():
        target_device = torch.device('cuda:0')
        
        # 确保模型在GPU
        if isinstance(diffusion.netG, nn.DataParallel):
            diffusion.netG.module = diffusion.netG.module.to(target_device)
        else:
            diffusion.netG = diffusion.netG.to(target_device)
        
        if hasattr(change_detection, 'netCD') and change_detection.netCD is not None:
            if isinstance(change_detection.netCD, nn.DataParallel):
                change_detection.netCD.module = change_detection.netCD.module.to(target_device)
            else:
                change_detection.netCD = change_detection.netCD.to(target_device)
        
        print(f"✅ 设备设置完成: {target_device}")

    # 设置训练优化
    use_amp = setup_training_optimization(diffusion, change_detection)
    
    # 创建性能监控器
    performance_monitor = PerformanceMonitor()

    # ==================== 训练循环 ====================
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0

    if opt['phase'] == 'train':
        print("\n" + "="*50)
        print("🚀 开始训练")
        print("="*50)
        
        for current_epoch in range(start_epoch, n_epoch):
            epoch_start_time = time.time()
            change_detection._clear_cache()
            
            train_result_path = f'{opt["path"]["results"]}/train/{current_epoch}'
            os.makedirs(train_result_path, exist_ok=True)
            
            # 训练阶段
            print(f"\n🎯 开始训练 Epoch {current_epoch}/{n_epoch-1}")
            message = f'学习率: {change_detection.optCD.param_groups[0]["lr"]:.7f}'
            logger.info(message)
            
            # 设置cd_head_v8的epoch（如果支持条件MoE）
            if hasattr(change_detection.netCD, 'set_condition') and opt['model_cd'].get('moe_config', {}).get('enabled', False):
                # 根据epoch设置不同的条件
                if current_epoch < 20:
                    condition = 'rainfall'  # 早期关注降雨
                elif current_epoch < 50:
                    condition = 'snowmelt'  # 中期关注融雪
                else:
                    condition = 'complex'   # 后期关注复杂情况
                change_detection.netCD.set_condition(condition)
                print(f"   MoE条件设置为: {condition}")
            
            for current_step, train_data in enumerate(train_loader):
                step_start_time = time.time()
                
                # 记录batch size用于MoE
                if hasattr(change_detection.netCD, '_batch_size'):
                    change_detection.netCD._batch_size = train_data['A'].shape[0]
                
                # 处理批次
                process_batch_efficiently(train_data, diffusion, change_detection, 
                                       opt, phase="train", current_epoch=current_epoch)
                
                # 处理MoE辅助损失（如果存在）
                if hasattr(change_detection.netCD, 'moe_aux_loss') and change_detection.netCD.training:
                    moe_loss = change_detection.netCD.moe_aux_loss
                    if moe_loss is not None and moe_loss.requires_grad:
                        moe_loss.backward(retain_graph=True)
                        if current_step % 100 == 0:
                            print(f"   MoE辅助损失: {moe_loss.item():.4f}")
                
                # 更新指标
                change_detection._collect_running_batch_states()
                
                # 记录性能
                step_time = time.time() - step_start_time
                performance_monitor.record_step(step_time)
                
                # 定期报告性能
                if current_step % 100 == 0:
                    performance_monitor.report(current_step)
                    
                    # 获取并保存注意力图（用于分析）
                    if hasattr(change_detection.netCD, 'get_attention_maps'):
                        attention_maps = change_detection.netCD.get_attention_maps()
                        if attention_maps:
                            # 这里可以保存或可视化注意力图
                            pass
                
                # 保存中间结果
                if current_step % 500 == 0:
                    utils.save_model(change_detection, current_epoch, current_step, opt)
            
            # 验证阶段
            if current_epoch % opt['val']['val_epoch'] == 0:
                print(f"\n📊 开始验证 Epoch {current_epoch}")
                val_metrics = validation(val_loader, current_epoch, diffusion, 
                                       change_detection, opt, tb_logger)
                
                # 保存最佳模型
                if val_metrics['mF1'] > best_mF1:
                    best_mF1 = val_metrics['mF1']
                    print(f"🏆 新的最佳mF1: {best_mF1:.5f}")
                    utils.save_best_model(change_detection, opt)
            
            # 更新学习率
            change_detection.scheduler.step()
            
            # 记录epoch时间
            epoch_time = time.time() - epoch_start_time
            print(f"⏱️ Epoch {current_epoch} 完成，用时: {epoch_time/60:.2f}分钟")
            
    elif opt['phase'] == 'test':
        print("\n" + "="*50)
        print("🧪 开始测试")
        print("="*50)
        
        change_detection._clear_cache()
        test_result_path = f'{opt["path"]["results"]}/test'
        os.makedirs(test_result_path, exist_ok=True)
        
        with torch.no_grad():
            for current_step, test_data in enumerate(test_loader):
                process_batch_efficiently(test_data, diffusion, change_detection, 
                                       opt, phase="test")
                
                # 保存预测结果
                visuals = change_detection.get_current_visuals()
                img_name = os.path.splitext(os.path.basename(test_data['ID'][0]))[0]
                utils.save_images(visuals, test_result_path, img_name)
                
                if current_step % 50 == 0:
                    print(f"处理进度: {current_step}/{len(test_loader)}")
        
        # 计算测试指标
        scores_dict = change_detection.running_metric.get_cm()
        test_metrics = compute_metrics(scores_dict)
        
        print("\n📊 测试结果:")
        for metric, value in test_metrics.items():
            print(f"   {metric}: {value:.5f}")
    
    print("\n✅ 程序执行完成！")