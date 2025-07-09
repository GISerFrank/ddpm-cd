import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from misc.metric_tools import ConfuseMatrixMeter
from misc.torchutils import get_scheduler

logger = logging.getLogger('base')


class DDPMCDModel(BaseModel):
    """
    DDPM-based Change Detection Model
    支持新的cd_head_v8架构和物理损失
    """
    def __init__(self, opt):
        super(DDPMCDModel, self).__init__(opt)
        
        # 基本配置
        self.opt = opt
        self.epoch = 0
        self.global_step = 0
        
        # 定义网络
        self.netCD = self.set_device(networks.define_CD(opt))
        self.schedule_phase = None
        
        # 设置训练相关
        if self.opt['phase'] == 'train':
            self.netCD.train()
            
            # 定义损失函数
            self.set_loss()
            
            # 定义优化器
            train_opt = opt['train']
            self.optimizer_type = train_opt['optimizer']['type']
            if self.optimizer_type == 'adam':
                self.optCD = torch.optim.Adam(
                    self.netCD.parameters(), 
                    lr=train_opt['optimizer']['lr'],
                    betas=(0.9, 0.999)
                )
            elif self.optimizer_type == 'adamw':
                self.optCD = torch.optim.AdamW(
                    self.netCD.parameters(),
                    lr=train_opt['optimizer']['lr'],
                    betas=(0.9, 0.999),
                    weight_decay=train_opt['optimizer'].get('weight_decay', 0.01)
                )
            else:
                raise NotImplementedError(f'Optimizer [{self.optimizer_type}] not implemented')
            
            # 定义学习率调度器
            self.scheduler = get_scheduler(self.optCD, train_opt)
            
            # 初始化日志
            self.log_dict = OrderedDict()
            
            # 初始化性能指标
            self.running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
            
            # TensorBoard logger
            if 'tb_logger' in opt['path']:
                from tensorboardX import SummaryWriter
                self.tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
        
        # 加载预训练权重
        self.load_network()
        self.print_network()
        
        # 记录数据加载器长度
        self.len_train_dataloader = opt.get('len_train_dataloader', 1)
        self.len_val_dataloader = opt.get('len_val_dataloader', 1)

    def feed_data(self, data):
        """Feed data to the model"""
        self.data = self.set_device(data)
        
        # 提取必要的数据
        self.A = self.data['A']
        self.B = self.data['B']
        self.label = self.data['L']
        
        # 物理数据（如果有）
        self.physical_data = self.data.get('physical_data', None)

    def optimize_parameters(self, features_A=None, features_B=None, current_epoch=None):
        """
        优化参数
        支持两种调用方式：
        1. 传入features_A和features_B（新接口）
        2. 使用临时保存的特征（兼容接口）
        """
        # 获取特征
        if features_A is None and hasattr(self, '_temp_features_A'):
            features_A = self._temp_features_A
            features_B = self._temp_features_B
            # 清理临时特征
            del self._temp_features_A
            del self._temp_features_B
        
        # 前向传播
        if self.opt['model_cd'].get('version') == 'v8':
            # cd_head_v8直接调用
            self.change_prediction = self.netCD(features_A, features_B, self.physical_data)
        else:
            # 旧版本兼容
            self.change_prediction = self.netCD(features_A, features_B)
        
        # 计算损失
        self.cal_loss()
        
        # 反向传播
        self.optCD.zero_grad()
        self.loss_v.backward()
        
        # 处理MoE辅助损失（如果存在）
        if hasattr(self.netCD, 'moe_aux_loss') and self.netCD.training:
            moe_loss = self.netCD.moe_aux_loss
            if moe_loss is not None and moe_loss.requires_grad:
                moe_loss.backward(retain_graph=True)
                self.log_dict['l_moe'] = moe_loss.item()
        
        # 梯度裁剪（可选）
        if self.opt['train'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.netCD.parameters(), 
                self.opt['train']['gradient_clip']
            )
        
        # 更新参数
        self.optCD.step()
        
        # 更新全局步数
        self.global_step += 1

    def test(self, features_A=None, features_B=None):
        """测试/验证阶段"""
        self.netCD.eval()
        
        # 获取特征
        if features_A is None and hasattr(self, '_temp_features_A'):
            features_A = self._temp_features_A
            features_B = self._temp_features_B
            del self._temp_features_A
            del self._temp_features_B
        
        with torch.no_grad():
            if self.opt['model_cd'].get('version') == 'v8':
                self.change_prediction = self.netCD(features_A, features_B, self.physical_data)
            else:
                self.change_prediction = self.netCD(features_A, features_B)
            
            # 计算损失（用于验证）
            self.cal_loss()
        
        self.netCD.train()

    def set_loss(self):
        """设置损失函数"""
        loss_type = self.opt['model_cd']['loss_type']
        
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'dice':
            from misc.losses import DiceLoss
            self.criterion = DiceLoss()
        elif loss_type == 'physics_constrained':
            from misc.losses import PhysicsConstrainedLoss
            self.criterion = PhysicsConstrainedLoss(
                base_loss_type=self.opt['model_cd'].get('base_loss', 'ce'),
                physics_weight=self.opt['model_cd'].get('physics_weight', 0.1)
            )
        else:
            raise NotImplementedError(f'Loss type [{loss_type}] not implemented')

    def cal_loss(self):
        """计算损失"""
        # 确保标签格式正确
        if self.label.dtype != torch.long and self.opt['model_cd']['loss_type'] == 'ce':
            self.label = self.label.long()
        
        if len(self.label.shape) == 4 and self.label.shape[1] == 1:
            self.label = self.label.squeeze(1)
        
        # 计算主损失
        if self.opt['model_cd']['loss_type'] == 'physics_constrained':
            # 物理约束损失需要额外信息
            self.loss_v = self.criterion(
                self.change_prediction, 
                self.label,
                physical_data=self.physical_data,
                slope_attention=getattr(self.netCD, 'slope_attention', None)
            )
        else:
            self.loss_v = self.criterion(self.change_prediction, self.label)
        
        # 记录损失
        self.log_dict['l_total'] = self.loss_v.item()

    def get_current_visuals(self):
        """获取当前可视化结果"""
        out_dict = OrderedDict()
        
        # 原始图像
        out_dict['A'] = self.A.detach().float().cpu()
        out_dict['B'] = self.B.detach().float().cpu()
        
        # 真实标签
        out_dict['L'] = self.label.detach().float().cpu()
        
        # 预测结果
        if hasattr(self, 'change_prediction'):
            if self.opt['model_cd']['out_channels'] == 2:
                # 使用softmax获取变化概率
                pred_prob = torch.softmax(self.change_prediction, dim=1)
                out_dict['pred'] = pred_prob[:, 1:2, :, :].detach().float().cpu()
            else:
                # 使用sigmoid
                out_dict['pred'] = torch.sigmoid(self.change_prediction).detach().float().cpu()
        
        # 物理数据可视化（如果有）
        if self.physical_data is not None:
            out_dict['physics'] = self.physical_data[:, 0:1, :, :].detach().float().cpu()  # 显示第一层（如DEM）
        
        return out_dict

    def get_current_losses(self):
        """获取当前损失"""
        return self.log_dict

    def print_network(self):
        """打印网络信息"""
        s, n = self.get_network_description(self.netCD)
        if isinstance(self.netCD, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netCD.__class__.__name__,
                                           self.netCD.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netCD.__class__.__name__)
        
        logger.info('Network CD structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def get_network_description(self, network):
        """获取网络描述"""
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, epoch, iter_step):
        """保存网络"""
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        
        # 保存网络
        network = self.netCD
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        
        # 保存优化器
        opt_state = {
            'epoch': epoch,
            'iter': iter_step,
            'scheduler': self.scheduler.state_dict(),
            'optimizer': self.optCD.state_dict()
        }
        torch.save(opt_state, opt_path)
        
        logger.info('Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        """加载网络"""
        load_path = self.opt['path_cd'].get('resume_state', None)
        if load_path is not None:
            logger.info('Loading pretrained model for CD [{:s}] ...'.format(load_path))
            
            gen_path = f'{load_path}_gen.pth'
            opt_path = f'{load_path}_opt.pth'
            
            # 加载网络
            network = self.netCD
            if isinstance(network, nn.DataParallel):
                network = network.module
            
            network.load_state_dict(torch.load(gen_path), strict=True)
            
            # 加载优化器（仅训练时）
            if self.opt['phase'] == 'train' and os.path.exists(opt_path):
                opt_state = torch.load(opt_path)
                self.begin_epoch = opt_state['epoch']
                self.begin_step = opt_state['iter']
                self.optCD.load_state_dict(opt_state['optimizer'])
                self.scheduler.load_state_dict(opt_state['scheduler'])

    def _update_metric(self):
        """更新评估指标"""
        if hasattr(self, 'change_prediction') and hasattr(self, 'label'):
            G_pred = self.change_prediction.detach()
            G_pred = torch.argmax(G_pred, dim=1)
            current_score = self.running_metric.update_cm(
                pr=G_pred.cpu().numpy(), 
                gt=self.label.detach().cpu().numpy()
            )
            return current_score
        return 0.0

    def _collect_running_batch_states(self):
        """收集运行时批次状态"""
        self.running_acc = self._update_metric()
        
        m = len(self.log_dict)
        if m == self.len_train_dataloader:
            for key in self.log_dict.keys():
                self.log_dict[key] /= m
            
            message = '[Training CD]. epoch: [%d/%d]. Aver_running_acc:%.5f\n' % \
                    (self.epoch, self.opt['train']['n_epoch']-1, self.running_acc/m)
            for k, v in self.log_dict.items():
                message += '%s: %.4e ' % (k, v)
                if hasattr(self, 'tb_logger'):
                    self.tb_logger.add_scalar(k, v, self.global_step)
            
            logger.info(message)
            self.log_dict = OrderedDict()

    def _clear_cache(self):
        """清理缓存"""
        self.running_metric.clear()