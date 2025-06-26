# 将你提供的物理约束损失函数代码完整复制到这里
"""
滑坡变化检测的物理约束损失函数
在标准交叉熵损失基础上增加地质领域知识约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LandslidePhysicsConstrainedLoss(nn.Module):
    """
    滑坡物理约束损失函数
    
    将地质领域知识融入损失函数，包括：
    1. 坡度约束：平地很少滑坡，极陡坡反而稳定
    2. 空间连续性：滑坡应该是连续区域，不是零散点
    3. 尺寸约束：避免预测过小的碎片
    4. 地质约束：不同岩石类型的易发性不同
    """
    
    def __init__(self, 
                 alpha=0,      # 坡度约束权重
                 beta=0.05,      # 空间连续性权重
                 gamma=0.03,     # 尺寸约束权重
                 delta=0,     # 地质约束权重
                 enable_progressive=True,  # 渐进式训练
                 warmup_epochs=10):       # 预热轮次
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.delta = delta
        self.enable_progressive = enable_progressive
        self.warmup_epochs = warmup_epochs
        
        # 基础损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        
        print(f"🔧 初始化物理约束损失函数:")
        print(f"   坡度约束权重 (alpha): {alpha}")
        print(f"   空间连续性权重 (beta): {beta}")
        print(f"   尺寸约束权重 (gamma): {gamma}")
        print(f"   地质约束权重 (delta): {delta}")
        print(f"   渐进式训练: {enable_progressive}")
        
    def forward(self, pred, target, physical_data=None, epoch=None):
        """
        计算物理约束损失
        
        Args:
            pred: 预测结果 [B, 2, H, W] 
            target: 真实标签 [B, H, W]
            physical_data: 物理数据 [B, num_layers, H, W] (可选)
                          0: DEM, 1: slope, 2: aspect, 3: geology, etc.
            epoch: 当前训练轮次 (用于渐进式训练)
            
        Returns:
            total_loss: 总损失值 (scalar)
        """
        # 1. 基础分类损失
        ce_loss = self.ce_loss(pred, target)
        
        # 2. 如果没有物理数据，只返回基础损失
        if physical_data is None:
            return ce_loss
        
        # 3. 计算物理约束权重（渐进式训练）
        constraint_weight = self._get_constraint_weight(epoch)
        
        # 4. 计算各种物理约束
        total_constraint = 0.0
        constraint_count = 0
        
        # 坡度约束
        if self.alpha > 0 and physical_data.shape[1] > 1:
            slope_constraint = self._slope_physics_constraint(pred, physical_data[:, 1])
            total_constraint += self.alpha * slope_constraint
            constraint_count += 1
            
        # 空间连续性约束
        if self.beta > 0:
            spatial_constraint = self._spatial_continuity_constraint(pred)
            total_constraint += self.beta * spatial_constraint
            constraint_count += 1
            
        # 滑坡尺寸约束
        if self.gamma > 0:
            size_constraint = self._landslide_size_constraint(pred)
            total_constraint += self.gamma * size_constraint
            constraint_count += 1
            
        # 地质约束
        if self.delta > 0 and physical_data.shape[1] > 3:
            geology_constraint = self._geology_constraint(pred, physical_data[:, 3])
            total_constraint += self.delta * geology_constraint
            constraint_count += 1
        
        # 5. 总损失
        total_loss = ce_loss + constraint_weight * total_constraint
        
        return total_loss
    
    def _get_constraint_weight(self, epoch):
        """
        计算约束权重（渐进式训练）
        前几个epoch约束权重较小，逐渐增加到1.0
        """
        if not self.enable_progressive or epoch is None:
            return 1.0
            
        if epoch < 3:
            # 前3个epoch几乎不用约束，让模型先学会基础分类
            return 0.1
        elif epoch < self.warmup_epochs:
            # 3到warmup_epochs之间线性增长
            return 0.1 + 0.9 * (epoch - 3) / (self.warmup_epochs - 3)
        else:
            return 1.0
    
    def _slope_physics_constraint(self, pred, slope):
        """
        坡度物理约束
        基于滑坡发生的坡度规律：
        - 平地 (<5°): 很少滑坡
        - 极陡坡 (>70°): 反而稳定（缺少松散物质）
        """
        landslide_prob = torch.softmax(pred, dim=1)[:, 1]  # 滑坡概率 [B, H, W]
        
        constraint_loss = 0.0
        
        # 约束1：平地很少发生滑坡
        flat_mask = (slope < 5).float()
        flat_penalty = (landslide_prob * flat_mask).mean()
        constraint_loss += flat_penalty
        
        # 约束2：极陡坡反而稳定（权重稍低，因为不是绝对规律）
        very_steep_mask = (slope > 70).float()
        steep_penalty = (landslide_prob * very_steep_mask).mean()
        constraint_loss += steep_penalty * 0.5
        
        return constraint_loss
    
    def _spatial_continuity_constraint(self, pred):
        """
        空间连续性约束
        滑坡通常是连续的区域，不是零散的像素点
        使用拉普拉斯算子检测预测的空间不连续性
        """
        landslide_prob = torch.softmax(pred, dim=1)[:, 1:2]  # [B, 1, H, W]
        
        # 拉普拉斯算子核
        laplacian_kernel = torch.tensor([[[0, 1, 0],
                                         [1, -4, 1], 
                                         [0, 1, 0]]],
                                       dtype=pred.dtype, device=pred.device)
        
        # 计算空间梯度
        edges = F.conv2d(landslide_prob, laplacian_kernel, padding=1)
        
        # 惩罚过多的边缘（即不连续的预测）
        discontinuity = torch.abs(edges).mean()
        
        return discontinuity
    
    def _landslide_size_constraint(self, pred):
        """
        滑坡尺寸约束
        真实滑坡通常有一定的最小尺寸，避免预测零散的小点
        通过惩罚预测的空间分散程度来实现
        """
        landslide_prob = torch.softmax(pred, dim=1)[:, 1]  # [B, H, W]
        
        batch_size = landslide_prob.shape[0]
        fragmentation_penalty = 0.0
        
        for i in range(batch_size):
            prob_map = landslide_prob[i]
            
            # 如果预测概率的总量很小，跳过
            if prob_map.sum() < 0.1:
                continue
                
            # 计算预测的空间集中度
            h, w = prob_map.shape
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=prob_map.device),
                torch.arange(w, dtype=torch.float32, device=prob_map.device),
                indexing='ij'
            )
            
            # 加权质心
            total_weight = prob_map.sum()
            centroid_y = (prob_map * y_coords).sum() / (total_weight + 1e-8)
            centroid_x = (prob_map * x_coords).sum() / (total_weight + 1e-8)
            
            # 计算分散程度（方差）
            var_y = (prob_map * (y_coords - centroid_y)**2).sum() / (total_weight + 1e-8)
            var_x = (prob_map * (x_coords - centroid_x)**2).sum() / (total_weight + 1e-8)
            
            # 如果分散程度过高，增加惩罚
            total_variance = var_y + var_x
            if total_variance > 100:  # 经验阈值，可调整
                fragmentation_penalty += 1.0 / (1.0 + total_variance / 100)
        
        return fragmentation_penalty / batch_size
    
    def _geology_constraint(self, pred, geology):
        """
        地质约束
        基于不同岩石类型的滑坡易发性
        在稳定岩石区域预测高滑坡概率会被惩罚
        """
        landslide_prob = torch.softmax(pred, dim=1)[:, 1]  # [B, H, W]
        
        # 岩石类型易发性字典（需要根据实际数据调整）
        # 这里假设：1-3是软岩（易滑），4-6是硬岩（稳定）
        stable_rock_types = [1, 2]  # 可以根据实际地质编码调整
        
        constraint_loss = 0.0
        
        # 对于稳定岩石类型，惩罚高滑坡预测概率
        for rock_type in stable_rock_types:
            rock_mask = (geology == rock_type).float()
            # 在稳定岩石区域预测高滑坡概率将被惩罚
            stable_rock_penalty = (landslide_prob * rock_mask).mean()
            constraint_loss += stable_rock_penalty * 0.5  # 权重0.5
        
        return constraint_loss
    
    def get_constraint_details(self, pred, target, physical_data):
        """
        获取各个约束项的详细数值（用于调试和分析）
        
        Returns:
            dict: 包含各个损失组件数值的字典
        """
        details = {}
        
        # 基础损失
        details['ce_loss'] = self.ce_loss(pred, target).item()
        
        if physical_data is not None:
            # 各个约束项
            if physical_data.shape[1] > 1:
                details['slope_constraint'] = self._slope_physics_constraint(
                    pred, physical_data[:, 1]).item()
            
            details['spatial_constraint'] = self._spatial_continuity_constraint(pred).item()
            details['size_constraint'] = self._landslide_size_constraint(pred).item()
            
            if physical_data.shape[1] > 3:
                details['geology_constraint'] = self._geology_constraint(
                    pred, physical_data[:, 3]).item()
        
        return details


class ProgressivePhysicsLoss(LandslidePhysicsConstrainedLoss):
    """
    增强版渐进式物理约束损失
    提供更细粒度的训练控制
    """
    
    def __init__(self, stage_epochs=[5, 15, 30], **kwargs):
        """
        Args:
            stage_epochs: 不同阶段的epoch边界
                         [5, 15, 30] 表示：
                         - 0-5: 基础训练
                         - 5-15: 加入坡度和空间约束
                         - 15-30: 加入所有约束
                         - 30+: 全约束训练
        """
        super().__init__(**kwargs)
        self.stage_epochs = stage_epochs
        
    def forward(self, pred, target, physical_data=None, epoch=None):
        """分阶段的渐进式训练"""
        
        if epoch is None or physical_data is None:
            return super().forward(pred, target, physical_data, epoch)
        
        # 基础损失
        ce_loss = self.ce_loss(pred, target)
        
        # 根据训练阶段决定使用哪些约束
        if epoch < self.stage_epochs[0]:
            # 阶段1：只用基础损失
            return ce_loss
            
        elif epoch < self.stage_epochs[1]:
            # 阶段2：基础损失 + 坡度约束 + 空间约束
            constraint = 0.0
            if self.alpha > 0 and physical_data.shape[1] > 1:
                constraint += self.alpha * self._slope_physics_constraint(pred, physical_data[:, 1])
            if self.beta > 0:
                constraint += self.beta * self._spatial_continuity_constraint(pred)
            return ce_loss + constraint * 0.5  # 权重减半
            
        elif epoch < self.stage_epochs[2]:
            # 阶段3：加入尺寸约束
            constraint = 0.0
            if self.alpha > 0 and physical_data.shape[1] > 1:
                constraint += self.alpha * self._slope_physics_constraint(pred, physical_data[:, 1])
            if self.beta > 0:
                constraint += self.beta * self._spatial_continuity_constraint(pred)
            if self.gamma > 0:
                constraint += self.gamma * self._landslide_size_constraint(pred)
            return ce_loss + constraint * 0.8  # 权重增加
            
        else:
            # 阶段4：全约束
            return super().forward(pred, target, physical_data, epoch)


def create_physics_loss(config):
    """
    根据配置创建物理约束损失函数
    
    Args:
        config: 配置字典，包含损失函数参数
        
    Returns:
        loss_function: 物理约束损失函数实例
    """
    # 默认参数
    default_config = {
        'alpha': 0.1,
        'beta': 0.05, 
        'gamma': 0.03,
        'delta': 0.02,
        'enable_progressive': True,
        'warmup_epochs': 10,
        'use_progressive': False
    }
    
    # 更新配置
    default_config.update(config)
    
    # 选择损失函数类型
    if default_config.get('use_progressive', False):
        loss_fn = ProgressivePhysicsLoss(
            alpha=default_config['alpha'],
            beta=default_config['beta'],
            gamma=default_config['gamma'],
            delta=default_config['delta'],
            enable_progressive=default_config['enable_progressive'],
            warmup_epochs=default_config['warmup_epochs']
        )
        print("🚀 使用增强版渐进式物理约束损失")
    else:
        loss_fn = LandslidePhysicsConstrainedLoss(
            alpha=default_config['alpha'],
            beta=default_config['beta'],
            gamma=default_config['gamma'],
            delta=default_config['delta'],
            enable_progressive=default_config['enable_progressive'],
            warmup_epochs=default_config['warmup_epochs']
        )
        print("🚀 使用标准物理约束损失")
    
    return loss_fn