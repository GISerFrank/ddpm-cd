"""
物理引导的变化聚焦模块 - 策略二
在已经理解了变化的全局形态后，使用物理信息进一步聚焦关键区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhysicsChangeFocusAttention(nn.Module):
    """
    物理引导的变化聚焦注意力
    
    与策略一不同，这里我们：
    1. 输入是已经包含丰富语义的变化特征（经过交叉注意力和Mamba处理）
    2. 使用物理特征生成"重要性掩码"，强化关键区域的变化信号
    3. 特别关注高风险物理环境（如陡坡、不稳定地质）的变化
    """
    
    def __init__(self, change_channels, physical_channels, hidden_dim=128, 
                 num_heads=4, dropout=0.1, temperature=1.0):
        super().__init__()
        
        self.change_channels = change_channels
        self.physical_channels = physical_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        # 变化特征投影
        self.change_proj = nn.Sequential(
            nn.Conv2d(change_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 物理特征投影（用于生成重要性权重）
        self.physics_proj = nn.Sequential(
            nn.Conv2d(physical_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 风险评估网络 - 学习哪些物理条件下的变化最重要
        self.risk_assessment = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 空间感知的聚焦机制
        self.spatial_focus = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim)
        )
        
        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, change_channels, 1),
            nn.BatchNorm2d(change_channels)
        )
        
    def forward(self, change_features, physical_features):
        """
        Args:
            change_features: [B, C, H, W] - 经过Mamba处理的变化特征
            physical_features: [B, P, H, W] - 编码的物理特征
        
        Returns:
            focused_changes: [B, C, H, W] - 聚焦后的变化特征
            attention_map: [B, 1, H, W] - 注意力图（用于可视化）
        """
        B, C, H, W = change_features.shape
        
        # 投影特征
        change_proj = self.change_proj(change_features)
        physics_proj = self.physics_proj(physical_features)
        
        # 生成物理风险评估图
        risk_map = self.risk_assessment(physics_proj)  # [B, 1, H, W]
        
        # 应用温度调节的风险图作为初始注意力
        risk_attention = torch.sigmoid(risk_map / self.temperature)
        
        # 准备多头注意力输入
        change_seq = change_proj.flatten(2).transpose(1, 2)  # [B, HW, D]
        physics_seq = physics_proj.flatten(2).transpose(1, 2)  # [B, HW, D]
        
        # 使用物理特征作为Query，变化特征作为Key和Value
        # 这样可以让物理高风险区域"查询"对应位置的变化
        attended_changes, attn_weights = self.multihead_attn(
            query=physics_seq,
            key=change_seq,
            value=change_seq,
            need_weights=True
        )
        
        # 重塑回2D
        attended_2d = attended_changes.transpose(1, 2).reshape(B, -1, H, W)
        
        # 空间感知的聚焦
        combined = torch.cat([change_proj, attended_2d], dim=1)
        focused = self.spatial_focus(combined)
        
        # 应用风险加权
        focused = focused * (1 + risk_attention)
        
        # 输出投影
        output = self.out_proj(focused)
        
        # 残差连接
        output = output + change_features
        
        return output, risk_attention


class AdaptivePhysicsFocus(nn.Module):
    """
    自适应物理聚焦模块
    根据不同的物理条件动态调整聚焦策略
    """
    
    def __init__(self, change_channels, physical_channels, hidden_dim=128,
                 num_experts=4, dropout=0.1):
        super().__init__()
        
        self.num_experts = num_experts
        
        # 物理条件分类器（决定使用哪种聚焦策略）
        self.condition_classifier = nn.Sequential(
            nn.Conv2d(physical_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # 多个专家聚焦模块（每个针对不同的物理条件）
        self.expert_focus = nn.ModuleList([
            PhysicsChangeFocusAttention(
                change_channels, physical_channels, hidden_dim, dropout=dropout
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, change_features, physical_features):
        """
        自适应地选择聚焦策略
        """
        B = change_features.shape[0]
        
        # 确定物理条件类型
        condition_weights = self.condition_classifier(physical_features)  # [B, num_experts]
        
        # 应用不同的专家模块
        expert_outputs = []
        expert_attentions = []
        
        for i, expert in enumerate(self.expert_focus):
            output, attention = expert(change_features, physical_features)
            expert_outputs.append(output)
            expert_attentions.append(attention)
        
        # 加权组合专家输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, C, H, W]
        condition_weights = condition_weights.view(B, self.num_experts, 1, 1, 1)
        
        focused_output = (expert_outputs * condition_weights).sum(dim=1)
        
        # 组合注意力图
        expert_attentions = torch.stack(expert_attentions, dim=1)
        combined_attention = (expert_attentions * condition_weights).sum(dim=1)
        
        return focused_output, combined_attention


class MultiScalePhysicsFocus(nn.Module):
    """
    多尺度物理聚焦模块
    为不同尺度配置不同的聚焦参数
    """
    
    def __init__(self, scale_dims, physical_dim=64, hidden_dim=128,
                 use_adaptive=True, dropout=0.1):
        super().__init__()
        
        self.scale_dims = scale_dims
        self.use_adaptive = use_adaptive
        
        # 为每个尺度创建聚焦模块
        self.focus_modules = nn.ModuleList()
        
        for dim in scale_dims:
            if use_adaptive:
                # 使用自适应聚焦
                module = AdaptivePhysicsFocus(
                    dim, physical_dim, hidden_dim, dropout=dropout
                )
            else:
                # 使用标准聚焦
                module = PhysicsChangeFocusAttention(
                    dim, physical_dim, hidden_dim, dropout=dropout
                )
            
            self.focus_modules.append(module)
    
    def forward(self, change_features, physical_features, scale_idx):
        """
        根据尺度选择合适的聚焦模块
        
        Args:
            change_features: [B, C, H, W] - Mamba输出的变化特征
            physical_features: [B, P, H, W] - 物理特征
            scale_idx: 当前尺度索引
        
        Returns:
            focused_changes: [B, C, H, W] - 聚焦后的变化特征
            attention_map: [B, 1, H, W] - 注意力图
        """
        if scale_idx < len(self.focus_modules):
            return self.focus_modules[scale_idx](change_features, physical_features)
        else:
            # 超出范围，使用最后一个
            return self.focus_modules[-1](change_features, physical_features)


class PhysicsGuidedGating(nn.Module):
    """
    物理引导的门控机制
    简化版的聚焦方法，通过门控直接调节变化强度
    """
    
    def __init__(self, change_channels, physical_channels, reduction=4):
        super().__init__()
        
        # 门控生成网络
        self.gate_conv = nn.Sequential(
            nn.Conv2d(physical_channels + change_channels, 
                     change_channels // reduction, 1),
            nn.BatchNorm2d(change_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(change_channels // reduction, change_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, change_features, physical_features):
        """
        简单但有效的门控聚焦
        """
        # 上采样物理特征到匹配变化特征的分辨率
        if physical_features.shape[2:] != change_features.shape[2:]:
            physical_features = F.interpolate(
                physical_features, 
                size=change_features.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # 拼接特征
        combined = torch.cat([change_features, physical_features], dim=1)
        
        # 生成门控权重
        gates = self.gate_conv(combined)
        
        # 应用门控
        gated_changes = change_features * gates
        
        # 残差连接
        output = gated_changes + change_features
        
        return output, gates