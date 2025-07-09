"""
条件化专家决策模块 (Conditional Expert Decision with MoE)
第五阶段：根据滑坡诱因类别，选择最合适的专家进行最终分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExpertNetwork(nn.Module):
    """
    单个专家网络
    每个专家专门处理特定类型的滑坡诱因
    """
    def __init__(self, in_channels, hidden_dim=256, out_channels=None, 
                 expert_type="general", dropout=0.1):
        super().__init__()
        
        self.expert_type = expert_type
        out_channels = out_channels or in_channels
        
        # 专家特定的特征提取
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # 专家特定的注意力机制
        self.expert_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 专家类型嵌入（用于条件化处理）
        self.type_embedding = nn.Parameter(torch.randn(1, hidden_dim, 1, 1))
        
    def forward(self, x, confidence=1.0):
        """
        Args:
            x: [B, C, H, W] - 输入特征
            confidence: 标量 - 该专家的置信度
        """
        # 特征提取
        features = self.feature_extraction(x)
        
        # 添加专家类型嵌入
        features = features + self.type_embedding
        
        # 生成专家特定的注意力
        attention = self.expert_attention(features)
        
        # 应用注意力
        attended_features = features * attention
        
        # 输出投影
        output = self.output_projection(attended_features)
        
        # 应用置信度加权
        output = output * confidence
        
        # 残差连接
        if output.shape == x.shape:
            output = output + x
        
        return output, attention


class GatingNetwork(nn.Module):
    """
    门控网络（路由器）
    根据输入特征和条件信息决定每个专家的权重
    """
    def __init__(self, in_channels, num_experts, num_conditions=5, 
                 temperature=1.0, use_load_balancing=True):
        super().__init__()
        
        self.num_experts = num_experts
        self.num_conditions = num_conditions
        self.temperature = temperature
        self.use_load_balancing = use_load_balancing
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 条件嵌入
        self.condition_embedding = nn.Embedding(num_conditions, 64)
        
        # 门控决策网络
        self.gate_network = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_experts)
        )
        
        # 负载均衡损失的重要性因子
        if use_load_balancing:
            self.importance_network = nn.Linear(128, num_experts)
            
    def forward(self, x, condition_id=None):
        """
        Args:
            x: [B, C, H, W] - 输入特征
            condition_id: [B] - 条件ID（滑坡诱因类别）
        
        Returns:
            gates: [B, num_experts] - 每个专家的权重
            load_balance_loss: 负载均衡损失
        """
        B = x.shape[0]
        
        # 编码特征
        feat_encoding = self.feature_encoder(x)  # [B, 128]
        
        # 处理条件信息
        if condition_id is not None:
            condition_emb = self.condition_embedding(condition_id)  # [B, 64]
            combined = torch.cat([feat_encoding, condition_emb], dim=1)
        else:
            # 如果没有条件信息，使用默认嵌入
            default_cond = torch.zeros(B, 64, device=x.device)
            combined = torch.cat([feat_encoding, default_cond], dim=1)
        
        # 计算门控权重
        logits = self.gate_network(combined)  # [B, num_experts]
        
        # 应用温度和softmax
        gates = F.softmax(logits / self.temperature, dim=1)
        
        # 计算负载均衡损失
        load_balance_loss = 0.0
        if self.use_load_balancing and self.training:
            # 计算重要性分数
            importance = self.importance_network(feat_encoding)
            importance = F.softmax(importance, dim=1)
            
            # 负载均衡损失：鼓励均匀使用专家
            load = gates.mean(dim=0)  # 每个专家的平均负载
            load_balance_loss = self.num_experts * (load * load).sum()
            
        return gates, load_balance_loss


class ConditionalMoE(nn.Module):
    """
    条件化专家混合层
    根据滑坡诱因动态选择和组合多个专家的预测
    """
    def __init__(self, in_channels, num_experts=4, num_conditions=5,
                 expert_types=None, hidden_dim=256, temperature=1.0,
                 use_load_balancing=True, dropout=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.num_conditions = num_conditions
        
        # 定义专家类型
        if expert_types is None:
            expert_types = ["rainfall", "earthquake", "human", "complex"]
        
        # 创建专家网络
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert_type = expert_types[i] if i < len(expert_types) else "general"
            self.experts.append(
                ExpertNetwork(
                    in_channels, hidden_dim, in_channels,
                    expert_type=expert_type, dropout=dropout
                )
            )
        
        # 创建门控网络
        self.gating = GatingNetwork(
            in_channels, num_experts, num_conditions,
            temperature=temperature, use_load_balancing=use_load_balancing
        )
        
        # Top-k选择（可选）
        self.top_k = min(2, num_experts)  # 只激活top-k个专家
        
        print(f"🎯 初始化条件化MoE层:")
        print(f"   专家数量: {num_experts}")
        print(f"   专家类型: {expert_types[:num_experts]}")
        print(f"   条件类别数: {num_conditions}")
        print(f"   Top-k选择: {self.top_k}")
        
    def forward(self, x, condition_id=None, return_expert_outputs=False):
        """
        Args:
            x: [B, C, H, W] - 输入的变化特征（来自第四阶段）
            condition_id: [B] - 滑坡诱因类别ID
            return_expert_outputs: 是否返回各专家的单独输出（用于分析）
        
        Returns:
            output: [B, C, H, W] - 最终的变化特征
            aux_loss: 辅助损失（负载均衡）
        """
        B, C, H, W = x.shape
        
        # 获取门控权重
        gates, load_balance_loss = self.gating(x, condition_id)  # [B, num_experts]
        
        # Top-k选择
        if self.top_k < self.num_experts:
            topk_gates, topk_indices = torch.topk(gates, self.top_k, dim=1)
            # 归一化top-k权重
            topk_gates = topk_gates / topk_gates.sum(dim=1, keepdim=True)
        else:
            topk_gates = gates
            topk_indices = torch.arange(self.num_experts).expand(B, -1).to(x.device)
        
        # 收集专家输出
        expert_outputs = []
        expert_attentions = []
        
        for i in range(self.num_experts):
            # 创建专家掩码
            expert_mask = (topk_indices == i).any(dim=1)
            
            if expert_mask.any():
                # 获取该专家的置信度
                confidence = gates[:, i].view(B, 1, 1, 1)
                
                # 运行专家网络
                expert_out, expert_attn = self.experts[i](x, confidence)
                expert_outputs.append(expert_out)
                expert_attentions.append(expert_attn)
            else:
                # 如果该专家未被选中，填充零
                expert_outputs.append(torch.zeros_like(x))
                expert_attentions.append(torch.zeros(B, 1, H, W, device=x.device))
        
        # 组合专家输出
        if self.top_k < self.num_experts:
            # 只组合top-k专家
            output = torch.zeros_like(x)
            for b in range(B):
                for k in range(self.top_k):
                    expert_idx = topk_indices[b, k]
                    weight = topk_gates[b, k]
                    output[b] += weight * expert_outputs[expert_idx][b]
        else:
            # 组合所有专家
            expert_outputs_stack = torch.stack(expert_outputs, dim=1)  # [B, num_experts, C, H, W]
            gates_expanded = gates.view(B, self.num_experts, 1, 1, 1)
            output = (expert_outputs_stack * gates_expanded).sum(dim=1)
        
        # 准备返回值
        if return_expert_outputs:
            return output, load_balance_loss, {
                'expert_outputs': expert_outputs,
                'expert_attentions': expert_attentions,
                'gates': gates,
                'topk_indices': topk_indices if self.top_k < self.num_experts else None
            }
        
        return output, load_balance_loss


class MultiScaleMoE(nn.Module):
    """
    多尺度MoE层
    为不同尺度配置不同的专家组合
    """
    def __init__(self, scale_dims, num_experts=4, num_conditions=5,
                 scale_specific_experts=None, temperature=1.0,
                 use_load_balancing=True, dropout=0.1):
        super().__init__()
        
        self.scale_dims = scale_dims
        self.moe_layers = nn.ModuleList()
        
        # 为每个尺度创建MoE层
        for i, dim in enumerate(scale_dims):
            # 可以为不同尺度配置不同的专家类型
            if scale_specific_experts and i < len(scale_specific_experts):
                expert_types = scale_specific_experts[i]
            else:
                # 默认专家类型
                expert_types = ["rainfall", "earthquake", "human", "complex"]
            
            moe = ConditionalMoE(
                dim, num_experts, num_conditions,
                expert_types=expert_types,
                hidden_dim=min(256, dim * 2),  # 根据通道数调整隐藏维度
                temperature=temperature,
                use_load_balancing=use_load_balancing,
                dropout=dropout
            )
            
            self.moe_layers.append(moe)
    
    def forward(self, x, condition_id, scale_idx):
        """
        根据尺度选择对应的MoE层
        
        Args:
            x: [B, C, H, W] - 输入特征
            condition_id: [B] - 条件ID
            scale_idx: 当前尺度索引
        """
        if scale_idx < len(self.moe_layers):
            return self.moe_layers[scale_idx](x, condition_id)
        else:
            # 超出范围，使用最后一个
            return self.moe_layers[-1](x, condition_id)


# 定义滑坡诱因类别
LANDSLIDE_TRIGGERS = {
    0: "rainfall",      # 降雨引发
    1: "earthquake",    # 地震引发  
    2: "human",         # 人类活动引发
    3: "snowmelt",      # 融雪引发
    4: "complex"        # 复合因素
}


def create_condition_embedding(batch_size, trigger_type, device):
    """
    创建条件嵌入的辅助函数
    
    Args:
        batch_size: 批次大小
        trigger_type: 触发类型（int或str）
        device: 设备
    """
    if isinstance(trigger_type, str):
        # 将字符串转换为ID
        trigger_id = [k for k, v in LANDSLIDE_TRIGGERS.items() if v == trigger_type][0]
    else:
        trigger_id = trigger_type
    
    condition_ids = torch.full((batch_size,), trigger_id, dtype=torch.long, device=device)
    return condition_ids