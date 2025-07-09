"""
交叉注意力模块 - 用于智能的交互式变化对比
不是简单的数学相减，而是通过注意力机制理解上下文中的重要差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块 - 实现时相特征之间的智能对比
    
    特点：
    1. 使用多头注意力机制捕捉复杂的时相关系
    2. 包含位置编码以保持空间信息
    3. 可选的残差连接和层归一化
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., 
                 proj_drop=0., use_pos_encoding=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pos_encoding = use_pos_encoding
        
        # Query来自时相A，Key和Value来自时相B
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 位置编码（可选）
        if use_pos_encoding:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, feat_a, feat_b):
        """
        Args:
            feat_a: [B, C, H, W] - 时相A的特征（作为Query）
            feat_b: [B, C, H, W] - 时相B的特征（作为Key和Value）
        
        Returns:
            diff_features: [B, C, H, W] - 交叉注意力后的差异特征
        """
        B, C, H, W = feat_a.shape
        
        # 重塑为序列格式 [B, HW, C]
        feat_a_seq = feat_a.flatten(2).transpose(1, 2)
        feat_b_seq = feat_b.flatten(2).transpose(1, 2)
        
        # 归一化
        feat_a_seq = self.norm1(feat_a_seq)
        feat_b_seq = self.norm2(feat_b_seq)
        
        # 添加位置编码
        if self.use_pos_encoding:
            feat_a_seq = feat_a_seq + self.pos_embed
            feat_b_seq = feat_b_seq + self.pos_embed
        
        # 计算Q, K, V
        q = self.q(feat_a_seq).reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(feat_b_seq).reshape(B, H*W, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力到值
        x = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 重塑回原始形状
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class TemporalCrossAttention(nn.Module):
    """
    时序交叉注意力模块 - 专门用于变化检测
    
    通过交叉注意力机制，让模型学习"什么是重要的变化"
    而不是简单的像素级差异
    """
    
    def __init__(self, in_channels, hidden_dim=None, num_heads=8, 
                 use_diff_enhance=True, dropout=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim or in_channels
        self.use_diff_enhance = use_diff_enhance
        
        # 特征投影
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 交叉注意力
        self.cross_attn = CrossAttentionBlock(
            dim=self.hidden_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        # 差异增强模块（可选）
        if use_diff_enhance:
            self.diff_enhance = nn.Sequential(
                nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 3, padding=1),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
                nn.BatchNorm2d(self.hidden_dim)
            )
        
        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, feat_a, feat_b):
        """
        智能的交互式对比
        
        Args:
            feat_a: [B, C, H, W] - 时相A的特征
            feat_b: [B, C, H, W] - 时相B的特征
            
        Returns:
            diff_features: [B, C, H, W] - 语义丰富的变化特征
        """
        # 特征投影
        feat_a_proj = self.feat_proj(feat_a)
        feat_b_proj = self.feat_proj(feat_b)
        
        # 双向交叉注意力
        # A查询B：理解B相对于A的变化
        attn_a2b = self.cross_attn(feat_a_proj, feat_b_proj)
        # B查询A：理解A相对于B的状态
        attn_b2a = self.cross_attn(feat_b_proj, feat_a_proj)
        
        # 智能差异计算
        if self.use_diff_enhance:
            # 拼接双向注意力结果
            diff_concat = torch.cat([attn_a2b, attn_b2a], dim=1)
            # 通过学习的方式融合差异
            diff_features = self.diff_enhance(diff_concat)
        else:
            # 简单的差异计算
            diff_features = attn_a2b - attn_b2a
        
        # 输出投影
        diff_features = self.out_proj(diff_features)
        
        # 残差连接
        diff_features = diff_features + (feat_a - feat_b)
        
        return diff_features


class MultiScaleCrossAttention(nn.Module):
    """
    多尺度交叉注意力模块
    
    为不同尺度的特征配置不同的注意力参数
    """
    
    def __init__(self, scale_dims, num_heads_list=None, dropout=0.1):
        super().__init__()
        
        self.scale_dims = scale_dims
        if num_heads_list is None:
            # 默认配置：深层用少的头，浅层用多的头
            num_heads_list = [4, 4, 8, 8, 8]
        
        self.cross_attns = nn.ModuleList()
        
        for i, dim in enumerate(scale_dims):
            num_heads = num_heads_list[i] if i < len(num_heads_list) else 8
            self.cross_attns.append(
                TemporalCrossAttention(
                    in_channels=dim,
                    num_heads=num_heads,
                    use_diff_enhance=True,
                    dropout=dropout
                )
            )
    
    def forward(self, feat_a, feat_b, scale_idx):
        """
        根据尺度索引选择对应的交叉注意力模块
        
        Args:
            feat_a, feat_b: 特征对
            scale_idx: 当前尺度的索引
        """
        if scale_idx < len(self.cross_attns):
            return self.cross_attns[scale_idx](feat_a, feat_b)
        else:
            # 如果超出范围，使用最后一个
            return self.cross_attns[-1](feat_a, feat_b)