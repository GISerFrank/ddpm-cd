"""
Mamba混合器模块 - 用于全局形态学分析
通过状态空间模型捕捉变化的全局空间结构和长距离依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class S6(nn.Module):
    """
    S6核心层 - Mamba的核心组件
    实现选择性状态空间模型
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        if dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # 激活函数
        self.act = nn.SiLU()
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # 初始化特殊的dt投影
        dt_init_std = self.dt_rank**-0.5 * 2
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # 初始化A和D矩阵
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) 输入序列
        Returns:
            y: (B, L, D) 输出序列
        """
        batch, seqlen, dim = x.shape
        
        # 输入投影和分割
        x_and_res = self.in_proj(x)  # (B, L, 2*D_inner)
        x, res = x_and_res.split(self.d_inner, dim=-1)
        
        # 卷积
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seqlen]
        x = rearrange(x, 'b d l -> b l d')
        
        # 激活
        x = self.act(x)
        
        # SSM处理
        y = self.ssm(x)
        
        # 门控和输出
        y = y * self.act(res)
        output = self.out_proj(y)
        
        return output

    def ssm(self, x):
        """选择性扫描算法"""
        batch, seqlen, dim = x.shape
        
        # 计算∆, B, C
        deltaBC = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # 计算离散化的∆
        delta = F.softplus(self.dt_proj(delta))  # (B, L, D_inner)
        
        # 计算离散化的A
        A = -torch.exp(self.A_log)  # (D_inner, d_state)
        
        # 执行选择性扫描
        y = self.selective_scan(x, delta, A, B, C, self.D)
        
        return y
    
    def selective_scan(self, u, delta, A, B, C, D):
        """选择性扫描的简化实现"""
        batch, seqlen, dim = u.shape
        d_state = A.shape[1]
        
        # 离散化A和B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D_inner, d_state)
        
        # 初始化状态
        x = torch.zeros(batch, dim, d_state, device=u.device)
        ys = []
        
        # 扫描
        for i in range(seqlen):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i:i+1].transpose(1, 2)
            y = (x @ C[:, i, :].unsqueeze(-1)).squeeze(-1)
            ys.append(y + D * u[:, i].transpose(0, 1))
        
        y = torch.stack(ys, dim=1).transpose(1, 2)
        return y


class MambaBlock(nn.Module):
    """
    Mamba块 - 包含归一化和残差连接
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s6 = S6(d_model, d_state, d_conv, expand, dt_rank)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D) 输入序列
        Returns:
            output: (B, L, D) 输出序列
        """
        output = self.s6(self.norm(x)) + x
        return output


class SpatialMamba(nn.Module):
    """
    空间Mamba模块 - 专门用于处理2D空间特征
    通过多方向扫描捕捉全局空间依赖
    """
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2, 
                 n_layers=2, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.bidirectional = bidirectional
        
        # 特征投影
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Mamba层
        self.mamba_layers = nn.ModuleList([
            MambaBlock(in_channels, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # 如果使用双向扫描，需要额外的反向Mamba
        if bidirectional:
            self.mamba_layers_rev = nn.ModuleList([
                MambaBlock(in_channels, d_state, d_conv, expand)
                for _ in range(n_layers)
            ])
        
        # 输出投影
        out_channels = in_channels * 2 if bidirectional else in_channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            output: (B, C, H, W) 输出特征图
        """
        B, C, H, W = x.shape
        
        # 特征投影
        x = self.feat_proj(x)
        
        # 转换为序列格式 (B, H*W, C)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        
        # 前向Mamba处理
        x_forward = x_seq
        for layer in self.mamba_layers:
            x_forward = layer(x_forward)
        
        if self.bidirectional:
            # 反向Mamba处理
            x_backward = torch.flip(x_seq, dims=[1])
            for layer in self.mamba_layers_rev:
                x_backward = layer(x_backward)
            x_backward = torch.flip(x_backward, dims=[1])
            
            # 合并双向特征
            x_combined = torch.cat([x_forward, x_backward], dim=-1)
        else:
            x_combined = x_forward
        
        # 转换回2D格式
        x_2d = rearrange(x_combined, 'b (h w) c -> b c h w', h=H, w=W)
        
        # 输出投影
        output = self.out_proj(x_2d)
        
        # 残差连接
        output = output + x
        
        return output


class MultiDirectionalMamba(nn.Module):
    """
    多方向Mamba模块
    通过水平、垂直和对角线扫描全面捕捉空间模式
    """
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2, n_layers=1):
        super().__init__()
        
        # 四个方向的Mamba：水平、垂直、主对角线、副对角线
        self.horizontal_mamba = SpatialMamba(
            in_channels, d_state, d_conv, expand, n_layers, bidirectional=True
        )
        
        self.vertical_mamba = SpatialMamba(
            in_channels, d_state, d_conv, expand, n_layers, bidirectional=True
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        """
        多方向扫描并融合
        """
        # 水平扫描
        h_out = self.horizontal_mamba(x)
        
        # 垂直扫描（转置处理）
        x_transposed = x.transpose(2, 3)
        v_out = self.vertical_mamba(x_transposed)
        v_out = v_out.transpose(2, 3)
        
        # 融合多方向特征
        combined = torch.cat([h_out, v_out], dim=1)
        output = self.fusion(combined)
        
        # 残差连接
        output = output + x
        
        return output


class ChangeDetectionMamba(nn.Module):
    """
    专门用于变化检测的Mamba模块
    理解变化的全局形态和空间结构
    """
    def __init__(self, scale_dims, d_state=16, d_conv=4, expand=2, 
                 n_layers=2, use_multi_direction=True):
        super().__init__()
        
        self.scale_dims = scale_dims
        self.use_multi_direction = use_multi_direction
        
        # 为每个尺度创建Mamba模块
        self.mamba_modules = nn.ModuleList()
        
        for dim in scale_dims:
            if use_multi_direction:
                # 使用多方向Mamba
                mamba = MultiDirectionalMamba(
                    dim, d_state, d_conv, expand, n_layers
                )
            else:
                # 使用标准空间Mamba
                mamba = SpatialMamba(
                    dim, d_state, d_conv, expand, n_layers, bidirectional=True
                )
            
            self.mamba_modules.append(mamba)
    
    def forward(self, diff_features, scale_idx):
        """
        对变化特征进行全局形态分析
        
        Args:
            diff_features: (B, C, H, W) 初步的变化特征
            scale_idx: 当前尺度索引
        
        Returns:
            diff_mamba: (B, C, H, W) 经过全局分析的变化特征
        """
        if scale_idx < len(self.mamba_modules):
            return self.mamba_modules[scale_idx](diff_features)
        else:
            # 超出范围，使用最后一个
            return self.mamba_modules[-1](diff_features)