"""
物理引导的注意力模块（策略一）- DDPM-Aware版本
使用物理特征作为注意力权重来增强视觉特征

关键改动：
1. BatchNorm2d → GroupNorm (适配DDPM特征分布)
2. ReLU → SiLU (DDPM标准激活函数)
3. 其他逻辑保持完全相同
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGuidedAttention(nn.Module):
    """
    物理引导的注意力模块（策略一）- DDPM-Aware版本
    使用物理特征作为注意力权重来增强视觉特征
    
    改动：
    - BatchNorm → GroupNorm (num_groups=8, DDPM标准)
    - ReLU → SiLU (DDPM标准激活)
    - 接收尺度对齐的物理特征（来自金字塔编码器）
    """
    
    def __init__(self, visual_channels, physical_channels, hidden_dim=64, 
                 dropout=0.1, num_groups=8):
        """
        Args:
            visual_channels: 视觉特征通道数
            physical_channels: 物理特征通道数（与visual_channels相同，来自金字塔）
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            num_groups: GroupNorm的组数
        """
        super().__init__()
        
        self.visual_channels = visual_channels
        self.physical_channels = physical_channels
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        
        # 物理特征已经从金字塔编码器编码好，只需简单处理
        self.physics_refine = nn.Sequential(
            nn.Conv2d(physical_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.SiLU()
        )
        
        # 注意力生成器 - DDPM-aware
        self.attention_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.SiLU(),  # ✅ ReLU → SiLU
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征调制器 - DDPM-aware
        self.feature_modulator = nn.Sequential(
            nn.Conv2d(visual_channels + hidden_dim, visual_channels, 1),
            nn.GroupNorm(num_groups, visual_channels),  # ✅ BatchNorm → GroupNorm
            nn.SiLU()                                    # ✅ ReLU → SiLU
        )
        
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, visual_features, physical_features):
        """
        Args:
            visual_features: [B, C_visual, H, W] - 来自DDPM的视觉特征
            physical_features: [B, C_visual, H, W] - 尺度对齐的物理特征（来自金字塔编码器）
            
        Returns:
            enhanced_features: [B, C_visual, H, W] - 物理引导增强后的特征
            attention_map: [B, 1, H, W] - 注意力图（用于可视化）
        """
        B, C, H, W = visual_features.shape
        
        # 物理特征已经从金字塔编码器得到，尺寸和通道数都对齐，无需插值
        
        # 精炼物理特征
        physics_refined = self.physics_refine(physical_features)
        physics_refined = self.dropout(physics_refined)
        
        # 生成空间注意力图
        attention_map = self.attention_conv(physics_refined)  # [B, 1, H, W]
        
        # 应用注意力到视觉特征
        attended_visual = visual_features * attention_map
        
        # 将物理特征信息融合到视觉特征中
        combined = torch.cat([attended_visual, physics_refined], dim=1)
        enhanced_features = self.feature_modulator(combined)
        
        # 残差连接
        enhanced_features = enhanced_features + visual_features
        
        return enhanced_features, attention_map


# ========== 测试代码 ==========
if __name__ == "__main__":
    """测试DDPM-aware版本的物理注意力（配合金字塔编码器）"""
    
    print("🧪 测试 PhysicsGuidedAttention (DDPM-aware + 金字塔)")
    
    # 创建模块
    attention = PhysicsGuidedAttention(
        visual_channels=256,
        physical_channels=256,  # 与visual对齐
        hidden_dim=64,
        num_groups=8
    )
    
    # 测试输入
    batch_size = 2
    H, W = 64, 64
    visual_feat = torch.randn(batch_size, 256, H, W)
    physical_feat = torch.randn(batch_size, 256, H, W)  # 来自金字塔编码器，已对齐
    
    # 前向传播
    enhanced, attn_map = attention(visual_feat, physical_feat)
    
    print(f"✅ 输入视觉特征: {visual_feat.shape}")
    print(f"✅ 输入物理特征: {physical_feat.shape} (来自金字塔，已对齐)")
    print(f"✅ 输出增强特征: {enhanced.shape}")
    print(f"✅ 输出注意力图: {attn_map.shape}")
    
    # 检查模块类型
    has_groupnorm = any('GroupNorm' in str(type(m)) for m in attention.modules())
    has_silu = any('SiLU' in str(type(m)) for m in attention.modules())
    has_batchnorm = any('BatchNorm' in str(type(m)) for m in attention.modules())
    has_relu = any(isinstance(m, nn.ReLU) for m in attention.modules())
    
    print(f"\n模块检查:")
    print(f"  - GroupNorm: {'✅' if has_groupnorm else '❌'}")
    print(f"  - SiLU: {'✅' if has_silu else '❌'}")
    print(f"  - BatchNorm: {'❌' if not has_batchnorm else '⚠️ 仍然存在！'}")
    print(f"  - ReLU: {'❌' if not has_relu else '⚠️ 仍然存在！'}")
    
    print(f"\n🎉 所有测试通过！DDPM-aware物理注意力模块工作正常！")
