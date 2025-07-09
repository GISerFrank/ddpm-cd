import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGuidedAttention(nn.Module):
    """
    物理引导的注意力模块（策略一）
    使用物理特征作为注意力权重来增强视觉特征
    """
    
    def __init__(self, visual_channels, physical_channels, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        self.visual_channels = visual_channels
        self.physical_channels = physical_channels
        self.hidden_dim = hidden_dim
        
        # 物理特征编码器
        self.physics_encoder = nn.Sequential(
            nn.Conv2d(physical_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力生成器
        self.attention_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征调制器
        self.feature_modulator = nn.Sequential(
            nn.Conv2d(visual_channels + hidden_dim, visual_channels, 1),
            nn.BatchNorm2d(visual_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, visual_features, physical_features):
        """
        Args:
            visual_features: [B, C_visual, H, W] - 来自DDPM的视觉特征
            physical_features: [B, C_physical, H, W] - 物理数据（DEM, slope等）
            
        Returns:
            enhanced_features: [B, C_visual, H, W] - 物理引导增强后的特征
        """
        B, C, H, W = visual_features.shape
        
        # 如果尺寸不匹配，调整物理特征尺寸
        if physical_features.shape[-2:] != (H, W):
            physical_features = F.interpolate(
                physical_features, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 编码物理特征
        physics_encoded = self.physics_encoder(physical_features)
        physics_encoded = self.dropout(physics_encoded)
        
        # 生成空间注意力图
        attention_map = self.attention_conv(physics_encoded)  # [B, 1, H, W]
        
        # 应用注意力到视觉特征
        attended_visual = visual_features * attention_map
        
        # 将物理特征信息融合到视觉特征中
        combined = torch.cat([attended_visual, physics_encoded], dim=1)
        enhanced_features = self.feature_modulator(combined)
        
        # 残差连接
        enhanced_features = enhanced_features + visual_features
        
        return enhanced_features, attention_map


class MultiScalePhysicsAttention(nn.Module):
    """
    多尺度物理引导注意力
    为不同分辨率的特征图使用不同的注意力模块
    """
    
    def __init__(self, feat_scales, inner_channel, channel_multiplier, 
                 physical_channels=2, hidden_dim=64):
        super().__init__()
        
        self.feat_scales = feat_scales
        self.attention_modules = nn.ModuleDict()
        
        # 为每个特征尺度创建注意力模块
        for scale in feat_scales:
            visual_channels = self._get_channels_for_scale(scale, inner_channel, channel_multiplier)
            self.attention_modules[str(scale)] = PhysicsGuidedAttention(
                visual_channels=visual_channels,
                physical_channels=physical_channels,
                hidden_dim=hidden_dim
            )
    
    def _get_channels_for_scale(self, scale, inner_channel, channel_multiplier):
        """根据尺度计算通道数"""
        if scale < 3:  # 256x256
            return inner_channel * channel_multiplier[0]
        elif scale < 6:  # 128x128
            return inner_channel * channel_multiplier[1]
        elif scale < 9:  # 64x64
            return inner_channel * channel_multiplier[2]
        elif scale < 12:  # 32x32
            return inner_channel * channel_multiplier[3]
        elif scale < 15:  # 16x16
            return inner_channel * channel_multiplier[4]
        else:
            raise ValueError(f"Unknown scale: {scale}")
    
    def forward(self, features_dict, physical_features):
        """
        Args:
            features_dict: {scale: features} - 不同尺度的特征
            physical_features: [B, C_physical, H, W] - 原始分辨率的物理特征
            
        Returns:
            enhanced_features_dict: 增强后的特征字典
            attention_maps: 各尺度的注意力图
        """
        enhanced_features = {}
        attention_maps = {}
        
        for scale, features in features_dict.items():
            if str(scale) in self.attention_modules:
                enhanced_feat, attn_map = self.attention_modules[str(scale)](
                    features, physical_features
                )
                enhanced_features[scale] = enhanced_feat
                attention_maps[scale] = attn_map
            else:
                # 如果没有对应的注意力模块，直接传递特征
                enhanced_features[scale] = features
        
        return enhanced_features, attention_maps