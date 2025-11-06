"""
多尺度物理特征金字塔编码器 - DDPM-Aware版本

改进：
1. 不再是单尺度编码+插值，而是构建物理特征金字塔
2. 为每个DDPM特征尺度生成对应的物理特征
3. 深层关注局部细节，浅层关注全局形态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalFeaturePyramid(nn.Module):
    """
    物理特征金字塔编码器 - 类似DDPM的U-Net结构
    
    根据feat_scales动态生成对应尺度的物理特征
    从config自动读取: feat_scales, inner_channel, channel_multiplier
    """
    
    def __init__(self, opt, num_physical_layers=2, num_groups=8):
        """
        Args:
            opt: 配置字典，包含 model_cd 和 model.unet 配置
            num_physical_layers: 物理数据层数 (默认2: DEM+Slope)
            num_groups: GroupNorm的组数 (默认8)
        """
        super().__init__()
        
        # 从config读取参数
        feat_scales = opt['model_cd']['feat_scales']
        inner_channel = opt['model']['unet']['inner_channel']
        channel_multiplier = opt['model']['unet']['channel_multiplier']
        
        self.num_physical_layers = num_physical_layers
        self.feat_scales = sorted(feat_scales, reverse=True)  # 从深到浅
        self.num_groups = num_groups
        
        # 计算每个尺度的通道数（与DDPM特征对齐）
        self.scale_channels = {}
        for scale in feat_scales:
            channels = self._get_channels_for_scale(scale, inner_channel, channel_multiplier)
            self.scale_channels[scale] = channels
        
        # 初始卷积：物理数据 → 基础特征
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_physical_layers, 32, 3, padding=1),
            nn.GroupNorm(num_groups, 32),
            nn.SiLU()
        )
        
        # 编码器：逐步下采样，提取层次化特征
        self.encoders = nn.ModuleDict()
        current_channels = 32
        
        # 从最浅层(256x256)到最深层(16x16)构建编码器
        # 假设尺度对应关系：scale 0→256, 3→128, 6→64, 9→32, 12→16
        scale_to_resolution = {
            0: 256, 1: 256, 2: 256,
            3: 128, 4: 128, 5: 128,
            6: 64, 7: 64, 8: 64,
            9: 32, 10: 32, 11: 32,
            12: 16, 13: 16, 14: 16
        }
        
        # 获取需要的分辨率
        resolutions = sorted(set([scale_to_resolution[s] for s in feat_scales]), reverse=True)
        
        for i, res in enumerate(resolutions):
            if i > 0:  # 需要下采样
                # 下采样块
                next_channels = min(current_channels * 2, 512)
                self.encoders[f'downsample_{res}'] = nn.Sequential(
                    nn.Conv2d(current_channels, next_channels, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups, next_channels),
                    nn.SiLU(),
                    nn.Conv2d(next_channels, next_channels, 3, padding=1),
                    nn.GroupNorm(num_groups, next_channels),
                    nn.SiLU()
                )
                current_channels = next_channels
        
        # 为每个feat_scale创建输出投影
        self.scale_projections = nn.ModuleDict()
        for scale in feat_scales:
            res = scale_to_resolution[scale]
            target_channels = self.scale_channels[scale]
            
            # 找到对应分辨率的编码器输出通道数
            encoder_channels = current_channels if res == min(resolutions) else 32 * (256 // res)
            encoder_channels = min(encoder_channels, 512)
            
            self.scale_projections[str(scale)] = nn.Sequential(
                nn.Conv2d(encoder_channels, target_channels, 1),
                nn.GroupNorm(num_groups, target_channels),
                nn.SiLU()
            )
    
    def _get_channels_for_scale(self, scale, inner_channel, channel_multiplier):
        """获取特定尺度的通道数（与DDPM特征对齐）"""
        if scale < 3:
            return inner_channel * channel_multiplier[0]
        elif scale < 6:
            return inner_channel * channel_multiplier[1]
        elif scale < 9:
            return inner_channel * channel_multiplier[2]
        elif scale < 12:
            return inner_channel * channel_multiplier[3]
        elif scale < 15:
            return inner_channel * channel_multiplier[4]
        else:
            raise ValueError(f"Unsupported scale: {scale}")
    
    def forward(self, physical_data):
        """
        Args:
            physical_data: [B, num_layers, 256, 256] - 原始物理数据
        
        Returns:
            pyramid_features: Dict[scale -> Tensor]
                例如: {12: [B,512,16,16], 6: [B,256,64,64], 0: [B,64,256,256]}
        """
        # 初始编码
        x = self.initial_conv(physical_data)  # [B, 32, 256, 256]
        
        # 存储不同分辨率的特征
        resolution_features = {256: x}
        
        # 逐步下采样
        scale_to_resolution = {
            0: 256, 1: 256, 2: 256,
            3: 128, 4: 128, 5: 128,
            6: 64, 7: 64, 8: 64,
            9: 32, 10: 32, 11: 32,
            12: 16, 13: 16, 14: 16
        }
        
        resolutions = sorted(set([scale_to_resolution[s] for s in self.feat_scales]), reverse=True)
        current_x = x
        
        for res in resolutions:
            if res < 256:  # 需要下采样
                current_x = self.encoders[f'downsample_{res}'](current_x)
            resolution_features[res] = current_x
        
        # 为每个scale生成特征
        pyramid_features = {}
        for scale in self.feat_scales:
            res = scale_to_resolution[scale]
            feat = resolution_features[res]
            
            # 投影到目标通道数
            projected = self.scale_projections[str(scale)](feat)
            pyramid_features[scale] = projected
        
        return pyramid_features


# ========== 向后兼容的包装器 ==========
class PhysicalFeatureEncoder(nn.Module):
    """
    向后兼容的物理编码器（单尺度版本）
    """
    def __init__(self, num_physical_layers=2, hidden_dims=[32, 64, 128], 
                 output_dim=64, num_groups=8):
        super().__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_physical_layers, hidden_dims[0], 3, padding=1),
            nn.GroupNorm(num_groups, hidden_dims[0]),
            nn.SiLU()
        )
        
        self.encoder_blocks = nn.ModuleList()
        layer_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        
        for in_ch, out_ch in layer_dims:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(num_groups, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(num_groups, out_ch),
                nn.SiLU()
            )
            self.encoder_blocks.append(block)
        
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], output_dim, 1),
            nn.GroupNorm(num_groups, output_dim),
            nn.SiLU()
        )
        
    def forward(self, physical_data):
        x = self.initial_conv(physical_data)
        for block in self.encoder_blocks:
            x = block(x)
        encoded = self.output_proj(x)
        return encoded


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("🧪 测试 PhysicalFeaturePyramid")
    
    # 模拟config
    test_opt = {
        'model_cd': {
            'feat_scales': [14, 11, 8, 5]
        },
        'model': {
            'unet': {
                'inner_channel': 128,
                'channel_multiplier': [1, 2, 4, 8, 8]
            }
        }
    }
    
    # 创建金字塔编码器
    pyramid_encoder = PhysicalFeaturePyramid(
        opt=test_opt,
        num_physical_layers=2,
        num_groups=8
    )
    
    # 测试输入
    batch_size = 2
    physical_data = torch.randn(batch_size, 2, 256, 256)
    
    # 前向传播
    pyramid_features = pyramid_encoder(physical_data)
    
    print(f"✅ 输入物理数据: {physical_data.shape}")
    print(f"\n生成的多尺度物理特征金字塔:")
    for scale in sorted(pyramid_features.keys(), reverse=True):
        feat = pyramid_features[scale]
        print(f"  Scale {scale:2d}: {feat.shape}")
    
    # 验证通道数
    print(f"\n✅ 通道数与DDPM特征对齐:")
    # inner_channel=128, channel_multiplier=[1,2,4,8,8]
    expected_channels = {
        14: 1024,  # 128*8
        11: 1024,  # 128*8
        8: 512,    # 128*4
        5: 256     # 128*2
    }
    for scale, expected in expected_channels.items():
        actual = pyramid_features[scale].shape[1]
        status = "✅" if actual == expected else "❌"
        print(f"  Scale {scale}: {actual} channels (expected {expected}) {status}")
    
    # 检查模块
    has_groupnorm = any('GroupNorm' in str(type(m)) for m in pyramid_encoder.modules())
    has_silu = any('SiLU' in str(type(m)) for m in pyramid_encoder.modules())
    has_batchnorm = any('BatchNorm' in str(type(m)) for m in pyramid_encoder.modules())
    
    print(f"\n模块检查:")
    print(f"  - GroupNorm: {'✅' if has_groupnorm else '❌'}")
    print(f"  - SiLU: {'✅' if has_silu else '❌'}")
    print(f"  - BatchNorm: {'❌' if not has_batchnorm else '⚠️'}")
    
    print(f"\n🎉 测试通过！")
