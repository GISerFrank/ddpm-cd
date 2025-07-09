import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cd_modules.psp import _PSPModule
from model.cd_modules.se import ChannelSpatialSELayer
from model.cd_modules.physics_attention import PhysicsGuidedAttention, MultiScalePhysicsAttention
from model.cd_modules.physical_encoder import PhysicalFeatureEncoder
from model.cd_modules.cross_attention import TemporalCrossAttention, MultiScaleCrossAttention
from model.cd_modules.mamba_mixer import ChangeDetectionMamba


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''Get the number of input channels for each scale'''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:  # 256 x 256
            in_channels += inner_channel * channel_multiplier[0]
        elif scale < 6:  # 128 x 128
            in_channels += inner_channel * channel_multiplier[1]
        elif scale < 9:  # 64 x 64
            in_channels += inner_channel * channel_multiplier[2]
        elif scale < 12:  # 32 x 32
            in_channels += inner_channel * channel_multiplier[3]
        elif scale < 15:  # 16 x 16
            in_channels += inner_channel * channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14')
    return in_channels


class PhysicsEnhancedBlock(nn.Module):
    """
    增强的Block，集成物理引导注意力（第一阶段）
    """
    def __init__(self, dim, dim_out, time_steps, physics_attention=None):
        super().__init__()
        # 原始的时间步拼接和处理
        self.time_fusion = nn.Sequential(
            nn.Conv2d(dim * len(time_steps), dim, 1) if len(time_steps) > 1 else nn.Identity(),
            nn.ReLU() if len(time_steps) > 1 else nn.Identity(),
        )
        
        # 物理引导注意力（策略一）
        self.physics_attention = physics_attention
        
        # 特征处理
        self.feature_conv = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, physical_features=None):
        # 时间步融合
        x = self.time_fusion(x)
        
        # 应用物理引导注意力（如果可用）
        if self.physics_attention is not None and physical_features is not None:
            x, _ = self.physics_attention(x, physical_features)
        
        # 特征处理
        x = self.feature_conv(x)
        return x


class AttentionBlock(nn.Module):
    """保持原有的AttentionBlock结构"""
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)


class cd_head_v8(nn.Module):
    '''
    Change detection head (version 8) - 五阶段智能推理
    第一阶段：物理引导的状态理解 ✓
    第二阶段：智能的交互式对比（交叉注意力）✓
    第三阶段：全局形态学分析（Mamba）✓
    第四阶段：物理引导的变化聚焦 - TODO
    第五阶段：条件化专家决策（MoE）- TODO
    '''

    def __init__(self, feat_scales, out_channels=2, inner_channel=None, 
                 channel_multiplier=None, img_size=256, time_steps=None,
                 physics_config=None, cross_attention_config=None, 
                 mamba_config=None):
        super(cd_head_v8, self).__init__()

        # 基础参数设置
        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.img_size = img_size
        self.time_steps = time_steps if time_steps is not None else [0]
        
        # 物理配置
        self.physics_config = physics_config or {}
        self.use_physics = self.physics_config.get('enabled', False)
        
        # 交叉注意力配置
        self.cross_attention_config = cross_attention_config or {}
        self.use_cross_attention = self.cross_attention_config.get('enabled', True)
        
        # Mamba配置
        self.mamba_config = mamba_config or {}
        self.use_mamba = self.mamba_config.get('enabled', True)
        
        # 物理特征编码器
        if self.use_physics:
            self.physical_encoder = PhysicalFeatureEncoder(
                input_channels=self.physics_config.get('num_physical_layers', 2),
                output_channels=self.physics_config.get('physical_embedding_dim', 64)
            )
        
        # 计算每个尺度的维度
        self.scale_dims = []
        for scale in self.feat_scales:
            dim = get_in_channels([scale], inner_channel, channel_multiplier)
            self.scale_dims.append(dim)
        
        # 多尺度交叉注意力模块（第二阶段）
        if self.use_cross_attention:
            self.cross_attention = MultiScaleCrossAttention(
                scale_dims=self.scale_dims,
                num_heads_list=self.cross_attention_config.get('num_heads_list', None),
                dropout=self.cross_attention_config.get('dropout', 0.1)
            )
        
        # Mamba全局形态分析模块（第三阶段）
        if self.use_mamba:
            self.mamba_mixer = ChangeDetectionMamba(
                scale_dims=self.scale_dims,
                d_state=self.mamba_config.get('d_state', 16),
                d_conv=self.mamba_config.get('d_conv', 4),
                expand=self.mamba_config.get('expand', 2),
                n_layers=self.mamba_config.get('n_layers', 2),
                use_multi_direction=self.mamba_config.get('use_multi_direction', True)
            )
        
        # 构建解码器
        self.decoder = nn.ModuleList()
        current_decoder_output_channels = 0
        
        for i in range(len(self.feat_scales)):
            dim = self.scale_dims[i]
            
            # 为每个尺度创建物理引导注意力模块（第一阶段）
            physics_attention = None
            if self.use_physics:
                physics_attention = PhysicsGuidedAttention(
                    visual_channels=dim,
                    physical_channels=self.physics_config.get('physical_embedding_dim', 64),
                    hidden_dim=self.physics_config.get('physical_embedding_dim', 64),
                    dropout=self.physics_config.get('dropout', 0.1)
                )
            
            # 使用增强的Block
            self.decoder.append(
                PhysicsEnhancedBlock(
                    dim=dim, 
                    dim_out=dim, 
                    time_steps=self.time_steps,
                    physics_attention=physics_attention
                )
            )
            current_block_output_channels = dim

            if i != len(self.feat_scales) - 1:
                dim_out_for_attention = get_in_channels(
                    [self.feat_scales[i + 1]], inner_channel, channel_multiplier
                )
                self.decoder.append(
                    AttentionBlock(dim=current_block_output_channels, dim_out=dim_out_for_attention)
                )
                current_decoder_output_channels = dim_out_for_attention
            else:
                current_decoder_output_channels = current_block_output_channels

        # 最终分类头
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(current_decoder_output_channels, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, feats_A, feats_B, physical_data=None):
        """
        五阶段智能推理流程
        
        Args:
            feats_A, feats_B: List[List[Tensor]] - DDPM特征
            physical_data: [B, num_layers, H, W] - 物理数据（DEM, slope等）
        """
        # 编码物理特征（如果可用）
        encoded_physics = None
        if self.use_physics and physical_data is not None:
            encoded_physics = self.physical_encoder(physical_data)
        
        # 解码过程
        lvl_idx = 0
        x = None
        
        for layer_idx, layer in enumerate(self.decoder):
            if isinstance(layer, PhysicsEnhancedBlock):
                # 收集当前尺度的所有时间步特征
                if len(self.time_steps) > 1:
                    list_to_cat_A = [feats_A[t_idx][lvl_idx] for t_idx in range(len(self.time_steps))]
                    list_to_cat_B = [feats_B[t_idx][lvl_idx] for t_idx in range(len(self.time_steps))]
                    f_A_cat = torch.cat(list_to_cat_A, dim=1)
                    f_B_cat = torch.cat(list_to_cat_B, dim=1)
                else:
                    f_A_cat = feats_A[0][lvl_idx]
                    f_B_cat = feats_B[0][lvl_idx]
                
                # ========== 第一阶段：独立的状态理解 ==========
                # 使用物理引导增强特征
                processed_f_A = layer(f_A_cat, encoded_physics)  # F_A_refined
                processed_f_B = layer(f_B_cat, encoded_physics)  # F_B_refined
                
                # ========== 第二阶段：智能的交互式对比 ==========
                if self.use_cross_attention:
                    # 使用交叉注意力进行智能对比
                    diff = self.cross_attention(processed_f_A, processed_f_B, lvl_idx)
                    # diff 现在是 diff_preliminary
                else:
                    # 回退到简单的差异计算
                    diff = torch.abs(processed_f_A - processed_f_B)
                
                # ========== 第三阶段：全局形态学分析（Mamba）==========
                if self.use_mamba:
                    # 使用Mamba捕捉全局空间依赖和形态结构
                    diff = self.mamba_mixer(diff, lvl_idx)
                    # diff 现在是 diff_mamba
                
                # ========== 第四阶段：物理引导的变化聚焦 ==========
                # TODO: 在这里添加第二次物理引导注意力（策略二）
                # diff_focused = self.physics_focus_attention(diff, encoded_physics)
                
                # ========== 第五阶段：条件化专家决策（MoE）==========
                # TODO: 在这里添加 MoE 层
                # diff = self.moe_layer(diff_focused, condition)
                
                # 残差连接
                if x is not None:
                    diff = diff + x
                
                lvl_idx += 1
                
            else:  # AttentionBlock
                diff = layer(diff)
                if layer_idx < len(self.decoder) - 1:
                    x = F.interpolate(diff, scale_factor=2, mode="bilinear", align_corners=False)
                else:
                    x = diff

        # 分类
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))
        
        return cm