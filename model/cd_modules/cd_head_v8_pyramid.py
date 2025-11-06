"""
cd_head_v8_pyramid - 适配物理特征金字塔 + MoE
五阶段智能推理流程，使用多尺度物理特征和条件化专家决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cd_modules.psp import _PSPModule
from model.cd_modules.se import ChannelSpatialSELayer
from model.cd_modules.cross_attention import MultiScaleCrossAttention
from model.cd_modules.mamba_mixer import ChangeDetectionMamba
from model.cd_modules.physics_focus_attention import PhysicsChangeFocusAttention

# 导入物理金字塔模块
from model.cd_modules.physical_encoder_pyramid import PhysicalFeaturePyramid
from model.cd_modules.physics_attention_pyramid import PhysicsGuidedAttention

# 🆕 导入 MoE 模块（第五阶段）
# 条件映射: 0=rainfall, 1=seismic, 2=snowmelt, 3=flood, 4=compound
from model.cd_modules.conditional_moe import MultiScaleMoE, create_condition_embedding


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''Get the number of input channels for each scale'''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:
            in_channels += inner_channel * channel_multiplier[0]
        elif scale < 6:
            in_channels += inner_channel * channel_multiplier[1]
        elif scale < 9:
            in_channels += inner_channel * channel_multiplier[2]
        elif scale < 12:
            in_channels += inner_channel * channel_multiplier[3]
        elif scale < 15:
            in_channels += inner_channel * channel_multiplier[4]
        else:
            print(f'Unbounded number for feat_scales: {scale}')
    return in_channels


class PhysicsEnhancedBlock(nn.Module):
    """
    物理增强Block - DDPM-aware版本
    集成物理引导注意力（策略一）
    """
    def __init__(self, dim, dim_out, time_steps, physics_attention=None, num_groups=8):
        super().__init__()
        
        # 时间步融合 - DDPM-aware
        self.time_fusion = nn.Sequential(
            nn.Conv2d(dim * len(time_steps), dim, 1) if len(time_steps) > 1 else nn.Identity(),
            nn.GroupNorm(num_groups, dim) if len(time_steps) > 1 else nn.Identity(),
            nn.SiLU() if len(time_steps) > 1 else nn.Identity(),
        )
        
        # 物理引导注意力（策略一）
        self.physics_attention = physics_attention
        
        # 特征处理 - DDPM-aware
        self.feature_conv = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(num_groups, dim_out),
            nn.SiLU()
        )
        
    def forward(self, x, physical_features=None):
        """
        Args:
            x: 拼接的时间步特征
            physical_features: 尺度对齐的物理特征（来自金字塔）
        """
        # 时间步融合
        x = self.time_fusion(x)
        
        # 应用物理引导注意力
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


class cd_head_v8_pyramid(nn.Module):
    '''
    Change detection head v8 - 物理金字塔 + MoE 版本
    
    五阶段智能推理：
    第一阶段：物理引导的状态理解 (尺度特定物理特征) ✓
    第二阶段：智能的交互式对比（交叉注意力）✓
    第三阶段：全局形态学分析（Mamba）✓
    第四阶段：物理引导的变化聚焦 (尺度特定物理特征) ✓
    第五阶段：条件化专家决策（MoE）✓ 🆕
    '''

    def __init__(self, opt, physics_attention_config=None, cross_attention_config=None, 
                 mamba_config=None, physics_focus_config=None, moe_config=None, num_groups=8):
        """
        Args:
            opt: 完整配置字典
            physics_attention_config: 物理注意力模块配置
            cross_attention_config: 交叉注意力配置
            mamba_config: Mamba配置
            physics_focus_config: 物理聚焦配置
            moe_config: MoE配置 🆕
            num_groups: GroupNorm的组数
        """
        super(cd_head_v8_pyramid, self).__init__()

        # 从opt读取基础参数
        feat_scales = opt['model_cd']['feat_scales']
        out_channels = opt['model_cd']['out_channels']
        inner_channel = opt['model']['unet']['inner_channel']
        channel_multiplier = opt['model']['unet']['channel_multiplier']
        img_size = opt['model_cd']['output_cm_size']
        time_steps = opt['model_cd'].get('t', [0])
        
        # 基础参数设置
        feat_scales_sorted = sorted(feat_scales, reverse=True)
        self.feat_scales = feat_scales_sorted
        self.img_size = img_size
        self.time_steps = time_steps
        self.num_groups = num_groups
        
        # 物理配置
        self.physics_attention_config = physics_attention_config or {}
        self.use_physics = self.physics_attention_config.get('enabled', False)
        
        # 交叉注意力配置
        self.cross_attention_config = cross_attention_config or {}
        self.use_cross_attention = self.cross_attention_config.get('enabled', False)
        
        # Mamba配置
        self.mamba_config = mamba_config or {}
        self.use_mamba = self.mamba_config.get('enabled', False)
        
        # 物理聚焦配置
        self.physics_focus_config = physics_focus_config or {}
        self.use_physics_focus = self.physics_focus_config.get('enabled', False) and self.use_physics
        
        # 🆕 MoE配置
        self.moe_config = moe_config or {}
        self.use_moe = self.moe_config.get('enabled', False)
        
        # 物理特征金字塔编码器
        if self.use_physics:
            num_physical_layers = self.physics_attention_config.get('num_physical_layers', 2)
            self.physical_pyramid = PhysicalFeaturePyramid(
                opt=opt,
                num_physical_layers=num_physical_layers,
                num_groups=num_groups
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
        
        # 🆕 条件化专家决策模块（第五阶段）
        if self.use_moe:
            self.moe_layer = MultiScaleMoE(
                scale_dims=self.scale_dims,
                num_experts=self.moe_config.get('num_experts', 4),
                num_conditions=self.moe_config.get('num_conditions', 5),
                scale_specific_experts=self.moe_config.get('scale_specific_experts', None),
                temperature=self.moe_config.get('temperature', 1.0),
                use_load_balancing=self.moe_config.get('use_load_balancing', True),
                dropout=self.moe_config.get('dropout', 0.1)
            )
        
        # 构建解码器
        self.decoder = nn.ModuleList()
        current_decoder_output_channels = 0
        
        for i in range(len(self.feat_scales)):
            scale = self.feat_scales[i]
            dim = self.scale_dims[i]
            
            # 为每个尺度创建物理引导注意力模块（第一阶段）
            physics_attention = None
            if self.use_physics:
                physics_attention = PhysicsGuidedAttention(
                    visual_channels=dim,
                    physical_channels=dim,  # 物理特征通道数与视觉特征对齐
                    hidden_dim=64,
                    dropout=self.physics_attention_config.get('dropout', 0.1),
                    num_groups=num_groups
                )
            
            # 使用增强的Block
            self.decoder.append(
                PhysicsEnhancedBlock(
                    dim=dim,
                    dim_out=dim,
                    time_steps=self.time_steps,
                    physics_attention=physics_attention,
                    num_groups=num_groups
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

        # 物理聚焦模块（第四阶段）
        if self.use_physics_focus:
            self.physics_focus_modules = nn.ModuleDict()
            for i, scale in enumerate(self.feat_scales):
                dim = self.scale_dims[i]
                self.physics_focus_modules[str(scale)] = PhysicsChangeFocusAttention(
                    change_channels=dim,
                    physical_channels=dim,  # 与视觉特征对齐
                    hidden_dim=self.physics_focus_config.get('hidden_dim', 128),
                    dropout=self.physics_focus_config.get('dropout', 0.1),
                    num_groups=num_groups
                )
        
        # 最终分类头
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(current_decoder_output_channels, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # 注意力图收集器
        self.attention_maps = {}
        
        # 🆕 MoE 辅助损失（用于训练）
        self.moe_aux_loss = None

    def forward(self, feats_A, feats_B, physical_data=None, condition_id=None):
        """
        五阶段智能推理流程
        
        Args:
            feats_A, feats_B: List[List[Tensor]] - DDPM特征
            physical_data: [B, num_layers, 256, 256] - 原始物理数据
            condition_id: [B] tensor 或 str - 滑坡诱因条件 🆕
                可选值: 'rainfall', 'seismic', 'snowmelt', 'flood', 'compound'
                或者直接传入 tensor [B] 每个元素是 0-4 的整数
        """
        batch_size = feats_A[0][0].shape[0]
        
        # 生成物理特征金字塔
        pyramid_physics = None
        if self.use_physics and physical_data is not None:
            pyramid_physics = self.physical_pyramid(physical_data)
            # pyramid_physics: Dict[scale -> Tensor]
        
        # 🆕 处理条件ID（用于MoE）
        if self.use_moe:
            if condition_id is None:
                # 默认使用 'compound' 类型
                condition_id = create_condition_embedding(
                    batch_size, 'compound', feats_A[0][0].device
                )
            elif isinstance(condition_id, str):
                # 字符串转 tensor
                condition_id = create_condition_embedding(
                    batch_size, condition_id, feats_A[0][0].device
                )
        
        # 清空注意力图和辅助损失
        self.attention_maps = {}
        self.moe_aux_loss = None
        
        # 解码过程
        lvl_idx = 0
        x = None
        
        for layer_idx, layer in enumerate(self.decoder):
            if isinstance(layer, PhysicsEnhancedBlock):
                current_scale = self.feat_scales[lvl_idx]
                
                # 收集当前尺度的所有时间步特征
                if len(self.time_steps) > 1:
                    list_to_cat_A = [feats_A[t_idx][lvl_idx] for t_idx in range(len(self.time_steps))]
                    list_to_cat_B = [feats_B[t_idx][lvl_idx] for t_idx in range(len(self.time_steps))]
                    f_A_cat = torch.cat(list_to_cat_A, dim=1)
                    f_B_cat = torch.cat(list_to_cat_B, dim=1)
                else:
                    f_A_cat = feats_A[0][lvl_idx]
                    f_B_cat = feats_B[0][lvl_idx]
                
                # 获取当前尺度的物理特征
                physics_for_scale = pyramid_physics[current_scale] if pyramid_physics else None
                
                # ========== 第一阶段：独立的状态理解 ==========
                processed_f_A = layer(f_A_cat, physics_for_scale)
                processed_f_B = layer(f_B_cat, physics_for_scale)
                
                # ========== 第二阶段：智能的交互式对比 ==========
                if self.use_cross_attention:
                    diff = self.cross_attention(processed_f_A, processed_f_B, lvl_idx)
                else:
                    diff = torch.abs(processed_f_A - processed_f_B)
                
                # ========== 第三阶段：全局形态学分析（Mamba）==========
                if self.use_mamba:
                    diff = self.mamba_mixer(diff, lvl_idx)
                
                # ========== 第四阶段：物理引导的变化聚焦 ==========
                if self.use_physics_focus and physics_for_scale is not None:
                    focus_module = self.physics_focus_modules[str(current_scale)]
                    diff, focus_attention = focus_module(diff, physics_for_scale)
                    self.attention_maps[f'scale_{current_scale}'] = focus_attention
                
                # ========== 🆕 第五阶段：条件化专家决策（MoE）==========
                if self.use_moe:
                    # 应用MoE层
                    diff, moe_aux_loss = self.moe_layer(diff, condition_id, lvl_idx)
                    
                    # 累积辅助损失（用于训练时的负载均衡）
                    if self.training:
                        if self.moe_aux_loss is None:
                            self.moe_aux_loss = moe_aux_loss
                        else:
                            self.moe_aux_loss += moe_aux_loss
                
                # 与上一层融合
                if x is not None:
                    if x.shape[2:] != diff.shape[2:]:
                        x = F.interpolate(x, size=diff.shape[2:], mode='bilinear', align_corners=False)
                    x = x + diff
                else:
                    x = diff
                
                lvl_idx += 1
                
            elif isinstance(layer, AttentionBlock):
                x = layer(x)
        
        # 最终分类
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x = self.clfr_stg1(x)
        x = self.relu(x)
        pred = self.clfr_stg2(x)
        
        return pred
    
    def get_attention_maps(self):
        """获取保存的注意力图（用于可视化分析）"""
        return self.attention_maps
    
    def get_moe_aux_loss(self):
        """🆕 获取MoE辅助损失（用于训练）"""
        return self.moe_aux_loss if self.moe_aux_loss is not None else 0.0


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("🧪 测试 cd_head_v8_pyramid with MoE")
    
    # 模拟config
    test_opt = {
        'model_cd': {
            'feat_scales': [14, 11, 8, 5],
            'out_channels': 2,
            'output_cm_size': 256,
            't': [50, 100, 400]
        },
        'model': {
            'unet': {
                'inner_channel': 128,
                'channel_multiplier': [1, 2, 4, 8, 8]
            }
        }
    }
    
    # 物理配置
    physics_attention_config = {
        'enabled': True,
        'num_physical_layers': 2,
        'dropout': 0.1
    }
    
    # 🆕 MoE配置
    moe_config = {
        'enabled': True,
        'num_experts': 5,  # 🔄 改为5个专家
        'num_conditions': 5,  # rainfall, seismic, snowmelt, flood, compound
        'temperature': 1.0,
        'use_load_balancing': True,
        'dropout': 0.1,
        'scale_specific_experts': None
    }
    
    # 创建模型
    model = cd_head_v8_pyramid(
        opt=test_opt,
        physics_attention_config=physics_attention_config,
        moe_config=moe_config,
        num_groups=8
    )
    
    # 测试输入
    batch_size = 2
    # 模拟DDPM特征: List[List[Tensor]]
    # 3个时间步 × 4个尺度
    feats_A = [
        [torch.randn(batch_size, 1024, 16, 16),   # scale 14
         torch.randn(batch_size, 1024, 32, 32),   # scale 11
         torch.randn(batch_size, 512, 64, 64),    # scale 8
         torch.randn(batch_size, 256, 128, 128)]  # scale 5
        for _ in range(3)
    ]
    feats_B = [
        [torch.randn(batch_size, 1024, 16, 16),
         torch.randn(batch_size, 1024, 32, 32),
         torch.randn(batch_size, 512, 64, 64),
         torch.randn(batch_size, 256, 128, 128)]
        for _ in range(3)
    ]
    physical_data = torch.randn(batch_size, 2, 256, 256)
    
    # 🆕 测试不同的条件
    print("\n" + "="*60)
    print("测试1: 使用默认条件 (compound)")
    pred1 = model(feats_A, feats_B, physical_data)
    print(f"✅ 输出预测: {pred1.shape}")
    
    print("\n测试2: 使用 'rainfall' 条件")
    pred2 = model(feats_A, feats_B, physical_data, condition_id='rainfall')
    print(f"✅ 输出预测: {pred2.shape}")
    
    print("\n测试3: 使用 tensor 条件")
    condition_tensor = torch.tensor([0, 1])  # rainfall, seismic
    pred3 = model(feats_A, feats_B, physical_data, condition_id=condition_tensor)
    print(f"✅ 输出预测: {pred3.shape}")
    
    print("\n" + "="*60)
    print(f"✅ 注意力图数量: {len(model.attention_maps)}")
    for scale, attn in model.attention_maps.items():
        print(f"  {scale}: {attn.shape}")
    
    print(f"\n🆕 MoE辅助损失: {model.get_moe_aux_loss()}")
    
    print(f"\n🎉 所有测试通过！")
