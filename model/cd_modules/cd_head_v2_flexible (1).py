# Change detection head (version 2 - Flexible) - FINAL CORRECT VERSION
# 🔥 关键修正：feats_A[0] 只包含配置的feat_scales，不是完整的0-14索引！

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.cd_modules.se import ChannelSpatialSELayer
    USE_ORIGINAL_SE = True
except ImportError:
    print("⚠️  未找到ChannelSpatialSELayer，使用简化版本")
    USE_ORIGINAL_SE = False


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''Get the number of input layers to the change detection head.'''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:
            in_channels += inner_channel*channel_multiplier[0]
        elif scale < 6:
            in_channels += inner_channel*channel_multiplier[1]
        elif scale < 9:
            in_channels += inner_channel*channel_multiplier[2]
        elif scale < 12:
            in_channels += inner_channel*channel_multiplier[3]
        elif scale < 15:
            in_channels += inner_channel*channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14') 
    return in_channels


def get_resolution_from_scale(scale):
    """根据scale索引获取对应的分辨率"""
    if scale < 3:    return 256
    elif scale < 6:  return 128
    elif scale < 9:  return 64
    elif scale < 12: return 32
    elif scale < 15: return 16
    else: raise ValueError(f"Invalid scale: {scale}")


class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        if USE_ORIGINAL_SE:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim_out, 3, padding=1),
                nn.ReLU(),
                ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim_out, 3, padding=1),
                nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim*len(time_steps), dim, 1) 
            if len(time_steps)>1
            else nn.Identity(),
            nn.ReLU()
            if len(time_steps)>1
            else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class cd_head_v2_flexible(nn.Module):
    '''
    Change detection head (version 2 - Flexible) FINAL CORRECT VERSION
    
    🔥 关键理解：
    - feats_A[0] 的长度 = len(feat_scales)，不是15！
    - feats_A[0][i] 对应 feat_scales[i] 的特征
    - 所以直接用 lvl_idx 索引即可：feats_A[0][lvl_idx] ✓
    '''

    def __init__(self, feat_scales, out_channels=2, inner_channel=None, 
                 channel_multiplier=None, img_size=256, time_steps=None):
        super(cd_head_v2_flexible, self).__init__()

        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.in_channels = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size = img_size
        self.time_steps = time_steps if time_steps is not None else [0]

        # 计算解码器输出分辨率
        self.min_scale = min(feat_scales)
        self.decoder_output_res = get_resolution_from_scale(self.min_scale)
        
        print(f"\n🎯 Flexible CD Head - FINAL CORRECT 配置:")
        print(f"   feat_scales: {self.feat_scales}")
        print(f"   使用原版SE: {USE_ORIGINAL_SE}")
        print(f"   解码器输出分辨率: {self.decoder_output_res}×{self.decoder_output_res}")
        print(f"   目标输出分辨率: {img_size}×{img_size}")

        # Decoder
        self.decoder = nn.ModuleList()
        current_decoder_output_channels = 0
        
        for i in range(len(self.feat_scales)):
            dim = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)
            
            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=self.time_steps)
            )
            current_block_output_channels = dim

            if i != len(self.feat_scales)-1:
                dim_out = get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
                self.decoder.append(
                    AttentionBlock(dim=current_block_output_channels, dim_out=dim_out)
                )
                current_decoder_output_channels = dim_out
            else:
                current_decoder_output_channels = current_block_output_channels

        # 上采样层
        self.upsample_layers = nn.ModuleList()
        current_res = self.decoder_output_res
        
        upsample_factor = img_size // current_res
        num_upsample = 0
        temp_factor = upsample_factor
        while temp_factor > 1:
            temp_factor //= 2
            num_upsample += 1
        
        if num_upsample > 0:
            print(f"   添加{num_upsample}个上采样层 ({current_res}×{current_res} -> {img_size}×{img_size})")
            
            in_channels_upsample = current_decoder_output_channels
            for i in range(num_upsample):
                out_channels_upsample = max(in_channels_upsample // 2, 64)
                
                self.upsample_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels_upsample, out_channels_upsample, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(out_channels_upsample, out_channels_upsample, 3, padding=1),
                    nn.ReLU()
                ))
                in_channels_upsample = out_channels_upsample
            
            final_channels = in_channels_upsample
        else:
            print(f"   无需额外上采样")
            final_channels = current_decoder_output_channels

        print()  # 空行

        # 分类头
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(final_channels, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, feats_A, feats_B):
        # Decoder
        lvl_idx = 0
        x = None

        for layer_idx, layer in enumerate(self.decoder):
            if isinstance(layer, Block):
                # 🔥🔥🔥 关键修正：直接用lvl_idx索引！
                # feats_A[0] 只有 len(feat_scales) 个元素
                # feats_A[0][0] 对应 feat_scales[0] 的特征
                # feats_A[0][1] 对应 feat_scales[1] 的特征
                # ...
                current_scale_feat_A = feats_A[0][lvl_idx]  # ✓ 正确！
                current_scale_feat_B = feats_B[0][lvl_idx]

                # 多时间步拼接
                if len(self.time_steps) > 1:
                    list_to_cat_A = [feats_A[t_idx][lvl_idx] 
                                     for t_idx in range(len(self.time_steps))]
                    list_to_cat_B = [feats_B[t_idx][lvl_idx] 
                                     for t_idx in range(len(self.time_steps))]
                    f_A_cat = torch.cat(list_to_cat_A, dim=1)
                    f_B_cat = torch.cat(list_to_cat_B, dim=1)
                else:
                    f_A_cat = current_scale_feat_A
                    f_B_cat = current_scale_feat_B
    
                processed_f_A = layer(f_A_cat)
                processed_f_B = layer(f_B_cat)
                diff = torch.abs(processed_f_A - processed_f_B)
                
                if x is not None:
                    diff = diff + x
                
                lvl_idx += 1
                
            else:  # AttentionBlock
                diff = layer(diff)
                if layer_idx < len(self.decoder) - 1:
                    x = F.interpolate(diff, scale_factor=2, mode="bilinear", align_corners=False)
                else:
                    x = diff

        # 应用上采样层
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)

        # 分类
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))
        return cm
