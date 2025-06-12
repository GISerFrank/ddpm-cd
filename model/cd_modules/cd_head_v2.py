# # Change detection head

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.modules.padding import ReplicationPad2d
# from model.cd_modules.psp import _PSPModule
# from model.cd_modules.se import ChannelSpatialSELayer

# def get_in_channels(feat_scales, inner_channel, channel_multiplier):
#     '''
#     Get the number of input layers to the change detection head.
#     '''
#     in_channels = 0
#     for scale in feat_scales:
#         if scale < 3: #256 x 256
#             in_channels += inner_channel*channel_multiplier[0]
#         elif scale < 6: #128 x 128
#             in_channels += inner_channel*channel_multiplier[1]
#         elif scale < 9: #64 x 64
#             in_channels += inner_channel*channel_multiplier[2]
#         elif scale < 12: #32 x 32
#             in_channels += inner_channel*channel_multiplier[3]
#         elif scale < 15: #16 x 16
#             in_channels += inner_channel*channel_multiplier[4]
#         else:
#             print('Unbounded number for feat_scales. 0<=feat_scales<=14') 
#     return in_channels

# class AttentionBlock(nn.Module):
#     def __init__(self, dim, dim_out):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(dim, dim_out, 3, padding=1),
#             nn.ReLU(),
#             ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
#         )

#     def forward(self, x):
#         return self.block(x)

# class Block(nn.Module):
#     def __init__(self, dim, dim_out, time_steps):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(dim*len(time_steps), dim, 1)
#             if len(time_steps)>1
#             else nn.Identity(),
#             nn.ReLU()
#             if len(time_steps)>1
#             else nn.Identity(),
#             nn.Conv2d(dim, dim_out, 3, padding=1),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return self.block(x)


# class cd_head_v2(nn.Module):
#     '''
#     Change detection head (version 2).
#     '''

#     def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256, time_steps=None):
#         super(cd_head_v2, self).__init__()

#         # Define the parameters of the change detection head
#         feat_scales.sort(reverse=True)
#         self.feat_scales    = feat_scales
#         self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_multiplier)
#         self.img_size       = img_size
#         self.time_steps     = time_steps

#         # Convolutional layers before parsing to difference head
#         self.decoder = nn.ModuleList()
#         for i in range(0, len(self.feat_scales)):
#             dim     = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

#             self.decoder.append(
#                 Block(dim=dim, dim_out=dim, time_steps=time_steps)
#             )

#             if i != len(self.feat_scales)-1:
#                 dim_out =  get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
#                 self.decoder.append(
#                 AttentionBlock(dim=dim, dim_out=dim_out)
#             )

#         # Final classification head
#         clfr_emb_dim = 64
#         self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
#         self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()

#     def forward(self, feats_A, feats_B):
#         # Decoder
#         lvl=0
#         for layer in self.decoder:
#             if isinstance(layer, Block):
#                 f_A = feats_A[0][self.feat_scales[lvl]]
#                 f_B = feats_B[0][self.feat_scales[lvl]]
#                 if len(self.time_steps) > 1:
#                     for i in range(1, len(self.time_steps)):
#                         f_A = torch.cat((f_A, feats_A[i][self.feat_scales[lvl]]), dim=1)
#                         f_B = torch.cat((f_B, feats_B[i][self.feat_scales[lvl]]), dim=1)
    
#                 diff = torch.abs( layer(f_A)  - layer(f_B) )
#                 if lvl!=0:
#                     diff = diff + x
#                 lvl+=1
#             else:
#                 diff = layer(diff)
#                 x = F.interpolate(diff, scale_factor=2, mode="bilinear")

#         # Classifier
#         cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))

#         return cm

# Change detection head

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from model.cd_modules.psp import _PSPModule
from model.cd_modules.se import ChannelSpatialSELayer

def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3: #256 x 256
            in_channels += inner_channel*channel_multiplier[0]
        elif scale < 6: #128 x 128
            in_channels += inner_channel*channel_multiplier[1]
        elif scale < 9: #64 x 64
            in_channels += inner_channel*channel_multiplier[2]
        elif scale < 12: #32 x 32
            in_channels += inner_channel*channel_multiplier[3]
        elif scale < 15: #16 x 16
            in_channels += inner_channel*channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14') 
    return in_channels

class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim*len(time_steps), dim, 1) # 输入通道数现在是 dim * time_steps 数量
            if len(time_steps)>1
            else nn.Identity(),
            nn.ReLU()
            if len(time_steps)>1
            else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x): # x 应该是拼接好多时间步特征后的张量
        return self.block(x)


class cd_head_v2(nn.Module):
    '''
    Change detection head (version 2).
    '''

    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256, time_steps=None):
        super(cd_head_v2, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True) # 例如 [14, 11, 8, 5, 2] (从深到浅)
        self.feat_scales    = feat_scales
        # self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_multiplier) # 这个总通道数可能不是直接用处
        self.img_size       = img_size
        self.time_steps     = time_steps if time_steps is not None else [0] # 确保time_steps不是None

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        # The decoder is built based on the order of self.feat_scales (deepest to shallowest)
        current_decoder_output_channels = 0 # 用于追踪最后一个 AttentionBlock 的输出通道
        for i in range(0, len(self.feat_scales)): # i (即lvl) 从 0 到 N_scales-1
            # dim 是单个时间步下，当前尺度 (self.feat_scales[i]) 的特征通道数
            dim = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)
            
            # Block 会接收拼接了所有时间步特征的输入，所以其第一个卷积的输入通道是 dim * len(self.time_steps)
            # Block 的输出通道数仍然是 dim (单个时间步的通道数)
            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=self.time_steps) 
            )
            current_block_output_channels = dim # Block的输出通道数

            if i != len(self.feat_scales)-1: # 如果不是最后一个Block (即不是最浅层)
                # AttentionBlock的输入通道是当前Block的输出通道
                # AttentionBlock的输出通道是下一个Block期望的输入通道 (单个时间步)
                dim_out_for_attention = get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
                self.decoder.append(
                    AttentionBlock(dim=current_block_output_channels, dim_out=dim_out_for_attention)
                )
                current_decoder_output_channels = dim_out_for_attention
            else: # 如果是最后一个Block
                current_decoder_output_channels = current_block_output_channels


        # Final classification head
        clfr_emb_dim = 64
        # 输入通道数是解码器最后一个模块的输出通道数
        self.clfr_stg1 = nn.Conv2d(current_decoder_output_channels, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, feats_A, feats_B):
        # feats_A 和 feats_B 的结构应该是: List[List[Tensor]]
        # 外层列表: 对应不同的时间步 (长度为 len(self.time_steps))
        # 内层列表: 对应不同的特征尺度 (长度为 len(self.feat_scales))
        #           并且内层列表的特征顺序应与 self.feat_scales 中的尺度顺序一致
        #           (即 feats_A[t_idx][lvl] 对应 self.feat_scales[lvl] 这个尺度)

        # Decoder
        lvl_idx = 0 # 用于索引内层列表 (特征尺度)
        x = None # 用于存储上一个尺度上采样后的特征

        for layer_idx, layer in enumerate(self.decoder):
            if isinstance(layer, Block):
                # feats_A[0][lvl_idx] 是第一个时间步、当前尺度的特征
                # feats_A[t_idx][lvl_idx] 是第 t_idx 时间步、当前尺度的特征
                
                # 获取当前尺度(由lvl_idx决定)的第一个时间步的特征
                current_scale_feat_A = feats_A[0][lvl_idx]
                current_scale_feat_B = feats_B[0][lvl_idx]

                # 如果有多个时间步，则拼接特征
                if len(self.time_steps) > 1:
                    # 收集所有时间步下，当前尺度的特征进行拼接
                    list_to_cat_A = [feats_A[t_idx][lvl_idx] for t_idx in range(len(self.time_steps))]
                    list_to_cat_B = [feats_B[t_idx][lvl_idx] for t_idx in range(len(self.time_steps))]
                    
                    f_A_cat = torch.cat(list_to_cat_A, dim=1)
                    f_B_cat = torch.cat(list_to_cat_B, dim=1)
                else: # 单个时间步
                    f_A_cat = current_scale_feat_A
                    f_B_cat = current_scale_feat_B
    
                # 将拼接后的特征送入Block层
                processed_f_A = layer(f_A_cat)
                processed_f_B = layer(f_B_cat)

                diff = torch.abs(processed_f_A - processed_f_B)
                
                if x is not None: # 如果不是第一个Block (即不是最深层)
                    diff = diff + x # 与上采样后的特征融合
                
                lvl_idx += 1 # 移动到下一个尺度
            else: # 如果是 AttentionBlock
                diff = layer(diff) # AttentionBlock 处理的是上一个Block的输出diff
                if layer_idx < len(self.decoder) -1 : # 如果不是解码器最后一个AttentionBlock
                    x = F.interpolate(diff, scale_factor=2, mode="bilinear", align_corners=False) # align_corners=False 通常推荐
                else: # 如果是最后一个 AttentionBlock（之后直接接分类头）
                    x = diff # 不再上采样，或者这里的 x 应该是解码器最终输出

        # Classifier
        # 确保 x 是解码器最终的输出特征
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))

        return cm

    