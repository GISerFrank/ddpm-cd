import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalFeatureEncoder(nn.Module):
    """
    物理特征编码器
    将原始物理数据（DEM、坡度等）编码为深度特征
    """
    
    def __init__(self, input_channels=2, output_channels=64, 
                 layer_configs=None):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # 默认层配置
        if layer_configs is None:
            layer_configs = {
                'dem': {'type': 'continuous', 'channels': 1},
                'slope': {'type': 'continuous', 'channels': 1},
                'aspect': {'type': 'circular', 'channels': 2},  # sin/cos编码
                'geology': {'type': 'categorical', 'channels': 10}  # one-hot
            }
        
        self.layer_configs = layer_configs
        
        # 特征特定的编码器
        self.encoders = nn.ModuleDict()
        
        # DEM编码器（连续值）
        self.encoders['dem'] = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 坡度编码器（连续值，但有物理意义）
        self.encoders['slope'] = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 融合层
        total_encoded_channels = len(self.encoders) * 16
        self.fusion = nn.Sequential(
            nn.Conv2d(total_encoded_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, physical_data):
        """
        Args:
            physical_data: [B, num_layers, H, W] - 物理数据
            
        Returns:
            encoded_features: [B, output_channels, H, W] - 编码后的特征
        """
        B, num_layers, H, W = physical_data.shape
        
        encoded_features = []
        
        # 分别编码每种物理数据
        if num_layers >= 1:  # DEM
            dem = physical_data[:, 0:1, :, :]
            dem_encoded = self.encoders['dem'](dem)
            encoded_features.append(dem_encoded)
        
        if num_layers >= 2:  # Slope
            slope = physical_data[:, 1:2, :, :]
            slope_encoded = self.encoders['slope'](slope)
            encoded_features.append(slope_encoded)
        
        # 拼接所有编码特征
        if encoded_features:
            combined = torch.cat(encoded_features, dim=1)
            output = self.fusion(combined)
        else:
            # 如果没有物理数据，返回零特征
            output = torch.zeros(B, self.output_channels, H, W, 
                               device=physical_data.device)
        
        return output
    
    def get_slope_attention_prior(self, slope_data):
        """
        基于坡度生成先验注意力
        滑坡更容易发生在特定坡度范围内（如20-45度）
        """
        # 假设slope_data已经归一化到[0,1]，对应0-90度
        optimal_slope_min = 20.0 / 90.0  # 约0.22
        optimal_slope_max = 45.0 / 90.0  # 约0.50
        
        # 创建软掩膜
        attention = torch.sigmoid(
            10 * (slope_data - optimal_slope_min)
        ) * torch.sigmoid(
            10 * (optimal_slope_max - slope_data)
        )
        
        return attention