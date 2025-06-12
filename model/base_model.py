import os
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    # def set_device(self, x):
    #     if isinstance(x, dict):
    #         for key, item in x.items():
    #             if item is not None:
    #                 x[key] = item.to(self.device, dtype=torch.float)
    #     elif isinstance(x, list):
    #         for item in x:
    #             if item is not None:
    #                 item = item.to(self.device, dtype=torch.float)
    #     else:
    #         x = x.to(self.device, dtype=torch.float)
    #     return x
    def set_device(self, data):
        """
        Moves data to the device specified in self.device.
        Handles dictionaries, lists, and tensors.
        """
        if isinstance(data, dict):
            x = {}
            for key, item in data.items():
                if item is None:
                    x[key] = None
                elif isinstance(item, torch.Tensor):
                    x[key] = item.to(self.device)
                    if item.is_floating_point(): # 只对浮点张量确保是torch.float
                         x[key] = x[key].float()
                    # 对于整数型标签等，保持其原有类型，除非特别需要转换
                    # 例如： if key == 'L' and not x[key].is_floating_point():
                    #           x[key] = x[key].long() # 或者根据损失函数需求
                elif isinstance(item, list):
                    x[key] = [self.set_device_item(sub_item) for sub_item in item]
                else:
                    x[key] = item # 例如 'Index' (如果是单个int) 或 'ID' 中的字符串
            return x
        elif isinstance(data, list):
            return [self.set_device_item(item) for item in data]
        elif isinstance(data, torch.Tensor):
            item_on_device = data.to(self.device)
            if data.is_floating_point():
                item_on_device = item_on_device.float()
            return item_on_device
        else:
            return data # 其他类型直接返回

    def set_device_item(self, item): # 辅助函数
        """Helper function to move individual items (primarily for lists)."""
        if item is None:
            return None
        if isinstance(item, torch.Tensor):
            moved_item = item.to(self.device)
            if item.is_floating_point():
                moved_item = moved_item.float()
            return moved_item
        elif isinstance(item, list): # 处理嵌套列表
            return [self.set_device_item(sub_item) for sub_item in item]
        elif isinstance(item, dict): # 处理嵌套字典 (虽然在列表中不常见)
             return self.set_device(item) # 复用主逻辑
        return item # 其他类型如str, int等直接返回

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
