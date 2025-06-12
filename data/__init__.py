# '''create dataset and dataloader'''
# import logging
# from re import split
# import torch.utils.data


# def create_dataloader(dataset, dataset_opt, phase):
#     '''create dataloader '''
#     if phase == 'train':
#         return torch.utils.data.DataLoader(
#             dataset,
#             batch_size=dataset_opt['batch_size'],
#             shuffle=dataset_opt['use_shuffle'],
#             num_workers=dataset_opt['num_workers'],
#             pin_memory=True)
#     elif phase == 'val':
#         return torch.utils.data.DataLoader(
#             dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
#     else:
#         raise NotImplementedError(
#             'Dataloader [{:s}] is not found.'.format(phase))

# #Create CD dataloader
# def create_cd_dataloader(dataset, dataset_opt, phase):
#     '''create dataloader '''
#     if phase == 'train' or 'val' or 'test':
#         return torch.utils.data.DataLoader(
#             dataset,
#             batch_size=dataset_opt['batch_size'],
#             shuffle=dataset_opt['use_shuffle'],
#             num_workers=dataset_opt['num_workers'],
#             pin_memory=True)
#     else:
#         raise NotImplementedError(
#             'Dataloader [{:s}] is not found.'.format(phase))

# # Create image dataset
# def create_image_dataset(dataset_opt, phase):
#     '''create dataset'''
#     mode = dataset_opt['mode']
#     from data.ImageDataset import ImageDataset as D
#     dataset = D(dataroot=dataset_opt['dataroot'],
#                 resolution=dataset_opt['resolution'],
#                 split=phase,
#                 data_len=dataset_opt['data_len']
#                 )
#     logger = logging.getLogger('base')
#     logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
#                                                            dataset_opt['name']))
#     return dataset

# # Create change-detection dataset
# def create_cd_dataset(dataset_opt, phase):
#     '''create dataset'''
#     mode = dataset_opt['mode']
#     from data.CDDataset import CDDataset as D
#     dataset = D(dataroot=dataset_opt['dataroot'],
#                 resolution=dataset_opt['resolution'],
#                 split=phase,
#                 data_len=dataset_opt['data_len']
#                 )
#     logger = logging.getLogger('base')
#     logger.info('Dataset [{:s} - {:s} - {:s}] is created.'.format(dataset.__class__.__name__,
#                                                            dataset_opt['name'],
#                                                            phase))
#     return dataset
'''create dataset and dataloader'''
import logging
# from re import split # 未使用，可以移除
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        # 原始代码中验证集的 batch_size 固定为1，shuffle为False，num_workers为1
        # 如果 ddpm_cd.py 始终使用 create_cd_dataloader 来创建 val_loader，则此处的硬编码不会生效
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        # 如果测试也可能调用这个函数，需要添加 test 的处理逻辑
        # 或者确保测试总是调用 create_cd_dataloader
        raise NotImplementedError(
            'Dataloader (vanilla) [{:s}] is not found or not supported for this phase.'.format(phase))

#Create CD dataloader
def create_cd_dataloader(dataset, dataset_opt, phase):
    '''create dataloader for Change Detection, typically uses dataset_opt for all phases'''
    if phase in ['train', 'val', 'test']: # 修正了逻辑判断
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    else:
        raise NotImplementedError(
            'CD Dataloader [{:s}] is not found.'.format(phase))

# Create image dataset
def create_image_dataset(dataset_opt, phase):
    '''create image dataset (e.g., for DDPM pretraining)'''
    # mode = dataset_opt.get('mode', None) # 使用 get 避免 KeyError
    from data.ImageDataset import ImageDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                resolution=dataset_opt['resolution'],
                split=phase,
                data_len=dataset_opt.get('data_len', -1) # 使用 get
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

# Create change-detection dataset
def create_cd_dataset(dataset_opt, phase):
    '''create change detection dataset'''
    # mode = dataset_opt.get('mode', None) # 使用 get 避免 KeyError
    # 确保这是您修改后的CDDataset.py中定义的类，并且文件名已更改
    from data.CDDataset_GVLM_CD import CDDataset as D 
    dataset = D(dataroot=dataset_opt['dataroot'],
                resolution=dataset_opt['resolution'],
                split=phase,
                data_len=dataset_opt.get('data_len', -1) # 使用 get
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name'],
                                                           phase))
    return dataset
