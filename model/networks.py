import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
from model.cd_modules.cd_head import cd_head
from model.cd_modules.cd_head_v2 import cd_head_v2, get_in_channels
from thop import profile, clever_format
import copy
import time
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'sr3':
        from .sr3_modules import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32
    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type=model_opt['diffusion']['loss'],    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        print("Distributed training")
        netG = nn.DataParallel(netG)
    return netG

# Change Detection Network
# def define_CD(opt):
#     cd_model_opt = opt['model_cd']
#     diffusion_model_opt = opt['model']
    
#     # Define change detection network head
#     # netCD = cd_head(feat_scales=cd_model_opt['feat_scales'], 
#     #                 out_channels=cd_model_opt['out_channels'], 
#     #                 inner_channel=diffusion_model_opt['unet']['inner_channel'], 
#     #                 channel_multiplier=diffusion_model_opt['unet']['channel_multiplier'],
#     #                 img_size=cd_model_opt['output_cm_size'],
#     #                 psp=cd_model_opt['psp'])
#     netCD = cd_head_v2(feat_scales=cd_model_opt['feat_scales'], 
#                     out_channels=cd_model_opt['out_channels'], 
#                     inner_channel=diffusion_model_opt['unet']['inner_channel'], 
#                     channel_multiplier=diffusion_model_opt['unet']['channel_multiplier'],
#                     img_size=cd_model_opt['output_cm_size'],
#                     time_steps=cd_model_opt["t"])
    
#     # Initialize the change detection head if it is 'train' phase 
#     if opt['phase'] == 'train':
#         # Try different initialization methods
#         # init_weights(netG, init_type='kaiming', scale=0.1)
#         init_weights(netCD, init_type='orthogonal')
#     if opt['gpu_ids'] and opt['distributed']:
#         assert torch.cuda.is_available()
#         netCD = nn.DataParallel(netCD)
    
#     ### Profiling ###
#     f_A, f_B = [], [] 
#     feat_scales = cd_model_opt['feat_scales'].copy()
#     feat_scales.sort(reverse=True)
#     h,w=8,8
#     for i in range(0, len(feat_scales)):
#         dim = get_in_channels([feat_scales[i]], diffusion_model_opt['unet']['inner_channel'], diffusion_model_opt['unet']['channel_multiplier'])
#         A = torch.randn(1,dim,h,w).cuda()
#         B = torch.randn(1,dim,h,w).cuda()
#         f_A.append(A)
#         f_B.append(B)
#         f_A.append(A)
#         f_B.append(B)
#         f_A.append(A)
#         f_B.append(B)
#         h*=2
#         w*=2
#     f_A_r = [ele for ele in reversed(f_A)]
#     f_B_r = [ele for ele in reversed(f_B)]

#     F_A=[]
#     F_B=[]
#     for t_i in range(0, len(cd_model_opt["t"])):
#         print(t_i)
#         F_A.append(f_A_r)
#         F_B.append(f_B_r)
#     flops, params = profile(copy.deepcopy(netCD).cuda(), inputs=(F_A,F_B,), verbose=False)
#     flops, params = clever_format([flops, params])
#     netGcopy = copy.deepcopy(netCD).cuda()
#     netGcopy.eval()
#     with torch.no_grad():
#         start = time.time()
#         _ = netGcopy(F_A, F_B)
#         end = time.time()
#     print('### Model Params: {} FLOPs: {} Time: {}ms ####'.format(params, flops, 1000*(end-start)))
#     del netGcopy, F_A, F_B, f_A_r, f_B_r, f_A, f_B
#     ### --- ###
#     return netCD
# Change Detection Network
def define_CD(opt):
    cd_model_opt = opt['model_cd']
    diffusion_model_opt = opt['model']
    
    # 检查版本
    version = cd_model_opt.get('version', 'v2')
    logger.info(f"Creating CD model version: {version}")
    
    # 1. 根据版本实例化相应的 CD 模型
    if version == 'v8':
        from model.cd_modules.cd_head_v8 import cd_head_v8
        
        # 准备物理配置
        physics_config = cd_model_opt.get('physics_attention', {})
        
        netCD = cd_head_v8(
            feat_scales=cd_model_opt['feat_scales'],
            out_channels=cd_model_opt['out_channels'],
            inner_channel=diffusion_model_opt['unet']['inner_channel'],
            channel_multiplier=diffusion_model_opt['unet']['channel_multiplier'],
            img_size=cd_model_opt['output_cm_size'],
            time_steps=cd_model_opt["t"],
            physics_config=physics_config
        )
        logger.info(f"CD v8 initialized with physics support: {physics_config.get('enabled', False)}")
        
    else:
        # 默认使用 v2
        netCD = cd_head_v2(
            feat_scales=cd_model_opt['feat_scales'], 
            out_channels=cd_model_opt['out_channels'], 
            inner_channel=diffusion_model_opt['unet']['inner_channel'], 
            channel_multiplier=diffusion_model_opt['unet']['channel_multiplier'],
            img_size=cd_model_opt['output_cm_size'],
            time_steps=cd_model_opt["t"]
        )
    
    # 2. 如果是训练阶段，初始化权重
    if opt['phase'] == 'train':
        init_weights(netCD, init_type='orthogonal')

    # 3. Profiling 准备
    if opt.get('gpu_ids') and torch.cuda.is_available():
        profile_device = torch.device(f"cuda:{opt['gpu_ids'][0]}")
    else:
        profile_device = torch.device('cpu')
    logger.info(f"Profiling on device: {profile_device}")

    # 4. 创建用于 profiling 的模型副本
    model_to_profile = copy.deepcopy(netCD)
    model_to_profile.to(profile_device)

    # 5. 创建用于 profiling 的虚拟输入
    F_A_prof, F_B_prof = [], [] 
    feat_scales = cd_model_opt['feat_scales'].copy()
    feat_scales.sort(reverse=True)
    h_curr, w_curr = 8, 8

    # 构建多尺度特征
    single_t_feats_A = []
    single_t_feats_B = []
    h_iter, w_iter = h_curr, w_curr
    
    for i in range(0, len(feat_scales)):
        dim = get_in_channels([feat_scales[i]], diffusion_model_opt['unet']['inner_channel'], 
                            diffusion_model_opt['unet']['channel_multiplier'])
        A_s = torch.randn(1, dim, h_iter, w_iter).to(profile_device)
        B_s = torch.randn(1, dim, h_iter, w_iter).to(profile_device)
        single_t_feats_A.append(A_s)
        single_t_feats_B.append(B_s)
        
        if i < len(feat_scales) - 1:
            h_iter *= 2
            w_iter *= 2
    
    # 为所有时间步复制特征
    for _ in range(0, len(cd_model_opt["t"])):
        F_A_prof.append(single_t_feats_A)
        F_B_prof.append(single_t_feats_B)

    # 6. 为 v8 版本添加物理数据输入
    if version == 'v8' and physics_config.get('enabled', False):
        # 创建虚拟物理数据
        num_physical_layers = physics_config.get('num_physical_layers', 2)
        physical_data = torch.randn(1, num_physical_layers, 256, 256).to(profile_device)
        
        # v8 版本的 profiling
        try:
            flops, params = profile(model_to_profile, inputs=(F_A_prof, F_B_prof, physical_data), verbose=False)
            flops, params = clever_format([flops, params])
        except Exception as e_profile:
            logger.warning(f"Thop profiling failed for v8: {e_profile}. Trying without physical data.")
            try:
                flops, params = profile(model_to_profile, inputs=(F_A_prof, F_B_prof, None), verbose=False)
                flops, params = clever_format([flops, params])
            except Exception as e2:
                logger.warning(f"Thop profiling failed completely: {e2}")
                flops, params = 0, 0
        
        # 推理时间测试
        netGcopy_for_timing = model_to_profile
        netGcopy_for_timing.eval()
        with torch.no_grad():
            start = time.time()
            _ = netGcopy_for_timing(F_A_prof, F_B_prof, physical_data)
            end = time.time()
    else:
        # v2 版本的 profiling
        try:
            flops, params = profile(model_to_profile, inputs=(F_A_prof, F_B_prof,), verbose=False)
            flops, params = clever_format([flops, params])
        except Exception as e_profile:
            logger.warning(f"Thop profiling failed: {e_profile}.")
            flops, params = 0, 0
        
        # 推理时间测试
        netGcopy_for_timing = model_to_profile
        netGcopy_for_timing.eval()
        with torch.no_grad():
            start = time.time()
            _ = netGcopy_for_timing(F_A_prof, F_B_prof)
            end = time.time()
    
    print('### Model Params: {} FLOPs: {} Time: {}ms ####'.format(params, flops, 1000*(end-start)))
    
    # 清理
    del model_to_profile, netGcopy_for_timing, F_A_prof, F_B_prof, single_t_feats_A, single_t_feats_B
    if version == 'v8' and physics_config.get('enabled', False):
        del physical_data

    # 7. 分布式训练处理
    if opt.get('gpu_ids') and opt.get('distributed'):
        assert torch.cuda.is_available()
        logger.info("Wrapping netCD with DataParallel for distributed training.")
        netCD = nn.DataParallel(netCD, opt['gpu_ids'])

    return netCD