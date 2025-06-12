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
    
    # 1. 实例化 netCD (此时它可能在CPU上)
    netCD = cd_head_v2(feat_scales=cd_model_opt['feat_scales'], 
                    out_channels=cd_model_opt['out_channels'], 
                    inner_channel=diffusion_model_opt['unet']['inner_channel'], 
                    channel_multiplier=diffusion_model_opt['unet']['channel_multiplier'],
                    img_size=cd_model_opt['output_cm_size'],
                    time_steps=cd_model_opt["t"])
    
    # 2. 如果是训练阶段，初始化权重 (通常在CPU上操作)
    if opt['phase'] == 'train':
        init_weights(netCD, init_type='orthogonal')

    # --- 开始为 profiling 准备设备和模型 ---
    # 3. 确定用于 profiling 的设备 (通常是第一个GPU或CPU)
    if opt.get('gpu_ids') and torch.cuda.is_available(): # 确保 gpu_ids 存在且CUDA可用
        profile_device = torch.device(f"cuda:{opt['gpu_ids'][0]}")
    else:
        profile_device = torch.device('cpu')
    logger.info(f"Profiling on device: {profile_device}")

    # 4. 创建 netCD 的一个深拷贝，并将其显式移动到 profile_device
    # 这是将要被 thop.profile 分析的模型实例
    model_to_profile = copy.deepcopy(netCD)
    model_to_profile.to(profile_device)
    # --- 结束为 profiling 准备模型 ---

    # 注意：原始的 netCD 实例的设备将由调用 define_CD 之后的 set_device 方法处理。
    #这里的 model_to_profile 仅用于当前的 profiling。

    # 5. 创建用于 profiling 的虚拟输入 F_A, F_B，并确保它们在 profile_device 上
    # (原始代码中 .cuda() 会将它们放到默认GPU，现在改为 .to(profile_device))
    F_A_prof, F_B_prof = [], [] 
    feat_scales = cd_model_opt['feat_scales'].copy()
    feat_scales.sort(reverse=True) # 深层特征在前
    h_curr, w_curr = 8, 8 # 假设最深层特征图的起始尺寸

    # 这个循环是为了构建一个符合 cd_head_v2.forward 输入结构的虚拟特征列表
    # feats_A 和 feats_B 应该是列表的列表: List[List[Tensor]]
    # 外层列表对应时间步 t (来自 opt['model_cd']['t'])
    # 内层列表对应不同的特征尺度 (来自 opt['model_cd']['feat_scales'])
    
    # 为一个时间步构建多尺度特征
    single_t_feats_A = []
    single_t_feats_B = []
    h_iter, w_iter = h_curr, w_curr
    for i in range(0, len(feat_scales)):
        # 计算该尺度特征的通道数
        dim = get_in_channels([feat_scales[i]], diffusion_model_opt['unet']['inner_channel'], diffusion_model_opt['unet']['channel_multiplier'])
        # 创建虚拟张量并移动到profile_device
        A_s = torch.randn(1, dim, h_iter, w_iter).to(profile_device)
        B_s = torch.randn(1, dim, h_iter, w_iter).to(profile_device)
        single_t_feats_A.append(A_s)
        single_t_feats_B.append(B_s)
        # 为下一个（更浅层）尺度更新h, w
        if i < len(feat_scales) - 1: # 避免在最后一个尺度后也乘以2
            h_iter *= 2
            w_iter *= 2
            
    # 为所有时间步复制这份多尺度特征列表
    # （原始代码中F_A.append(f_A_r)暗示了每个时间步的虚拟特征可以相同）
    for _ in range(0, len(cd_model_opt["t"])):
        F_A_prof.append(single_t_feats_A) # 使用深拷贝以防万一
        F_B_prof.append(single_t_feats_B)

    # 6. 使用已在正确设备上的 model_to_profile 和 F_A_prof, F_B_prof 进行分析
    try:
        flops, params = profile(model_to_profile, inputs=(F_A_prof, F_B_prof,), verbose=False)
        flops, params = clever_format([flops, params])
    except Exception as e_profile:
        logger.warning(f"Thop profiling failed: {e_profile}. Check model and dummy input device/shapes. Setting flops/params to 0.")
        flops, params = 0, 0
    
    # 7. 第二次分析（推理时间），同样确保模型在正确设备上
    # 可以重用 model_to_profile，因为它只是被profile过，权重没变
    netGcopy_for_timing = model_to_profile # 假设profile不改变模型状态
    netGcopy_for_timing.eval()
    with torch.no_grad():
        start = time.time()
        _ = netGcopy_for_timing(F_A_prof, F_B_prof) # 使用相同的虚拟输入
        end = time.time()
    print('### Model Params: {} FLOPs: {} Time: {}ms ####'.format(params, flops, 1000*(end-start)))
    
    del model_to_profile, netGcopy_for_timing, F_A_prof, F_B_prof, single_t_feats_A, single_t_feats_B 
    ### --- ###

    # 8. （可选，但原始代码中有）如果需要分布式训练，在这里处理原始的 netCD
    # 注意：如果进行了这个DataParallel包装，那么返回的netCD将是DataParallel对象
    # set_device 方法也需要能正确处理这种情况
    if opt.get('gpu_ids') and opt.get('distributed'): # 检查gpu_ids和distributed标志
        assert torch.cuda.is_available()
        logger.info("Wrapping netCD with DataParallel for distributed training.")
        netCD = nn.DataParallel(netCD, opt['gpu_ids'])
        # 注意：DataParallel 会自动将模型数据复制到指定的多个GPU上
        # 但主模型仍在 opt['gpu_ids'][0] 上。
        # 如果不进行分布式训练，netCD 仍保持原样（可能在CPU或被后续set_device移动）

    return netCD