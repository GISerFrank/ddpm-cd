import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_CD_model(opt):
    try:
        # 尝试导入新的类名
        from .cd_model import DDPMCDModel as M
        logger.info('Using new DDPMCDModel with physics loss support')
    except ImportError:
        try:
            # 回退到旧的类名
            from .cd_model import CD as M
            logger.info('Using legacy CD model')
        except ImportError:
            raise ImportError("Cannot import CD model. Check cd_model.py")
    
    m = M(opt)
    logger.info('CD Model [{:s}] is created.'.format(m.__class__.__name__))
    return m