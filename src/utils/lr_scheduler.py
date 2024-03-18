import torch


def get_lr_scheduler(cfg: dict,
                     optimizer: torch.optim.Optimizer):
    """
    Returns a learning scheduler, according to the configuration file.
    """
    # Get the scheduler type
    scheduler = cfg['lr_scheduler']
    if scheduler == 'StepLR':
        # Step scheduler:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=cfg['lr_step'],
                                               gamma=cfg['lr_gamma'])
    elif scheduler == 'CosineAnnealing':
        # Cosine annealing schedule:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                    T_0=cfg['lr_restart'],
                                                                    T_mult=cfg['lr_mult'],
                                                                    eta_min=cfg['lr_min'])
    elif scheduler == 'LambdaLR':
        # LambdaLR:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
        return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                 lr_lambda=lambda epoch: max(0.95 ** epoch, cfg['lr_min'] / cfg['lr']))
    else:
        # Constant learning rate
        return None

