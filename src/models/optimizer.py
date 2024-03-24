import torch
from torch import optim


def get_optimizer(model,
                  optimizer: str,
                  learning_rate: float):
    """
    Returns an instance of torch.optim.optimizer.Optimizer type
    See - https://pytorch.org/docs/stable/optim.html
    """
    if optimizer == 'SGD':
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        return optim.SGD(model.parameters(),
                         lr=learning_rate,
                         momentum=0.9,
                         dampening=0,
                         weight_decay=1e-4,
                         nesterov=True)
    elif optimizer == 'ADAM':
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        return optim.Adam(model.parameters(),
                          lr=learning_rate,
                          betas=(0.9, 0.999),
                          eps=1e-08,
                          weight_decay=1e-4)
    elif optimizer == 'ADAM_W':
        # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
        return optim.AdamW(model.parameters(),
                           lr=learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-8,
                           weight_decay=1e-4)
