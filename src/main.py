import torch
from torch import nn
import numpy as np
import os
# import rarfile as rar
import nibabel as nib
import data
from data.dataset import *
from data.data_loader import get_data_loader
from data import data_loader as loader

from trainer import Trainer
import models
from models.fcnn_model import FCnnModel
from models.loss import CombinedLoss
from models.optimizer import get_optimizer

from utils import logger
import json


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)

    # Construct the path to the parent directory
    parent_dir = os.path.dirname(current_dir)

    # Read and log the current configuration
    cfg = json.load(open(parent_dir + '/config/config.json', 'r'))

    # Data path
    DATA_PATH = cfg['data_path']
    EXPERIMENT = cfg['exp_name']
    BATCH_SIZE = cfg['batch_size']
    LOG_PATH = os.path.join(cfg['exp_path'], 'log', EXPERIMENT + '.log')

    # Setup logger
    logger.create_logger(LOG_PATH)

    # Log the current configuration
    LOGGER = logging.getLogger(__name__)
    LOGGER.info('====Configuration:===')
    for key, param in cfg.items():
        LOGGER.info(f'{key}: {param}')
    LOGGER.info('=====================\n')

    # Initialize CUDA
    torch.cuda.init()

    # Set up the device
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOGGER.info(f'Device: {device}')

    # # Testing the training loop
    # Initializing the model
    model = FCnnModel(cfg)

    # Log the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f'Number of parameters: {total_params}')

    # Initializing the loss & the optimizer
    loss_fn = CombinedLoss()
    optimizer = get_optimizer(model=model,
                              optimizer='SGD')

    # Training
    trainer = Trainer(cfg=cfg,
                      model=model,
                      loss_fn=loss_fn,
                      optim=optimizer,
                      scheduler=None,
                      device=device)
    trainer.train()

