from data.dataset import *
from datetime import datetime

from data.data_loader import get_data_loaders

from models.loss import get_loss_fn

from trainer import Trainer
from src.models.fcnn_model import FCnnModel
from src.models.loss import *
from src.models.optimizer import get_optimizer

from utils import logger
from utils.stats_manager import StatsManager

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
    BASE_PATH = cfg['base_path']
    PLANE = cfg['plane']
    DATA_PATH = BASE_PATH + cfg['data_path']
    EXPERIMENT = cfg['exp_name']
    BATCH_SIZE = cfg['batch_size']

    # Set an experiment timestamp if name is missing
    if EXPERIMENT is None:
        EXPERIMENT = datetime.now().strftime("%m-%d-%y_%H-%M")

    # Setup the logger
    LOG_PATH = os.path.join(BASE_PATH, cfg['log_path'].format(PLANE), EXPERIMENT + '.log')
    logger.create_logger(LOG_PATH)

    # Set up the stats manager
    SUMMARY_PATH = os.path.join(BASE_PATH, cfg['summary_path'].format(PLANE), EXPERIMENT)
    stats_manager = StatsManager(cfg['num_classes'], SUMMARY_PATH)

    # Set up the checkpoint manager
    CHECKPOINT_PATH = os.path.join(BASE_PATH, cfg['checkpoint_path'].format(PLANE), EXPERIMENT)
    cfg['checkpoint_path'] = CHECKPOINT_PATH

    # Log the current configuration
    LOGGER = logging.getLogger(__name__)
    LOGGER.info('====Configuration:===')
    for key, param in cfg.items():
        LOGGER.info(f'{key}: {param}')
    LOGGER.info('=====================\n')

    # Initialize CUDA
    # torch.cuda.init()

    # Set up the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    LOGGER.info(f'Device: {device}')

    # Initializing the model
    model = FCnnModel(cfg)

    # Log the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f'Number of parameters: {total_params}')

    # Create the data loaders
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    # Initializing the loss & the optimizer
    loss_fn = get_loss_fn(cfg['loss_function'])

    # Initialize the optimizer
    optimizer_type = cfg['optimizer']
    optimizer = get_optimizer(model=model,
                              optimizer=optimizer_type,
                              learning_rate=cfg['lr'])

    # Initialize the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   step_size=cfg['lr_step'],
                                                   gamma=cfg['lr_gamma'])
    # lr_scheduler = None

    # Initialize the trainer
    trainer = Trainer(cfg=cfg,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      model=model,
                      loss_fn=loss_fn,
                      optim=optimizer,
                      lr_sch=lr_scheduler,
                      stats_manager=stats_manager,
                      device=device)

    # Train
    trainer.train()

    # Test
    trainer.test()