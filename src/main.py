from data.dataset import *
from datetime import datetime

from data.data_loader import get_data_loaders
from data.data_loader import get_data_loader

from trainer import Trainer
from src.models.fcnn_model import FCnnModel
from src.models.loss import *
from src.models.optimizer import get_optimizer

from utils import logger
from utils.lr_scheduler import get_lr_scheduler
from utils.stats_manager import StatsManager

import json

from sklearn.model_selection import KFold


def get_splits(val_size, subjects, shuffle=True):
    n = np.floor(1 / val_size)
    kf = KFold(n_splits=n, shuffle=shuffle, random_state=5)
    splits = kf.split(subjects)

    splits_list = []
    for j, split in enumerate(splits):
        train_subjects = [os.path.basename(subject_paths[i]) for j in split[0]]
        val_subjects = [os.path.basename(subject_paths[i]) for j in split[1]]

        # Append the tuple of train and val subjects to subject_splits
        splits_list.append((train_subjects, val_subjects))
    return splits_list


def get_subjects_paths_splits(is_cross, data_path, val_size, hdf_file_path=None):
    splits = []
    if hdf_path is not None:
        hdf5_file = os.path.join(data_path, hdf_file_path)
        with h5py.File(hdf5_file, "r") as hf:
            subjects = hf['subjects'][:]

            # Retrieve the splits if performing cross-validation
            if is_cross:
                if 'splits' in hf:
                    for s in hf['splits'].values():
                        splits.append((s['train'][:], s['val'][:]))
                else:
                    train_split, val_split, test = du.get_train_test_split(subjects,
                                                                           1 - val_size,
                                                                           val_size)
                    splits = [(train_split, val_split)]
            else:
                splits = get_splits(val_size, subjects, shuffle=True)
    else:
        subjects = [os.path.join(data_path, s) for s in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, s))]

        if is_cross:
            splits = get_splits(val_size, subject_paths, shuffle=True)
        else:
            train_split, val_split, test = du.get_train_test_split(subject_paths,
                                                                   1 - val_size,
                                                                   val_size)
            splits = [(train_split, val_split)]
    return subjects, splits


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)

    # Construct the path to the parent directory
    parent_dir = os.path.dirname(current_dir)

    # Read and log the current configuration
    cfg = json.load(open(parent_dir + '/config/config.json', 'r'))

    # Log the current configuration
    LOGGER = logging.getLogger(__name__)
    LOGGER.info('====Configuration:===')
    for key, param in cfg.items():
        LOGGER.info(f'{key}: {param}')
    LOGGER.info('=====================\n')

    # Data path
    BASE_PATH = cfg['base_path']
    PLANE = cfg['plane']
    DATA_PATH = BASE_PATH + cfg['data_path']
    EXPERIMENT = cfg['exp_name']
    BATCH_SIZE = cfg['batch_size']

    # Set an experiment timestamp if name is missing
    if EXPERIMENT is None:
        EXPERIMENT = datetime.now().strftime("%m-%d-%y_%H-%M")
        cfg['exp_name'] = EXPERIMENT

    # Initialize CUDA
    # torch.cuda.init()

    # Set up the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # Retrive the subjects paths and splits
    cross_val = cfg['cross_validation']
    hdf_path = cfg['hdf5_file'] if cfg['hdf5_dataset'] else None
    subject_paths, subject_splits = get_subjects_paths_splits(cross_val, DATA_PATH, cfg['val_size'], hdf_path)

    # Iterate through the splits
    for i, (train, val) in enumerate(subject_splits):
        # Convert from indexes to paths if performing cross-validation
        if cross_val:
            train = [t.decode('utf-8') for t in train]
            val = [v.decode('utf-8') for v in val]
            EXPERIMENT = datetime.now().strftime("%m-%d-%y_%H-%M")
            cfg['exp_name'] = EXPERIMENT

            # Log the current split
            LOGGER.info('\n=====================')
            LOGGER.info(f'Training on split: {i}')

        # Create the data loaders
        train_loader = get_data_loader(cfg, train, 'train')
        val_loader = get_data_loader(cfg, val, 'val')
        # if test is not None:
        #     test_loader = get_data_loader(cfg, test, 'test')
        # else:
        #     test_loader = None
        test_loader = None

        # Setup the logger
        LOG_PATH = os.path.join(BASE_PATH, cfg['log_path'].format(PLANE), EXPERIMENT + '.log')
        logger.create_logger(LOG_PATH)

        # Set up the stats manager
        stats_manager = StatsManager(cfg)

        # Set up the checkpoint manager
        CHECKPOINT_PATH = os.path.join(BASE_PATH, cfg['checkpoint_path'].format(PLANE), EXPERIMENT)

        LOGGER.info(f'Device: {device}')

        # Initializing the model
        model = FCnnModel(cfg)

        # Log the number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        LOGGER.info(f'Number of parameters: {total_params}')

        # Create the data loaders
        # train_loader, val_loader, test_loader = get_data_loaders(cfg)

        # Initializing the loss & the optimizer
        loss_fn = get_loss_fn(cfg)

        # Initialize the optimizer
        optimizer_type = cfg['optimizer']
        optimizer = get_optimizer(model=model,
                                  optimizer=optimizer_type,
                                  learning_rate=cfg['lr'])

        # Initialize the learning rate scheduler
        lr_scheduler = get_lr_scheduler(cfg=cfg,
                                        optimizer=optimizer)

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
                          device=device,
                          experiment=EXPERIMENT,
                          checkpoint=CHECKPOINT_PATH)

        # Train
        trainer.train()

    # Test
    # trainer.test()
