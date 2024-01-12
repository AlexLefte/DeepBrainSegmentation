import torch.utils.data
from src.data.dataset import *
from torchvision import transforms
import torchio as tio
from torch.utils.data import DataLoader
import random


def get_data_loaders(cfg):
    """
    Creates a PyTorch data using the subjects' custom dataset

    Parameters
    ----------
    cfg: dict
        configuration parameters

    Returns
    -------
    data_loader: torch.utils.data
    """
    # Get the data path
    data_path = cfg['base_path'] + cfg['data_path']

    # Get the batch size
    batch_size = cfg['batch_size']

    # Get the subjects' paths
    subject_paths = [os.path.join(data_path, s) for s in os.listdir(data_path)
                     if os.path.isdir(os.path.join(data_path, s))]

    # Shuffle the subjects
    random.shuffle(subject_paths)

    # Return validation mode:
    validation = cfg['val_data_loader']

    # Return test mode:
    test = cfg['test_data_loader']

    # Get the dataset length
    subjects_count = len(subject_paths)

    # Get the sizes of each split
    if validation:
        train_size = int(cfg['train_size'] * subjects_count)
        if test:
            val_size = int(cfg['val_size'] * subjects_count)
        else:
            val_size = subjects_count - train_size
    else:
        train_size = subjects_count

    # Creating the custom datasets
    # # Training DataLoader
    train_set = subject_paths[:train_size]
    train_set = [subject_paths[0]]
    train_dataset = SubjectsDataset(cfg=cfg,
                                    subjects=train_set,
                                    mode='train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        pin_memory=True
    )

    # # Validation DataLoader
    if validation:
        val_set = subject_paths[train_size: train_size + val_size]
        val_set = [subject_paths[1]]
        val_dataset = SubjectsDataset(cfg=cfg,
                                      subjects=val_set,
                                      mode='test',
                                      weights_dict=train_dataset.weights_dict)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size
        )

    # # Test DataLoader
    if test:
        test_set = subject_paths[train_size + val_size:]
        test_dataset = SubjectsDataset(cfg=cfg,
                                       subjects=test_set,
                                       mode='test',
                                       weights_dict=train_dataset.weights_dict)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size
        )

    if test:
        return train_loader, val_loader, test_loader
    elif validation:
        return train_loader, val_loader, None
    else:
        return train_loader, None, None
