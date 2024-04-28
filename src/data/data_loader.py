import torch.utils.data
from src.data.dataset import *
from torchvision import transforms
import torchio as tio
from torch.utils.data import DataLoader
import random


def get_data_loader(cfg, split, mode):
    """
    Creates a PyTorch data loader
    """
    # Get the batch size
    batch_size = cfg['batch_size']

    # Get loader
    dataset = SubjectsDataset(cfg=cfg,
                              subjects=split,
                              mode=mode)
    shuffle = mode == 'train'
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle
    )

    return loader


def get_data_loaders(cfg):
    """
    Creates a PyTorch data loader using the subjects' custom dataset

    Parameters
    ----------
    cfg: dict
        configuration parameters

    Returns
    -------
    data_loader: torch.utils.data
    """
    # Get the batch size
    batch_size = cfg['batch_size']

    # Get validation loader flag:
    validation = cfg['val_data_loader']

    # Get test loader flag:
    test = cfg['test_data_loader']

    # Initialize the sets
    train_set, val_set, test_set = [], [], []

    # Check if hdf5 dataset exists
    if not cfg['hdf5_dataset']:
        # Get the data path
        data_path = cfg['base_path'] + cfg['data_path']

        # Get the subjects' paths
        subject_paths = [os.path.join(data_path, s) for s in os.listdir(data_path)
                         if os.path.isdir(os.path.join(data_path, s))]

        # Get the test/val/train split
        train_set, val_set, test_set = du.get_train_test_split(subject_paths,
                                                               cfg['train_size'],
                                                               cfg['test_size'])

    # Creating the custom datasets
    # # Training DataLoader
    train_dataset = SubjectsDataset(cfg=cfg,
                                    subjects=train_set,
                                    mode='train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )

    # # Validation DataLoader
    if validation:
        val_dataset = SubjectsDataset(cfg=cfg,
                                      subjects=val_set,
                                      mode='val')
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size
        )

    # Test DataLoader
    if test:
        # For testing purposes, the test dataloader will be composed of a subject,
        # to see how the network performs
        test_dataset = SubjectsDataset(cfg=cfg,
                                       subjects=test_set,
                                       mode='test')
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


def get_inference_data_loader(path, cfg, plane):
    """
    Creates a test data loader

    Parameters
    ----------
    path: str
        Path towards the input subject file or towards the input directory
    cfg: dict
        Configuration settings
    """
    if os.path.isdir(path):
        subjects_list = [os.path.join(path, s) for s in os.listdir(path)]
    elif os.path.isfile(path):
        subjects_list = [path]
    else:
        raise ValueError(f"{path} is neither a directory nor a file.")

    # Create both the custom dataset and the data loader
    dataset = InferenceSubjectsDataset(cfg=cfg,
                                       subjects=subjects_list,
                                       plane=plane)
    loader = DataLoader(dataset=dataset,
                        batch_size=cfg['batch_size'])

    return loader
