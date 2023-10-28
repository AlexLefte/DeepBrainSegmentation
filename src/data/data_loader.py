import torch.utils.data
from dataset import *
from torchvision import transforms
from torch.utils.data import DataLoader


def get_data_loader(cfg,
                    data_path: str,
                    batch_size: int,
                    mode: str):
    """
    Creates a PyTorch data using the subjects' custom dataset

    Parameters
    ----------
    data_path: str
        data directory path

    batch_size: int
        the size of the batch

    mode: str
        train/test data loader

    shuffle: bool
        whether to shuffle the data or not

    Returns
    -------
    data_loader: torch.utils.data
    """

    # For transforms, also check : https://torchio.readthedocs.io/transforms/augmentation.html

    if mode == 'train':
        # Create a transform
        transform = transforms.Compose([
            transforms.Resize(size=(256, 311, 320)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ])
        shuffle = True
    elif mode == 'val' or mode == 'test':
        transform = transforms.Compose([
            transforms.Resize(size=(256, 311, 320)),
            transforms.ToTensor()
        ])
        shuffle = False
    # elif mode == "test":
    #     transform = transforms.Compose([
    #         transforms.Resize(size=(64, 64)),
    #         transforms.ToTensor()
    #     ])
    #     shuffle = False
    else:
        print("Invalid mode.")
        return None

    # Create the SubjectsDataset instance
    dataset = SubjectsDataset(cfg=cfg,
                              path=data_path,
                              transform=transform,
                              target_transform=None)

    # Create data
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

    return data_loader

