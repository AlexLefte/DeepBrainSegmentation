import torch.utils.data
from src.data.dataset import *
from torchvision import transforms
import torchio as tio
from torch.utils.data import DataLoader


def get_data_loader(cfg,
                    data_path: str,
                    batch_size: int,
                    mode: str):
    """
    Creates a PyTorch data using the subjects' custom dataset

    Parameters
    ----------
    cfg: dict
        configuration parameters

    data_path: str
        data directory path

    batch_size: int
        the size of the batch

    mode: str
        train/test data loader

    Returns
    -------
    data_loader: torch.utils.data
    """

    # For transforms, also check : https://torchio.readthedocs.io/transforms/augmentation.html

    # Assert shapes match
    assert (mode == 'train'
            or mode == 'test'
            or mode == 'eval'), 'Incorrect data_loader mode.'

    if mode == 'train':
        # Perform data augmentation
        transform = get_aug_transforms()
        shuffle = True

        # Create the SubjectsDataset instance
        dataset = SubjectsDataset(cfg=cfg,
                                  path=data_path,
                                  mode=mode,
                                  transform=transform)
    elif mode == 'eval' or mode == 'test':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        shuffle = False

        # Create the SubjectsDataset instance
        dataset = SubjectsDatasetTest(cfg=cfg,
                                      path=data_path,
                                      mode=mode,
                                      transform=transform)
    else:
        print("Invalid mode.")
        return None

    # Create data
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

    return data_loader


def get_aug_transforms():
    """
    Provides data augmentation transforms
    See U-net paper:  https://arxiv.org/pdf/1809.10486.pdf (section Data Augmentation)
    See TorchIO -> Transforms -> Augmentation: https://torchio.readthedocs.io/transforms/augmentation.html
    """
    # Append: rotation -> scaling -> elastic deformation -> gamma correction
    transforms_list = tio.Compose([
        tio.RandomAffine(
            scales=(1.0, 1.0),
            degrees=10,
            translation=(0, 0, 0),
            isotropic=True,
            center='image',
            default_pad_value='minimum',
            image_interpolation='linear',
            include=['img', 'label', 'weight'],
        ),
        tio.RandomAffine(
            scales=(0.8, 1.15),
            degrees=0,
            translation=(0, 0, 0),
            isotropic=True,
            center='image',
            default_pad_value='minimum',
            image_interpolation='linear',
            include=['img', 'label', 'weight'],
        ),
        # tio.RandomElasticDeformation(
        #     num_control_points=7,
        #     max_displacement=15,
        #     locked_borders=4,
        #     image_interpolation='linear',
        #     include=['img', 'label', 'weight'],
        # ),
        tio.transforms.RandomGamma(
            log_gamma=(-0.3, 0.3), include=['img']
        )]
    )

    return transforms_list
