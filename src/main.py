import torch
from torch import nn
import numpy as np
import os
# import rarfile as rar
import nibabel as nib
import data
from data.dataset import SubjectsDataset
from data import data_loader as loader
from models.FCnnModel import FCnnModel
import sys
import matplotlib.pyplot as plt
from utils import logger
import json


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Redirect standard output to the output file
    # with open(OUTPUT_PATH, 'w') as file:
    # sys.stdout = file  # Redirect standard output to the file

    # Initialize config
    cfg = json.load(open('./config.json', 'r'))

    # Data path
    DATA_PATH = cfg['data_path']
    EXPERIMENT = cfg['exp_name']
    LOG_PATH = os.path.join(cfg['exp_path'], "log", EXPERIMENT + ".log")

    # Setup logger
    logger.create_logger(os.path.join(cfg))

    # Initialize CUDA
    torch.cuda.init()

    # Set up the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # Initializing some training data
    # We don't need for now the data
    train_data = SubjectsDataset(DATA_PATH)

    # Define some parameters
    # params = {
    #     'in_channels': 1,
    #     'filters': 64,
    #     'kernel_h': 5,
    #     'kernel_w': 5,
    #     'stride': 1,
    #     'pool': 2,
    #     'pool_kernel': 1,
    #     'pool_stride': 2,
    #     'classifier_kernel': 1,
    #     'num_classes': 79,
    #     'device': device
    # }

    # Test the DataLoaders:
    train_data_loader = loader.get_data_loader(cfg=cfg,
                                               data_path=DATA_PATH,
                                               batch_size=BATCH_SIZE,
                                               mode='train')

    # Initialize our model
    model = FCnnModel(params=cfg).to(device)

    # Select one slice from our training set
    # mri, labeled_mri = train_data.__getitem__(0)
    # image, labels = mri[len(mri) // 2], labeled_mri[len(labeled_mri) // 2]
    #
    # # Set data on device
    # image, labels = image.to(torch.float32).to(device), labels.to(torch.float32).to(device)
    # uniques = torch.unique(image).cpu().numpy()
    # print(uniques)
    # print(len(uniques))


    # # Create a histogram
    # plt.hist(uniques, bins=20, color='blue', edgecolor='black')
    #
    # # Set labels and title
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Tensor Values')
    #
    # # Display the histogram
    # plt.show()


    # # Check shapes
    # print(f"Image shape: {image.shape}. Labeled image shape: {labels.shape}")
    # print(f"Image data type: {image.dtype}")

    # Checking structures labeling
    # print(torch.unique(labels))

    # Observe the tensor shapes when forwarding the image through the network
    # result = model(torch.unsqueeze(torch.unsqueeze(image, 0), 1))
    # print(result.shape)
