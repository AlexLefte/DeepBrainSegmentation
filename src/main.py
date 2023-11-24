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
    # Redirect standard output to the output file
    # with open(OUTPUT_PATH, 'w') as file:
    # sys.stdout = file  # Redirect standard output to the file

    # Initialize config
    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)
    # Construct the path to the parent directory
    parent_dir = os.path.dirname(current_dir)
    cfg = json.load(open(parent_dir + '/config/config.json', 'r'))

    # Data path
    DATA_PATH = cfg['data_path']
    EXPERIMENT = cfg['exp_name']
    BATCH_SIZE = cfg['batch_size']
    LOG_PATH = os.path.join(cfg['exp_path'], "log", EXPERIMENT + ".log")

    # Setup logger
    logger.create_logger(LOG_PATH)

    # Initialize CUDA
    # torch.cuda.init()

    # Set up the device
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    # # Testing the training loop
    # Initializing the model
    model = FCnnModel(cfg)

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

    # # Testing the dice loss
    # dice_loss = models.loss.DiceLoss()
    # a = torch.tensor([[[0.81, 0.15], [0.02, 0.3]], [[0.19, 0.85], [0.98, 0.7]]])
    # # b = torch.tensor([[[[1, 1], [0, 1]], [[0, 0], [1, 1]]], [[[0, 0], [1, 0]], [[1, 1], [0, 0]]]])
    # b = torch.tensor([[0, 1], [1, 0]])
    # loss = dice_loss(a, b)
    # print(loss)

    # Initializing some training data
    # We don't need for now the data
    # train_data = SubjectsDatasetTest(cfg=cfg,
    #                                  path=DATA_PATH,
    #                                  mode='val')

    # # Testing data loaders
    # train_data = get_data_loader(cfg,
    #                              DATA_PATH,
    #                              BATCH_SIZE,
    #                              'train')

    # # Test the data loader and display some transformed data
    # for batch_idx, batch in enumerate(train_data):
    #     if batch_idx == 3:
    #         images, labels, weights = (
    #             batch['image'],
    #             batch['labels'],
    #             batch['weights']
    #         )
    #         plt.figure()
    #         plt.imshow(images[10].squeeze(dim=0).numpy(), cmap='gray', origin='lower')
    #         plt.title('After')
    #         plt.show()

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
    # train_data_loader = loader.get_data_loader(cfg=cfg,
    #                                            data_path=DATA_PATH,
    #                                            batch_size=BATCH_SIZE,
    #                                            mode='train')

    # Initialize our model
    # model = FCnnModel(params=cfg).to(device)

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
