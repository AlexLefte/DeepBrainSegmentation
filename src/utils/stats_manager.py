import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from src.utils import logger
from src.utils.metrics import *


class StatsManager:
    """
    Class that stores training/test stats and also writes them into the SummaryWriter

    Methods
    -------
    __init__
        Constructor
    """

    def __init__(self,
                 cfg: dict):
        """
        Constructor

        Parameters
        ----------
        cfg:
            Configuration dictionary
        """
        self.cfg = cfg
        self.num_classes = cfg['num_classes']
        self.summary_writer = SummaryWriter(cfg['summary_path'])
        self.batch_losses = []
        self.accuracy = AccScore(num_classes=self.num_classes)
        self.dice = DiceSimilarityCoefficient(num_classes=self.num_classes,
                                              dice_matr=True)
        self.batch_idx = 1

    def update_batch_stats(self,
                           mode: str,
                           preds: Tensor,
                           labels: Tensor,
                           loss: float,
                           ce_loss: float,
                           dice_loss: float,
                           accuracy: float,
                           lr=None):
        """
        Saves the loss, the learning rate, and other stats

        Parameters
        ----------
        mode:
            Train/Validation mode
        preds:
            Predictions
        labels:
            Ground truth
        loss:
            Total loss
        ce_loss:
            Weighted cross entropy loss
        dice_loss:
            Dice loss
        accuracy:
            Accuracy
        lr:
            Learning rate (Only for training)
        """
        # Add losses to the board
        self.summary_writer.add_scalar(f'Combined_loss/{mode}', loss, self.batch_idx)
        self.summary_writer.add_scalar(f'CE_loss/{mode}', ce_loss, self.batch_idx)
        self.summary_writer.add_scalar(f'Dice_loss/{mode}', dice_loss, self.batch_idx)

        # Add the accuracy and other metrics
        self.summary_writer.add_scalar(f'Accuracy/{mode}', accuracy, self.batch_idx)

        # If training, save the learning rate
        if mode == 'train':
            self.summary_writer.add_scalar(f'Learning rate', lr, self.batch_idx)

        # Update Accuracy & Dice matrices
        self.accuracy.update_cnf_matr(pred=preds,
                                      y=labels)
        # self.dice.update_dice(pred=preds,
        #                       y=labels)

        # Increment the batch idx
        self.batch_idx += 1

    def update_epoch_stats(self,
                           mode: str,
                           epoch: int):
        """
        Updates the confusion matrix, precision-recall curve and dice similarity coefficient
        """
        # Stop at epoch 8:
        if epoch == 5:
            print('Stop at epoch 8.')

        # Write the accuracy and the confusion matrix
        accuracy = self.accuracy.get_acc()
        conf_matrix = self.accuracy.get_confusion_matrix()
        self.summary_writer.add_scalar(f'Mean_acc/{mode}', accuracy, epoch)
        self.summary_writer.add_figure(f'Confusion_matrix/{mode}', conf_matrix, epoch)

        # Write the dice similarity coefficient and its associated matrix
        # dice = self.dice.get_dice_coefficient()
        # dice_matrix = self.dice.get_dice_matrix()
        # self.summary_writer.add_scalar(f'Dice_coefficient/{mode}', dice, epoch)
        # self.summary_writer.add_figure(f'Dice_matrix/{mode}', dice_matrix, epoch)

    def update_pr_curves(self):
        # TODO
        pass
