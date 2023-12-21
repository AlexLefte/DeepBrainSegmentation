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
                 num_classes: int,
                 path: str):
        """
        Constructor

        Parameters
        ----------
        num_classes: int
            THe number of classes
        path: string
            The experiment's path
        """
        self.num_classes = num_classes
        self.summary_writer = SummaryWriter(path)
        self.batch_losses = []
        # self.accuracy = AccScore(num_classes=self.num_classes)
        # self.dice = DiceSimilarityCoefficient(num_classes=self.num_classes,
        #                                       dice_matr=False)
        self.y_pred = []
        self.y_true = []
        self.batch_idx = 1

    def update_batch_stats(self,
                           mode: str,
                           preds: Tensor,
                           labels: Tensor,
                           loss: float,
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
        accuracy:
            Accuracy
        lr:
            Learning rate (Only for training)
        """
        # Append the predictions and labels
        self.y_pred.extend(preds.cpu().numpy())
        self.y_true.extend(labels.cpu().numpy())

        # Add losses to the board
        self.summary_writer.add_scalar(f'Loss/{mode}', loss, self.batch_idx)

        # Add the accuracy and other metrics
        self.summary_writer.add_scalar(f'Accuracy/{mode}', accuracy, self.batch_idx)

        # If training, save the learning rate
        if mode == 'train':
            self.summary_writer.add_scalar(f'Learning rate', lr, self.batch_idx)

        # # Update Accuracy & Dice matrices
        # self.accuracy.update(y_pred=preds,
        #                      y_true=labels)
        # self.dice.update(y_pred=preds,
        #                  y_true=labels)

        # Increment the batch idx
        self.batch_idx += 1

    def update_epoch_stats(self,
                           mode: str,
                           epoch: int):
        """
        Updates the confusion matrix, precision-recall curve and dice similarity coefficient
        """
        # Concatenate both the predictions and labels
        y_pred_concat = np.concatenate(self.y_pred)
        y_true_concat = np.concatenate(self.y_true)

        # Flatten these np.ndarray-type objects
        y_pred_flat = y_pred_concat.flatten()
        y_true_flat = y_true_concat.flatten()

        # Write the accuracy and the confusion matrix
        # accuracy = self.accuracy.get_score()
        # self.accuracy.reset()
        accuracy = get_accuracy(y_pred_flat,
                                y_true_flat,
                                self.num_classes)
        self.summary_writer.add_scalar(f'Mean_acc/{mode}', accuracy, epoch)
        # conf_matrix = self.accuracy.get_matrix()
        # self.summary_writer.add_figure(f'Confusion_matrix/{mode}', conf_matrix, epoch)

        # Write the dice similarity coefficient and its associated matrix
        # dice = self.dice.get_score()
        # self.dice.reset()
        dice = get_overall_dsc(y_pred_flat,
                               y_true_flat)
        self.summary_writer.add_scalar(f'Overall_DSC/{mode}', dice, epoch)
        # dice_matrix = self.dice.get_matrix()
        # self.summary_writer.add_figure(f'Dice_matrix/{mode}', dice_matrix, epoch)

        # Write the mean class dsc
        dice_per_class = get_class_dsc(y_pred_flat,
                                       y_true_flat,
                                       self.num_classes)
        self.summary_writer.add_scalar(f'DSC_per_class/{mode}', dice_per_class, epoch)

        # Reset the prediction/ground truth lists
        self.y_pred = []
        self.y_true = []

    def update_pr_curves(self):
        # TODO
        pass
