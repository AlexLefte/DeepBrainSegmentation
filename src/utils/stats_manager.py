import os
from torch.utils.tensorboard import SummaryWriter
import logging
from src.utils.metrics import *

LOGGER = logging.getLogger(__name__)


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
            The number of classes
        path: string
            The experiment's path
        """
        self.num_classes = num_classes
        self.summary_writer = SummaryWriter(path)
        self.batch_losses = []
        self.y_pred = []
        self.y_true = []
        self.batch_idx = 1

    def update_batch_stats(self,
                           preds: Tensor,
                           labels: Tensor):
        """
        Saves the loss, the learning rate, and other stats

        Parameters
        ----------
        preds:
            Predictions
        labels:
            Ground truth
        """
        # Append the predictions and labels
        self.y_pred.extend(preds.cpu().numpy())
        self.y_true.extend(labels.cpu().numpy())

    def update_epoch_stats(self,
                           mode: str,
                           loss: float,
                           learning_rate: float,
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

        # Write the loss value
        self.summary_writer.add_scalar(f'Loss/{mode}', loss, epoch)

        # Write the accuracy and the confusion matrix
        accuracy = get_accuracy(y_pred_flat,
                                y_true_flat,
                                self.num_classes)
        self.summary_writer.add_scalar(f'Accuracy/{mode}', accuracy, epoch)
        # conf_matrix = self.accuracy.get_matrix()
        # self.summary_writer.add_figure(f'Confusion_matrix/{mode}', conf_matrix, epoch)

        # Write the dice similarity coefficient and its associated matrix
        dice = get_overall_dsc(y_pred_flat,
                               y_true_flat)
        self.summary_writer.add_scalar(f'DSC/{mode}', dice, epoch)
        # dice_matrix = self.dice.get_matrix()
        # self.summary_writer.add_figure(f'Dice_matrix/{mode}', dice_matrix, epoch)

        # Write the mean dsc (per class)
        # dice_per_class = get_class_dsc(y_pred_flat,
        #                                y_true_flat,
        #                                self.num_classes)
        dice_sub, dice_cort, dice_mean = get_cortical_subcortical_class_dsc(y_pred_flat,
                                           y_true_flat,
                                           self.num_classes)
        # self.summary_writer.add_scalar(f'DSC_mean_per_class/{mode}', dice_per_class, epoch)
        self.summary_writer.add_scalar(f'DSC_sub/{mode}', dice_sub, epoch)
        self.summary_writer.add_scalar(f'DSC_cort/{mode}', dice_cort, epoch)
        self.summary_writer.add_scalar(f'DSC_mean_per_class/{mode}', dice_mean, epoch)

        # Write the current learning rate:
        if mode == 'train':
            self.summary_writer.add_scalar(f'Learning rate', learning_rate, self.batch_idx)

        # Log some info
        mode = str.capitalize(mode)
        # LOGGER.info(f"Epoch: {epoch} | {mode} loss: {loss:.4f} | {mode} dsc: {dice_per_class:.4f} | "
        #             f"{mode} accuracy: {accuracy:.4f}")

        # Reset the prediction/ground truth lists
        self.y_pred = []
        self.y_true = []

        # return dice_per_class
        return dice_mean

    def update_pr_curves(self):
        # TODO
        pass
