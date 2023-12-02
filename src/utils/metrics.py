import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


class AverageHausdorff:
    pass


class AccScore:
    """
    Stores the confusion matrix and computes the overall accuracy
    """
    def __init__(self,
                 num_classes: int):
        """
        Constructs an AccScore object
        """
        self.num_classes = num_classes
        self.cnf_matr = torch.zeros(self.num_classes, self.num_classes)

    def update_cnf_matr(self,
                        pred: Tensor,
                        y: Tensor):
        """
        Updates the confusion matrix
        """
        y = y.cpu().numpy().flatten()
        pred = pred.cpu().numpy().flatten()
        self.cnf_matr += confusion_matrix(y, pred, labels=np.asarray(range(self.num_classes)))

    def get_acc(self):
        """
        Computes and returns the accuracy
        """
        return self.cnf_matr.diagonal().sum() / self.cnf_matr.sum()

    def get_confusion_matrix(self):
        """
        Return the confusion matrix
        """
        # constant for classes
        classes = [str(x) for x in range(self.num_classes)]

        # Build confusion matrix
        cf_matrix = self.cnf_matr.cpu().numpy()
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        return sn.heatmap(df_cm, annot=True).get_figure()

    def reset(self):
        """
        Resets the Confusion Matrix
        """
        self.cnf_matr = torch.zeros(self.num_classes, self.num_classes)


class DiceSimilarityCoefficient:
    """
    Stores the dice similarity class matrix and computes the overall coefficient
    """
    def __init__(self,
                 num_classes: int,
                 dice_matr: bool = True):
        self.num_classes = num_classes
        self.shape = (self.num_classes, self.num_classes) if dice_matr else (self.num_classes, )
        self.intersect = torch.zeros(self.shape)
        self.union = torch.zeros(self.shape)
        self.dice_matr = dice_matr

    def update_dice(self,
                    pred: Tensor,
                    y: Tensor):
        """
        Updates the confusion matrix
        """
        for i in range(self.num_classes):
            # Get all indexes where class 'i' is found
            labels_i = y == i
            if self.dice_matr:
                for j in range(self.num_classes):
                    # Get all indexes where class 'j' was predicted
                    preds_j = pred == j
                    self.intersect[i][j] += (labels_i * preds_j).sum()
                    self.union[i][j] += (labels_i + preds_j).sum()
            else:
                preds_i = pred == i
                self.intersect[i] = (labels_i * preds_i).sum()
                self.union[i] = (labels_i + preds_i).sum()

    def get_dice_coefficient(self):
        """
        Computes the overall dice
        """
        intersect = torch.diagonal(self.intersect)
        union = torch.diagonal(self.union)
        dice = 2 * torch.div(intersect, union)
        return torch.mean(dice)

    def get_dice_matrix(self):
        dice_matr = 2 * torch.div(self.intersect, self.union).cpu().numpy()

        # constant for classes
        classes = [str(x) for x in range(self.num_classes)]

        # Build dice matrix
        df_cm = pd.DataFrame(dice_matr / np.sum(dice_matr, axis=1)[:, None], index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        return sn.heatmap(df_cm, annot=True).get_figure()

    def reset(self):
        """
        Resets the intersection & union Matrix
        """
        self.intersect = torch.zeros(self.shape)
        self.union = torch.zeros(self.shape)
