import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Metric base class

    Methods
    -------
    update:
        Updates the metrics
    get_score:
        Returns the score value
    get_matrix:
        Returns the score matrix
    reset:
        Resets to the initial values
    """
    @abstractmethod
    def update(self, y_pred: Tensor, y_true: Tensor):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def get_matrix(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class AverageHausdorff:
    pass


class AccScore(Metric):
    """
    Stores the confusion matrix and computes the overall accuracy
    """
    def __init__(self,
                 num_classes: int,
                 device: str = 'cpu'):
        """
        Constructs an AccScore object
        """
        super().__init__()
        self.num_classes = num_classes
        self.cnf_matr = torch.zeros(self.num_classes, self.num_classes).to(device)
        self.device = device

    def update(self,
               y_pred: Tensor,
               y_true: Tensor):
        """
        Updates the confusion matrix
        """
        y_true = y_true.cpu().numpy().flatten()
        y_pred = y_pred.cpu().numpy().flatten()
        batch_cnf_matr = confusion_matrix(y_true, y_pred, labels=np.asarray(range(self.num_classes)))
        self.cnf_matr += torch.tensor(batch_cnf_matr).to(self.device)

    def get_score(self):
        """
        Computes and returns the accuracy
        """
        return self.cnf_matr.diagonal().sum() / self.cnf_matr.sum()

    def get_matrix(self):
        """
        Return the confusion matrix
        """
        # Classes list
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
        self.cnf_matr = torch.zeros(self.num_classes, self.num_classes).to(self.device)


class DiceSimilarityCoefficient:
    """
    Stores the dice similarity class matrix and computes the overall coefficient
    """
    def __init__(self,
                 num_classes: int,
                 dice_matr: bool = True,
                 device: str = 'cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.shape = (self.num_classes, self.num_classes) if dice_matr else (self.num_classes, )
        self.intersect = torch.zeros(self.shape).to(device)
        self.union = torch.zeros(self.shape).to(device)
        self.dice_matr = dice_matr
        self.device = device
        self.y_pred = []
        self.y_true = []

    def update(self,
               y_pred: Tensor,
               y_true: Tensor):
        """
        Updates the confusion matrix
        """
        # for i in range(self.num_classes):
        #     # Get all indexes where class 'i' is found
        #     labels_i = y_true == i
        #     if self.dice_matr:
        #         for j in range(self.num_classes):
        #             # Get all indexes where class 'j' was predicted
        #             preds_j = y_pred == j
        #             self.intersect[i][j] += (labels_i * preds_j).sum()
        #             self.union[i][j] += (labels_i + preds_j).sum()
        #     else:
        #         preds_i = y_pred == i
        #         self.intersect[i] += (labels_i * preds_i).sum()
        #         self.union[i] += (labels_i + preds_i).sum()
        self.y_pred.extend(y_pred.cpu().numpy())
        self.y_true.extend(y_true.cpu().numpy())

    def get_score(self):
        """
        Computes the overall dice
        """
        # intersect = np.intersect1d(self.y_pred, self.y_true, return_indices=False)
        # intersect = np.sum(self.y_pred == self.y_true)
        self.y_pred = np.concatenate(self.y_pred).flatten()
        self.y_true = np.concatenate(self.y_true).flatten()
        intersect = np.sum(np.where((self.y_pred == self.y_true), 1, 0))
        dice_score = 2 * intersect / (self.y_pred.size + self.y_true.size)
        # print(dice_score)
        return dice_score
        # if self.dice_matr:
        #     intersect = torch.diagonal(self.intersect)
        #     union = torch.diagonal(self.union)
        #     dice = 2 * torch.div(intersect, union)
        # else:
        #     dice = 2 * torch.div(self.intersect, self.union)
        # return torch.mean(dice)

    def get_matrix(self):
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
        self.intersect = torch.zeros(self.shape).to(self.device)
        self.union = torch.zeros(self.shape).to(self.device)
        self.y_true = []
        self.y_pred = []


def get_confusion_matrix(y_pred: np.ndarray,
                         y_true: np.ndarray,
                         num_classes: int):
    """
    Returns the confusion matrix
    """
    return confusion_matrix(y_true, y_pred, labels=np.asarray(range(num_classes)))


def get_cnf_matrix_figure(y_pred: np.ndarray,
                          y_true: np.ndarray,
                          num_classes: int):
    """
    Returns the confusion matrix figure, ready to be written in the SummaryWriter
    """
    # Classes list
    classes = [str(x) for x in range(num_classes)]

    # Build confusion matrix
    cf_matrix = get_confusion_matrix(y_pred,
                                     y_true,
                                     num_classes)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    return sn.heatmap(df_cm, annot=True).get_figure()


def get_accuracy(y_pred: np.ndarray,
                 y_true: np.ndarray,
                 percentage: True):
    """
    Computes and returns the accuracy score
    """
    acc = np.mean(y_pred == y_true)
    if percentage:
        acc *= 100
    return acc


def get_overall_dsc(y_pred: np.ndarray,
                    y_true: np.ndarray):
    """
    Returns the overall dice score
    """
    intersect = np.sum(y_pred == y_true)
    dice_score = 2 * intersect / (y_pred.size + y_true.size)
    return dice_score


def get_cortical_subcortical_class_dsc(y_pred: np.ndarray,
                                      y_true: np.ndarray,
                                      num_classes: int):
    """
    Returns the overall dice score
    """
    # Initialize the intersect and union
    intersect = []
    union = []

    for i in range(num_classes):
        # Get all indexes where class 'i' is found
        labels_i = (y_true == i).astype(int)

        # Get all indexes for which class 'i' has been predicted
        preds_i = (y_pred == i).astype(int)

        # Compute the intersection and union
        intersect.append(np.sum(labels_i * preds_i))
        union.append(np.sum(labels_i + preds_i))

    # Compute the dice score per class
    intersect = np.asarray(intersect)
    union = np.asarray(union)
    dsc = 2 * (intersect / union)
    return np.mean(dsc[1:34]), np.mean(dsc[34:]), np.mean(dsc[1:])


def get_class_dsc(y_pred: np.ndarray,
                  y_true: np.ndarray,
                  num_classes: int,
                  return_mean: bool = True):
    """
    Returns the overall dice score
    """
    # Initialize the intersect and union
    intersect = []
    union = []

    for i in range(num_classes):
        # Get all indexes where class 'i' is found
        labels_i = (y_true == i).astype(int)

        # Get all indexes for which class 'i' has been predicted
        preds_i = (y_pred == i).astype(int)

        # Compute the intersection and union
        intersect.append(np.sum(labels_i * preds_i))
        union.append(np.sum(labels_i + preds_i))

    # Compute the dice score per class
    intersect = np.asarray(intersect)
    union = np.asarray(union)
    dsc = 2 * (intersect / union)
    if return_mean:
        return np.mean(dsc)
    else:
        return dsc
