import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1.0):
        """
            Parameters
            ----------
            eps
                To avoid multiplying and dividing with 0
        """
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
            Returns the dice loss

            Parameters
            ----------
            y_true
                Ground truth
            y_pred
                Predictions
        """
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice


def dice_loss(y_true: Tensor,
              y_pred: Tensor,
              eps: float = 1e-5):

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    return (2.0 * intersection + eps) / (union + eps)


# Create your weighted cross-entropy loss function
class WeightedBCELoss(nn.Module):
    """
    Computes the weighted Binary Cross-Entropy loss
    """
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=self.weights)
        return loss


class CombinedLoss(nn.Module):
    """
        Computes the result of a composite loss function:
        Median frequency balanced logistic loss + Dice loss
    """
    pass

