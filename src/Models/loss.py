import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1.0):
        """
        Ref:
        https://paperswithcode.com/method/dice-loss
        https://arxiv.org/pdf/2006.14822.pdf
        https://link.springer.com/chapter/10.1007/978-3-319-66179-7_27
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


# def dice_loss(y_true: Tensor,
#               y_pred: Tensor,
#               eps: float = 1e-5):
#
#     intersection = torch.sum(y_true * y_pred)
#     union = torch.sum(y_true) + torch.sum(y_pred)
#     return (2.0 * intersection + eps) / (union + eps)


# Create your weighted cross-entropy loss function
class WeightedCELoss(nn.Module):
    """
    Computes the weighted Cross-Entropy loss
    """
    def __init__(self, weights):
        """
        Constructor
        """
        super(WeightedCELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        loss = nn.functional.cross_entropy(input=y_pred,
                                           target=y_true,
                                           weight=self.weights)
        return loss


class CombinedLoss(nn.Module):
    """
    Computes the result of a composite loss function:
    Median frequency balanced logistic loss + Dice loss
    """

    def __init__(self, weights):
        """
        Constructor
        """
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy = WeightedCELoss(weights=weights)
        self.weights = weights

    def forward(self,
                input: Tensor,
                targets: Tensor):
        """
        Computes the composite loss
        """
        if input.is_cuda():
            targets = targets.to('cuda')

        cross_entropy_loss = self.cross_entropy(y_pred=input,
                                                y_true=targets)
        dice_loss = self.dice_loss(y_pred=input,
                                   y_true=targets)
        return cross_entropy_loss + dice_loss




