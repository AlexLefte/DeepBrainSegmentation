import torch
from torch import Tensor, nn
import src.data.data_utils as du
import numpy as np


def get_loss_fn(loss_type: str):
    """
    Returns the loss function

    Parameters
    ----------
    loss_type: string
        The loss type
    """
    # Initialize the loss function
    loss_fn = None

    # Choose the loss type accordingly
    if loss_type == 'dice_loss_&_cross_entropy':
        loss_fn = CombinedLoss()
    elif loss_type == 'unified_focal_loss':
        loss_fn = Unified_CatFocal_FocalTversky()

    # Return loss function
    return loss_fn


def get_one_hot_encoded(t: Tensor,
                        classes_num: int):
    """
    Computes the one-hot encoded version of a tensor

    Parameters 
    ----------
    t: tensor-like of shape (D, H, W)
        The ground truth tensor to be one-hot encoded
    classes_num: int
        The number of classes

    Returns
    -------
    t_encoded: tensor-like of shape (D, C, H, W)
        The one-hot encoded ground truth tensor
    """
    # Ensure ground_truth is a Long tensor
    t = t.long()

    # Apply one-hot encoding
    t_encoded = torch.nn.functional.one_hot(t, classes_num)

    # Reshape to the desired shape (D, C, H, W)
    t_encoded = t_encoded.permute(0, 3, 1, 2)

    # Return the encoded tensor
    return t_encoded


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

    def forward(self,
                y_pred: Tensor,
                y_true: Tensor,
                weights: Tensor = None):
        """
        Returns the dice loss

        Parameters
        ----------
        y_true
            Ground truth
        y_pred
            Predictions
        weights
            Class weights
        """
        # Get the number of classes:
        c = y_pred.shape[1]

        # Get the one-hot encoded version of the ground truth tensor
        y_true_encoded = get_one_hot_encoded(t=y_true,
                                             classes_num=c)

        # Apply softmax on the prediction logits
        y_pred = torch.softmax(input=y_pred,
                               dim=1)

        # Check the weights
        if weights is None:
            weights = torch.ones(c).to(y_pred.device)

        # Compute the intersection (numerator)
        intersection = (y_pred * y_true_encoded).sum(dim=0).sum(dim=1).sum(dim=1)

        # Compute the union (denominator)
        union = (y_pred + y_true_encoded).sum(dim=0).sum(dim=1).sum(dim=1)

        # Compute the channel-wise loss
        dice_per_channel = weights * (1.0 - (2.0 * intersection) / (union + self.eps))
        return dice_per_channel.mean()


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
    def __init__(self):
        """
        Constructor
        """
        super(WeightedCELoss, self).__init__()

    @staticmethod
    def forward(self,
                y_pred,
                y_true,
                weights):
        return nn.functional.cross_entropy(input=y_pred,
                                           target=y_true,
                                           weight=weights)


class CombinedLoss(nn.Module):
    """
    Computes the result of a composite loss function:
    Median frequency balanced logistic loss + Dice loss
    """

    def __init__(self):
        """
        Constructor
        """
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        # self.dice_loss_2 = DiceLoss2()
        # self.cross_entropy = WeightedCELoss()
        # self.weights = weights

    def forward(self,
                y_pred: Tensor,
                y_true: Tensor,
                weights: Tensor,
                weights_list: Tensor):
        """
        Computes the composite loss
        """
        if y_pred.is_cuda:
            y_true = y_true.to('cuda')

        # See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        cross_entropy_loss = nn.functional.cross_entropy(input=y_pred,
                                                         target=y_true.long(),
                                                         weight=weights_list)

        dice_loss = self.dice_loss(y_pred=y_pred,
                                   y_true=y_true,
                                   weights=None)

        # dice_loss = self.dice_loss_2(y_pred=torch.nn.functional.softmax(y_predict, dim=1),
        #                                y_true=y,
        #                                weights=weights)

        return cross_entropy_loss + dice_loss, cross_entropy_loss, dice_loss
        # return cross_entropy_loss


class DiceLoss2(nn.Module):
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
        super(DiceLoss2, self).__init__()
        self.eps = eps

    def forward(self,
                y_pred: Tensor,
                y_true: Tensor,
                weights: Tensor = None):
        """
        Returns the dice loss

        Parameters
        ----------
        y_true
            Ground truth
        y_pred
            Predictions
        weights
            Class weights
        """
        # Create a tensor with the same shape as y_pred, filled with 0
        y_true_encoded = torch.zeros_like(y_pred)

        # Substitute the value at the index associated with the correct class with 1.
        # Iterate over each class
        for class_index in range(y_pred.shape[1]):
            # Identify the indices where the target tensor has the current class
            # class_indices = (y_true == class_index).nonzero(as_tuple=False)  # initial
            class_indices = (y_true == class_index)

            # Set the corresponding values in the binary prediction tensor to 1 for the current class
            y_true_encoded[class_indices[:, 0], class_index, class_indices[:, 1], class_indices[:, 2]] = 1

        # Check the weights
        if weights is None:
            weights = 1

        # Compute the intersection (numerator)
        intersection = torch.sum(y_pred * y_true_encoded)

        # Compute the union (denominator)
        union = torch.sum(y_pred + y_true_encoded)

        # Compute the channel-wise loss
        dice = weights * (1.0 - (2.0 * intersection + self.eps) / (union + self.eps))
        return dice.sum() / y_pred.shape[1]


class CategoricalFocalLoss(nn.Module):
    """
    Implements the focal loss
    href: https://www.sciencedirect.com/science/article/pii/S0895611121001750
    """
    def __init__(self,
                 alpha: float = 0.7,
                 gamma: float = 2,
                 device: str = 'cpu',
                 suppress_bkg: bool = False):
        """
        Constructor

        Parameters
        ----------
        alpha: float
            Places a greater penalty on false negatives, rather on false positives
        gamma: float or tensor-like of shape (C)
            Down-weighting exponent
        device: str
            Training device
        suppress_bkg: bool
            True if we want to place a greater penalty on false positives for the background class
        """
        super(CategoricalFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.suppress_bkg = suppress_bkg

    def forward(self,
                y_pred: Tensor,
                y_true: Tensor,
                weights: Tensor,
                weights_list: Tensor):
        """
        Forward method

        Parameters
        ----------
        y_pred: tensor-like of shape (D, C, H, W)
            Prediction logits
        y_true: tensor-like of shape (D, H, W)
            Ground truth
        weights: Tensor-like of shape: (D, C, H, W)
            Class weights
        weights_list: Tensor-like of shape (, 79)
            Class weights list
        """
        # Get the number of classes
        c = y_pred.shape[1]

        # Get the one-hot encoded version of the ground truth tensor
        y_true_encoded = get_one_hot_encoded(t=y_true,
                                             classes_num=c)

        # Prepare the class weights tensor and the gamma tensor
        # if self.alpha is float:
        #     alpha = torch.ones(c) * self.alpha
        # if self.gamma is float:
        #     self.gamma = torch.ones(c) * self.gamma

        # Compute the cross_entropy
        ce = -y_true_encoded * torch.log(y_pred + 0.000001)

        if self.suppress_bkg:
            # Compute the background loss
            bkg_probs = y_pred[:, 0, :, :]
            bkg_ce = ce[:, 0, :, :]
            bkg_loss = (1 - self.alpha) * torch.pow(1 - bkg_probs, self.gamma) * bkg_ce
            bkg_loss = bkg_loss.unsqueeze(dim=1)

            # Compute the foreground loss
            fg_ce = ce[:, 1:, :, :]
            fg_loss = self.alpha * fg_ce

            # print(f'Bg ce: {str(bkg_ce)}, fg ce: {str(fg_ce)}')
            # bkg_print_loss = torch.mean(torch.sum(bkg_loss, dim=1))
            # fg_print_loss = torch.mean(torch.sum(fg_loss, dim=1))
            # print(f'Bg loss: {str(bkg_print_loss)}, fg loss: {str(fg_print_loss)}')

            # Stack these losses
            focal_loss = torch.cat(([bkg_loss, fg_loss]), dim=1)
        else:
            self.alpha = weights

            # Compute the modulating_factor
            modulating_factor = (1 - y_pred) ** self.gamma

            # Compute the categorical focal loss
            focal_loss = modulating_factor * ce

        # Return the mean loss
        focal_loss = torch.mean(self.alpha * torch.sum(focal_loss, dim=1))
        return focal_loss


class FocalTverskyLoss(nn.Module):
    """
    Implements the focal Tversky loss
    href: https://www.sciencedirect.com/science/article/pii/S0895611121001750
    """
    def __init__(self,
                 alpha: float = 0.7,
                 gamma: float = 3 / 4,
                 device: str = 'cpu',
                 eps: float = 0.000001,
                 suppress_bkg: bool = False):
        """
        Constructor

        Parameters
        ----------
        alpha: float
            Increasing the alpha factor places a greater penalty on false negatives compared to false positives.
        gamma: float or tensor-like of shape (C)
            Down-weighting exponent
        device: str
            Training device
        eps: float
            Smoothing factor to avoid 0 division.
        suppress_bkg: bool
            Factor that signals whether to suppress the background and enhance the foreground
        """
        self.alpha = alpha
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.eps = eps
        self.suppress_bkg = suppress_bkg

    def forward(self,
                y_pred: Tensor,
                y_true: Tensor,
                weights: Tensor,
                weights_list: Tensor):
        """
        Forward method

        Parameters
        ----------
        y_pred: tensor-like of shape (D, C, H, W)
            Prediction logits
        y_true: tensor-like of shape (D, H, W)
            Ground truth
        weights: Tensor-like of shape: (D, C, H, W)
            Class weights
        weights_list: Tensor-like of shape (, 79)
            Class weights list
        """
        # Compute the one-hot-encoded version of the ground truth tensor
        y_true_encoded = get_one_hot_encoded(y_true, y_pred.shape[1])

        # Compute tp (True Positives), fp (False Positives) and fn (False Negatives)
        tp = torch.sum(y_pred * y_true_encoded, dim=(0, 2, 3))
        fp = torch.sum((1 - y_true_encoded) * y_pred, dim=(0, 2, 3))
        fn = torch.sum(y_true_encoded * (1 - y_pred), dim=(0, 2, 3))

        # Compute the Trevsky indexes (TI)
        tversky_idxs = tp / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.eps)

        if self.suppress_bkg:
            bkg_tversky = (1 - tversky_idxs[0]).unsqueeze(dim=0)
            # print('Bkg tverski loss: ' + str(torch.mean(bkg_tversky)))
            fg_tversky = (1 - tversky_idxs[1:]) * (1 - tversky_idxs[1:]) ** (-self.gamma)
            # print('Fg tverski loss: ' + str(torch.mean(fg_tversky)))
            focal_tversky_loss = torch.cat(([bkg_tversky, fg_tversky]), dim=0)
        else:
            # Compute the Focal Tversky Loss per class
            focal_tversky_loss = (1 - tversky_idxs) ** self.gamma

        # Return the average loss
        return torch.mean(focal_tversky_loss)


class Unified_CatFocal_FocalTversky(nn.Module):
    """
    Implements the asymmetric unified focal loss from:
    https://www.sciencedirect.com/science/article/pii/S0895611121001750
    """

    def __init__(self,
                 alpha: float = 0.6,
                 lmbd: float = 0.5,
                 gamma: float = 0.5):
        """
        Constructor

        Parameters
        ----------
        alpha: Float
            Places a greater penalty on false negatives, rather on false positives
        lmbd: float
            Controls the weight given to the asymmetric Categorical Focal loss and the asymmetric Focal Tversky
        gamma:
            Controls both the background suppression and the foreground enhancement
        """
        super().__init__()
        self.alpha = alpha
        self.lmbd = lmbd
        self.gamma = gamma
        self.categorical_focal = CategoricalFocalLoss(alpha=alpha,
                                                      gamma=1/gamma,
                                                      suppress_bkg=False)
        self.focal_tverski = FocalTverskyLoss(alpha=alpha,
                                              gamma=gamma,
                                              suppress_bkg=True)

    def forward(self,
                y_pred: Tensor,
                y_true: Tensor,
                weights: Tensor,
                weights_list: Tensor):
        """
        Forward method

        Parameters
        ----------
        y_pred: Tensor-like of shape: (D, C, H, W)
            Prediction logits
        y_true: Tensor-like of shape: (D, H, W)
            Labels
        weights: Tensor-like of shape: (D, C, H, W)
            Class weights
        weights_list: Tensor-like of shape (, 79)
            Class weights list
        """
        y_pred = torch.softmax(input=y_pred, dim=1)

        cat_focal_loss = self.categorical_focal(y_pred=y_pred,
                                                y_true=y_true,
                                                weights=weights,
                                                weights_list=weights_list)
        tverski_loss = self.focal_tverski(y_pred=y_pred,
                                          y_true=y_true,
                                          weights=weights,
                                          weights_list=weights_list)

        return self.lmbd * cat_focal_loss + (1 - self.lmbd) * tverski_loss
