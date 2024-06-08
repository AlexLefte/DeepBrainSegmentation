import torch
from torch import Tensor, nn
import src.data.data_utils as du
import numpy as np

from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


def get_loss_fn(cfg: dict):
    """
    Returns the loss function

    Parameters
    ----------
    cfg: dict
        Configuration dictionary
    """
    # Initialize the loss function
    loss_fn = None

    # Get the loss type
    loss_type = cfg['loss_function']

    # Choose the loss type accordingly
    if loss_type == 'dice_loss_&_cross_entropy':
        loss_fn = CombinedLoss()
    elif loss_type == 'unified_focal_loss':
        loss_fn = CombinedFocalLoss(gamma=cfg['loss_gamma'])

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
        ce = nn.functional.cross_entropy(input=y_pred,
                                         target=y_true)
        weighted_ce = torch.mean(torch.mul(ce, weights))
        return weighted_ce


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
        ce = nn.functional.cross_entropy(input=y_pred,
                                         target=y_true.long(),
                                         reduction='none')
        cross_entropy_loss = torch.mean(torch.mul(ce, weights))
        # cross_entropy_loss = nn.functional.cross_entropy(input=y_pred,
        #                                                  target=y_true.long(),
        #                                                  weight=weights_list)

        dice_loss = self.dice_loss(y_pred=y_pred,
                                   y_true=y_true,
                                   weights=None)

        # dice_loss = self.dice_loss_2(y_pred=torch.nn.functional.softmax(y_predict, dim=1),
        #                                y_true=y,
        #                                weights=weights)

        # return cross_entropy_loss + dice_loss, cross_entropy_loss, dice_loss
        return cross_entropy_loss + dice_loss


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
    Implements the focal loss for handling class imbalance in classification tasks.
    Reference: https://www.sciencedirect.com/science/article/pii/S0895611121001750

    The focal loss focuses on hard-to-classify examples by down-weighting the loss for well-classified examples and up-weighting for those that are misclassified.
    """
    def __init__(self,
                 gamma: float = 2,
                 device: str = 'cpu',
                 suppress_bkg: bool = False,
                 eps: float = 1e-6):
        """
        Constructor

        Parameters
        ----------
        gamma: float or tensor-like of shape (C)
            The focusing parameter that reduces the loss contribution from easy examples (background)
        device: str
            The device on which the training is performed, e.g., 'cpu' or 'cuda'.
        suppress_bkg: bool
            A flag indicating whether to apply additional suppression on the background class
        eps: float
            Small constant to avoid division by zero.
        """
        super(CategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.device = device
        self.suppress_bkg = suppress_bkg
        self.eps = eps

    def forward(self,
                y_pred: Tensor,
                y_true: Tensor,
                alpha: Tensor):
        """
        Forward method

        Parameters
        ----------
        y_pred: Tensor
            Predicted logits, shape: (D, C, H, W).
        y_true: Tensor
            Ground truth labels, shape: (D, H, W).
        alpha: Tensor
            Class weights mask, shape: (D, C, H, W).
        """
        # Get the number of classes from the prediction logits
        c = y_pred.shape[1]

        # Compute the one-hot encoded version of the ground truth tensor
        y_true_encoded = get_one_hot_encoded(t=y_true,
                                             classes_num=c)

        # Compute the cross_entropy loss
        ce = -y_true_encoded * torch.log(y_pred + self.eps)

        if self.suppress_bkg:
            # Suppress the background loss component
            bkg_probs = y_pred[:, 0, :, :]
            bkg_ce = ce[:, 0, :, :]
            bkg_loss = torch.pow(1 - bkg_probs, self.gamma) * bkg_ce
            bkg_loss = bkg_loss.unsqueeze(dim=1)

            # Concatenate back the background and the foreground classes
            fg_ce = ce[:, 1:, :, :]
            focal_loss = torch.cat(([bkg_loss, fg_ce]), dim=1)
        else:
            # Compute the modulating factor
            modulating_factor = (1 - y_pred) ** self.gamma

            # Compute the categorical focal loss
            focal_loss = modulating_factor * ce

        # Multiply by class weights, sum across classes, and then average.
        focal_loss = torch.mean(alpha * torch.sum(focal_loss, dim=1))
        return focal_loss


class FocalTverskyLoss(nn.Module):
    """
    Implements the focal Tversky loss function.
    Reference: https://www.sciencedirect.com/science/article/pii/S0895611121001750

    The focal Tversky loss is designed to handle imbalanced datasets by focusing more on hard-to-classify examples and
    penalizing false negatives more than false positives for foreground classes.
    """
    def __init__(self,
                 gamma: float = 3 / 4,
                 device: str = 'cpu',
                 eps: float = 0.000001,
                 suppress_bkg: bool = False):
        """
        Constructor

        Parameters
        ----------
        gamma: float or tensor-like of shape (C)
            Rare class enhancement exponent
        device: str
            Training device (e.g., 'cpu' or 'cuda').
        eps: float
            Small constant to avoid division by zero.
        suppress_bkg: bool
            Flag indicating whether to enhance the focus on foreground classes.
        """
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.eps = eps
        self.enhance_fg = suppress_bkg

    def forward(self,
                y_pred: Tensor,
                y_true: Tensor,
                alpha: Tensor):
        """
        Forward method

        Parameters
        ----------
        y_pred: Tensor
            Predicted logits, shape: (D, C, H, W).
        y_true: Tensor
            Ground truth labels, shape: (D, H, W).
        alpha: Tensor
            Class weights list, shape: (, C).
        """
        # Compute the one-hot-encoded version of the ground truth tensor => Tensor with shape (D, C, H, W)
        y_true_encoded = get_one_hot_encoded(y_true, y_pred.shape[1])

        # Compute TP (True Positives), FP (False Positives) and FN (False Negatives)
        tp = torch.sum(y_pred * y_true_encoded, dim=(0, 2, 3))
        fp = torch.sum((1 - y_true_encoded) * y_pred, dim=(0, 2, 3))
        fn = torch.sum(y_true_encoded * (1 - y_pred), dim=(0, 2, 3))

        # Compute the Tversky indexes (TI)
        tversky_idxs = tp / (tp + alpha * fn + (1 - alpha) * fp + self.eps)

        # Check if we need to enhance foreground classes
        if self.enhance_fg:
            bkg_tversky = (1 - tversky_idxs[0]).unsqueeze(dim=0)

            # Apply the enhancement to the foreground classes
            fg_tversky = (1 - tversky_idxs[1:]) * ((1 - tversky_idxs[1:]) ** (-self.gamma))

            # Concatenate the background and foreground values back together
            focal_tversky_loss = torch.cat(([bkg_tversky, fg_tversky]), dim=0)
        else:
            # Compute the Focal Tversky Loss per class
            focal_tversky_loss = (1 - tversky_idxs) ** self.gamma

        # Return the mean focal Tversky loss across all classes
        return torch.mean(focal_tversky_loss)


class CombinedFocalLoss(nn.Module):
    """
    Implements the asymmetric unified focal loss from:
    https://www.sciencedirect.com/science/article/pii/S0895611121001750

    This class combines two types of focal losses: Categorical Focal Loss and Focal Tversky Loss.
    It is designed to handle imbalanced datasets by putting more focus on hard-to-classify examples.
    """
    def __init__(self,
                 alpha: float = 0.6,
                 lmbd: float = 0.5,
                 gamma: float = 0.5):
        """
        Constructor

        Parameters
        ----------
        alpha : float
            Penalty parameter that gives higher weight to false negatives compared to false positives.
        lmbd : float
            Weight balancing parameter between the asymmetric Categorical Focal loss and the Focal Tversky loss.
        gamma : float
            Parameter controlling the suppression of background classes and enhancement of foreground classes.
        """
        super().__init__()
        self.alpha = alpha
        self.lmbd = lmbd
        self.gamma = gamma

        # Initialize the Categorical Focal Loss with specified alpha and gamma parameters
        self.categorical_focal = CategoricalFocalLoss(gamma=1/gamma,
                                                      suppress_bkg=True)

        # Initialize the Focal Tversky Loss with specified alpha and gamma parameters
        self.focal_tverski = FocalTverskyLoss(gamma=gamma,
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
        y_pred: Tensor
            Predicted logits, shape: (D, C, H, W).
        y_true: Tensor
            Ground truth labels, shape: (D, H, W).
        weights: Tensor
            Class weights, shape: (D, C, H, W).
        weights_list: Tensor
            Class weights list, shape: (, 79).

        Returns
        -------
        Combined loss : float
            Sum of Categorical Focal Loss and Focal Tversky Loss.
        """
        # Apply softmax to the predicted logits to obtain probabilities
        y_pred = torch.softmax(input=y_pred, dim=1)

        # Compute the Categorical Focal Loss
        cat_focal_loss = self.categorical_focal(y_pred=y_pred,
                                                y_true=y_true,
                                                alpha=weights)

        # Compute the Focal Tversky Loss
        tverski_loss = self.focal_tverski(y_pred=y_pred,
                                          y_true=y_true,
                                          alpha=weights_list)

        # Return the sum of the two losses
        return cat_focal_loss + tverski_loss


class TestDiceLoss(_Loss):
    """Calculate Dice Loss.

    Methods
    -------
    forward
        Calulate the DiceLoss
    """
    def forward(
            self,
            output: Tensor,
            target: Tensor,
            weights=None,
            ignore_index=None
    ) -> float:
        """Calulate the DiceLoss.

        Parameters
        ----------
        output : Tensor
            N x C x H x W Variable
        target : Tensor
            N x C x W LongTensor with starting class at 0
        weights : Optional[int]
            C FloatTensor with class wise weights(Default value = None)
        ignore_index : Optional[int]
            ignore label with index x in the loss calculation (Default value = None)

        Returns
        -------
        float
            Calculated Diceloss

        """
        eps = 0.001

        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0

        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0

        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (
                1 - (numerator / denominator)
        )  # Channel-wise weights

        return loss_per_channel.sum() / output.size(1)


class TestCrossEntropy2D(nn.Module):
    """2D Cross-entropy loss implemented as negative log likelihood.

    Attributes
    ----------
    nll_loss
        calculated cross-entropy loss

    Methods
    -------
    forward
        returns calculated cross entropy
    """

    def __init__(self, weight=None, reduction: str = "none"):
        """Construct CrossEntropy2D object.

        Parameters
        ----------
        weight : Optional[Tensor]
            a manual rescaling weight given to each class. If given, has to be a Tensor of size `C`. Defaults to None
        reduction : str
            Specifies the reduction to apply to the output, as in nn.CrossEntropyLoss. Defaults to 'None'

        """
        super(TestCrossEntropy2D, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        print(
            f"Initialized {self.__class__.__name__} with weight: {weight} and reduction: {reduction}"
        )

    def forward(self, inputs, targets):
        """Feedforward."""
        return self.nll_loss(inputs, targets)


class TestCombinedLoss(nn.Module):
    """For CrossEntropy the input has to be a long tensor.

    Attributes
    ----------
    cross_entropy_loss
        Results of cross entropy loss
    dice_loss
        Results of dice loss
    weight_dice
        Weight for dice loss
    weight_ce
        Weight for float
    """

    def __init__(self, weight_dice=1, weight_ce=1):
        """Construct CobinedLoss object.

        Parameters
        ----------
        weight_dice : Real
            Weight for dice loss. Defaults to 1
        weight_ce : Real
            Weight for cross entropy loss. Defaults to 1

        """
        super(TestCombinedLoss, self).__init__()
        self.cross_entropy_loss = TestCrossEntropy2D()
        self.dice_loss = TestDiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(
            self,
            y_pred: Tensor,
            y_true: Tensor,
            weights: Tensor,
            weights_list: Tensor):
        # Typecast to long tensor --> labels are bytes initially (uint8),
        # index operations require LongTensor in pytorch
        y_true = y_true.type(torch.LongTensor)
        # Due to typecasting above, target needs to be shifted to gpu again
        if y_pred.is_cuda:
            y_true = y_true.cuda()

        input_soft = F.softmax(y_pred, dim=1)  # Along Class Dimension
        dice_val = torch.mean(self.dice_loss(input_soft, y_true))
        ce_val = torch.mean(
            torch.mul(self.cross_entropy_loss.forward(y_pred, y_true), weights)
        )
        total_loss = torch.add(
            torch.mul(dice_val, self.weight_dice), torch.mul(ce_val, self.weight_ce)
        )

        return total_loss
