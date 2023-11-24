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

        # Get the number of classes:
        c = y_pred.shape[1]

        # Substitute the value at the index associated with the correct class with 1.
        # Iterate over each class
        for class_index in range(c):
            # Identify the indices where the target tensor has the current class
            # class_indices = (y_true == class_index).nonzero(as_tuple=False)
            class_indices = (y_true == class_index)

            # Set the corresponding values in the binary prediction tensor to 1 for the current class
            # y_true_encoded[class_indices[:, 0], class_index, class_indices[:, 1], class_indices[:, 2]] = 1
            # y_true_encoded[class_indices[:, 0], class_index, class_indices[:, 1]] = 1
            y_true_encoded[:, class_index, :, :][class_indices] = 1

        # Check the weights
        if weights is None:
            weights = torch.ones(c)

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
    def forward(y_pred,
                y_true,
                weights):
        loss = nn.functional.cross_entropy(input=y_pred,
                                           target=y_true,
                                           weight=weights)
        return loss


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
        self.dice_loss_2 = DiceLoss2()
        # self.cross_entropy = WeightedCELoss()
        # self.weights = weights

    def forward(self,
                y_predict: Tensor,
                y: Tensor,
                weights: Tensor,
                weights_dict: Tensor):
        """
        Computes the composite loss
        """
        if y_predict.is_cuda:
            y = y.to('cuda')

        # cross_entropy_loss = self.cross_entropy(y_pred=y_predict,
        #                                         y_true=y,
        #                                         weights=weights)
        # See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        cross_entropy_loss = nn.functional.cross_entropy(input=y_predict,
                                                         target=y.long(),
                                                         weight=weights_dict)

        dice_loss = self.dice_loss(y_pred=torch.nn.functional.softmax(y_predict, dim=1),
                                   y_true=y,
                                   weights=weights_dict)

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
            class_indices = (y_true == class_index).nonzero(as_tuple=False)

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