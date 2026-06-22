"""Loss Function Definitions"""

import torch
import torch.nn as nn
import torch.nn.functional as Functional

ALPHA = 1
GAMMA = 2

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice



class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA):
        # flatten the label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        BCE = Functional.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        inputs_proba = torch.sigmoid(inputs)
        BCE_EXP = torch.exp(-BCE)

        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss



class NCCLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(NCCLoss, self).__init__()

    def forward(self, inputs, targets):
        # flatten the label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        inputs_mean = torch.mean(inputs)
        targets_mean = torch.mean(targets)

        numerator = torch.sum((inputs - inputs_mean) * (targets - targets_mean))
        denominator = torch.sqrt(torch.sum((inputs - inputs_mean) ** 2) * torch.sum((targets - targets_mean) ** 2))

        ncc_loss = 1 - (numerator / denominator)

        return ncc_loss



class NCCLossSimple(nn.Module):
    """Simple implementation for Normalised Cross Correlation that can be
    minimised with upper-bound of alpha and lower-bound of 0.
    """

    def __init__(self, alpha=1):
        super(NCCLossSimple, self).__init__()
        self.NCC = None  # -1(very dissimilar) to 1(very similar)
        self.alpha = alpha

    def forward(self, y, yp):
        EPSILON = 1e-6

        y_ = y - torch.mean(y)
        yp_ = yp - torch.mean(yp)

        self.NCC = torch.sum(y_ * yp_) / (((torch.sum(y_ ** 2)) * torch.sum(yp_ ** 2) + EPSILON) ** 0.5)
        error = self.alpha * (1 - self.NCC)

        return error