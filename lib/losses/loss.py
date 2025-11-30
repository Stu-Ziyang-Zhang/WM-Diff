import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, pos_weight=None, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction=reduction)
    
    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        return F.binary_cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)


class DiffusionLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DiffusionLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predicted_noise, target_noise):
        return F.l1_loss(predicted_noise, target_noise, reduction=self.reduction)


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(torch.float32))
            if torch.cuda.is_available():
                nll_weight = nll_weight.cuda()
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        return self.nll_loss(inputs, targets)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs, 1)) ** self.gamma * F.log_softmax(inputs, 1), targets)
