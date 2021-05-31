import sys
import torch
import torch.nn.functional as F
from . import utils as utils

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    from : https://raw.githubusercontent.com/delijati\
            /pytorch-siamese/master/contrastive.py
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance

        if isinstance(x0, list):
            x0 = torch.cat(x0)
        if isinstance(x1, list):
            x1 = torch.cat(x1)
        y = y.float()
        diff = x0 - x1
        dist_sq = torch.sum(utils.safe_pow(diff, 2), 1)
        euclidean = torch.sqrt(dist_sq)

        mdist = self.margin - euclidean
        dist = torch.clamp(mdist, min=0.0)

        loss = y * dist_sq + (1 - y) * utils.safe_pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, y, euclidean


class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class HardTripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()



class DoubleMarginContrastiveLoss(torch.nn.Module):
    def __init__(self, l_m, u_m):
        super().__init__()
        self.l_m = l_m
        self.u_m = u_m

    def forward(self, x0, x1, y):
        diff = x0 - x1
        dist_sq = torch.sum(utils.safe_pow(diff, 2), 1)

        euclidean = torch.sqrt(dist_sq)

        ld = torch.clamp(euclidean - self.l_m, min = 0)
        ud = torch.clamp(self.u_b  - euclidean, min = 0)

        loss = y * utils.safe_pow(ld, 2) + (1 - y) * utils.safe_pow(ud, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, y, euclidean

def MaxElemConstrastiveLoss(ContrastiveLoss):

    def forward(self, x0, x1, y):

        d_list = []
        for e0, e1 in zip(x0, x1):
            e0_len = e0.shape[0]
            e1_len = e1.shape[0]

            e0_rep = e0.repeat(1, e1_len).view(e1_len * e0_len, -1)
            e1_rep = e1.repeat(e0_len, 1)

            d_list.append(F.pairwise_distance(e0_rep, e1_rep).min())

        euclidean = torch.cat(d_list)

        mdist = self.margin - euclidean
        dist = torch.clamp(mdist, min=0.0)
        loss = y * utils.safe_pow(euclidean, 2) + (1 - y) * utils.safe_pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, y, euclidean




class NegativeLogLikelihood(torch.nn.Module):

    def forward(self, x, y):

        lsmax = F.log_softmax(x, dim=1)
        loss = F.nll_loss(lsmax, y)

        return loss, y, lsmax



_LOSSES = {"contrastive" : ContrastiveLoss,
           "doublecontrastive" : DoubleMarginContrastiveLoss}

def get_loss(key):
    if key in _LOSSES:
        return _LOSSES[key]
    current_module = sys.modules[__name__]
    if key in current_module.__dict__:
        return getattr(current_module, key)
    else:
        return getattr(torch.nn, key)
