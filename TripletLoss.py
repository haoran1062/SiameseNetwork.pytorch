# encoding:utf-8
import torch 
import torch.nn.functional as F

import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, use_focal_loss=False, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if use_focal_loss:
            self.CELoss = FocalLoss()
        else:
            self.CELoss = nn.CrossEntropyLoss()
            
    def forward(self, anchor, positive, negative, anc_output, pos_output, neg_output, anc_label, pos_label, neg_label, size_average=True):
        
        softmax1 = self.CELoss(anc_output, anc_label)
        softmax2 = self.CELoss(pos_output, pos_label)
        softmax3 = self.CELoss(neg_output, neg_label)
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        trip_loss = losses.mean() if size_average else losses.sum()

        return softmax1, softmax2, softmax3, trip_loss