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

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, use_focal_loss=False, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if use_focal_loss:
            self.CELoss = FocalLoss()
        else:
            self.CELoss = nn.CrossEntropyLoss()
            

    def forward(self, feature1, feature2, output1, output2, label1, label2, sim_labels):

        softmax1 = self.CELoss(output1, label1)
        softmax2 = self.CELoss(output2, label2)
        euclidean_distance = F.pairwise_distance(feature1, feature2)
        loss_contrastive = torch.mean((1-sim_labels) * torch.pow(euclidean_distance, 2) + (sim_labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 

        return loss_contrastive, softmax1, softmax2