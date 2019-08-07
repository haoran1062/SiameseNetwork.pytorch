# encoding:utf-8
import torch 
import torch.nn.functional as F

import torch.nn as nn
CELoss = nn.CrossEntropyLoss()
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label1, label2, sim_labels):
        softmax1 = CELoss(output1, label1)
        softmax2 = CELoss(output2, label2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-sim_labels) * torch.pow(euclidean_distance, 2) + (sim_labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 

        return loss_contrastive, softmax1, softmax2