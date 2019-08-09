# encoding:utf-8
import os, numpy as np, torch, cv2, jpeg4py, time, random
import torch.nn as nn

import torch.nn.functional as F
from torchvision import transforms
from backbones.resnet import resnet18
from backbones.seResnet import se_resnext50_32x4d


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.config = config
        self.backbone = self.load_backbone()
        
        
    def load_backbone(self):
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False
        if self.config.backbone_type == 'resnet18':
            model_ft = resnet18(pretrained=self.config.use_pre_train)
            set_parameter_requires_grad(model_ft, self.config.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.config.class_num)
        elif self.config.backbone_type == 'se-resnext50':
            pretrained = None
            if self.config.use_pre_train:
                pretrained = 'imagenet'
            model_ft = se_resnext50_32x4d(pretrained=pretrained, input_3x3=self.config.input_3x3)
            set_parameter_requires_grad(model_ft, self.config.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.config.class_num)
        
        else:
            print('backbone not supported!')
            exit()
        

        return model_ft

    def forward_once(self, x):
        # print(self.backbone(x))
        feature, output = self.backbone(x)
        return feature, output
    
    def forward(self, input1, input2=None):
        feature1, output1 = self.forward_once(input1)
        if input2 is not None:
            feature2, output2 = self.forward_once(input2)
            return feature1, output1, feature2, output2
        
        return feature1, output1
        

if __name__ == "__main__":
    from train_config import Config
    from torchsummary import summary


    
    Net = SiameseNetwork(Config)
    print(Net)
    