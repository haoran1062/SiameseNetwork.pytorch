# encoding:utf-8
import os, numpy as np, torch, cv2, jpeg4py, time, random
import torch.nn as nn

import torch.nn.functional as F
from torchvision import transforms, models


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
            model_ft = models.resnet18(pretrained=self.config.use_pre_train)
            set_parameter_requires_grad(model_ft, self.config.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.config.out_features)
        else:
            print('backbone not supported!')
            exit()
        

        return model_ft

    def forward_once(self, x):
        output = self.backbone(x)
        return output
    
    def forward(self, input1, input2):
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            return output1, output2
        

if __name__ == "__main__":
    from train_config import Config
    from torchsummary import summary


    
    Net = SiameseNetwork(Config)
    print(Net)
    