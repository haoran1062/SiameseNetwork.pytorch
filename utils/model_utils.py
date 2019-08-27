#encoding:utf-8
import torchvision, torch, sys, time, math
from torchvision import datasets, models, transforms

import torch.nn as nn
sys.path.insert(0, sys.path[0] + '/backbones')
from backbones.OriginDenseNet import densenet121
from backbones.seResnet import se_resnext50_32x4d
from backbones.seResnet_features import se_resnext50_32x4d_features

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, input_size=224):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    elif model_name == "resnext50":

        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224
    
    elif model_name == 'se-resnext50':

        model_ft = se_resnext50_32x4d()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features * 100
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        # input_size = 224
    
    elif model_name == 'se-resnext50_features':
        model_ft = se_resnext50_32x4d_features()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = int( pow((input_size/32 + 7 - 1), 2) * 2048 )
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    elif model_name == "resnext101":

        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        # input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        # input_size = 224
    
    elif model_name == 'mobilenet':
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes),
        )

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        # input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        # model_ft = models.densenet121(pretrained=use_pretrained)
        model_ft = densenet121(pretrained=use_pretrained, num_classes=num_classes)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.originclassifier.in_features
        # print(num_ftrs)
        model_ft.originclassifier = nn.Linear(num_ftrs, num_classes) 
        # input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

def load_model(model_path, feature_extract):
    model_ft = torch.load(model_path)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier = model_ft.classifier # nn.Linear(num_ftrs, num_classes) 
    input_size = 224
    return model_ft, input_size
