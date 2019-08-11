#encoding:utf-8
import os, sys, time, numpy as np, cv2, copy, argparse, random
from glob import glob 

import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from utils.data_utils import *
from utils.train_utils import *
from utils.visual import Visual
from torchsummary import summary
from dataLoader import ClassifyDataset
from SiameseNet import SiameseNetwork
from ContrastiveLoss import ContrastiveLoss
from eval_config import Config
import torch.nn.functional as F

cfg = Config()

def load_siamese_model(config, device='cuda:0'):

    model_ft = SiameseNetwork(cfg)
    model_p = nn.DataParallel(model_ft.to(device), device_ids=config.gpu_ids)
    model_p.load_state_dict(torch.load(config.resume_from_path))
    model_p.eval()
    return model_p

def get_label_from_path(in_path):
    t_str = in_path.split('/')[-2]
    return int(t_str)

def get_id_map(in_file):
    id_name_map = {}
    
    with open(in_file, 'r') as f:
        itt = 0
        for line in f:
            id_name_map[itt] = line.strip()
            itt += 1
    return id_name_map

data_transforms = transforms.Compose([
    transforms.Lambda(lambda img: padding_resize(img, resize=cfg.input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    
    in_bpath = '/data/datasets/classify_data/checkout_cls_data/truth_data/201906-0807_all/all/'
    # in_bpath = '/data/datasets/classify_data/truth_data/20190712_classify_train/'
    # in_bpath = '/data/datasets/sync_data/classify_sync_instances/UE4_cls_0805/UE4_cls_0805/all/'
    file_list = glob(in_bpath + '*/*.jpg')
    f_len = len(file_list)

    # net = SiameseNetwork(cfg)
    net = load_siamese_model(cfg)
    # print(net)
    net.eval()
    id_name_map = get_id_map(cfg.id_name_txt)

    with torch.no_grad():
        while True:
            a_img_file = file_list[random.randint(0, f_len - 1)]
            label_a = get_label_from_path(a_img_file)
            img_a = cv2.imread(a_img_file)

            if random.randint(0, 1):
                while True:
                    b_img_file = file_list[random.randint(0, f_len - 1)]
                    label_b = get_label_from_path(b_img_file)
                    if label_a==label_b:
                        break
            else:
                b_img_file = file_list[random.randint(0, f_len - 1)]
                label_b = get_label_from_path(b_img_file)

            img_b = cv2.imread(b_img_file)

            t_img_a = data_transforms(img_a).unsqueeze(0)
            t_img_b = data_transforms(img_b).unsqueeze(0)

            # t_img_a = torch.cat([t_img_a]*10, 0)
            # t_img_b = torch.cat([t_img_b]*10, 0)

            ta = time.clock()
            feature_a, softmax_a, feature_b, softmax_b = net(t_img_a, t_img_b)
            tb = time.clock()
            print(tb - ta)

            pred = F.softmax(softmax_a, 1)[0]
            # print(pred.shape)
            p_a_cls = id_name_map[pred.argmax().cpu().item()]
            p_a_conf = pred[pred.argmax()].cpu().item()

            pred = F.softmax(softmax_b, 1)[0]
            # print(pred.shape)
            p_b_cls = id_name_map[pred.argmax().cpu().item()]
            p_b_conf = pred[pred.argmax()].cpu().item()
            
            # print(softmax_a.shape, softmax_b.shape)

            distance = F.pairwise_distance(feature_a[0].unsqueeze(0), feature_b[0].unsqueeze(0)).detach().to('cpu').numpy()[0]
            # print('a img label : %d\t b img label : %d\t distance : %.2f'%(label_a, label_b, distance))
            print('a gt label/pred a label : %d/%s\tconfidence: %.2f\t b gt label/b img label : %d/%s\tconfidence: %.2f\tdistance : %.2f'%(label_a, p_a_cls, p_a_conf, label_b, p_b_cls, p_b_conf, distance))
            # print('a img pred : %d\t b img pred : %d'%(pred_a, pred_b))
            
            cv2.imshow('a img', img_a)
            cv2.imshow('b img', img_b)
            if cv2.waitKey(10000)&0xFF == ord('q'):
                break

        

