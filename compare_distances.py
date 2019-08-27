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
from tqdm import tqdm

cfg = Config()

def load_siamese_model(config, device='cuda:0'):

    model_ft = SiameseNetwork(cfg)
    model_p = nn.DataParallel(model_ft.to(device), device_ids=config.gpu_ids)
    model_p.load_state_dict(torch.load(config.resume_from_path)['net'])
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
    
    # in_bpath = '/data/datasets/truth_data/classify_data/201906-201907_checked/all/'
    in_bpath = '/data/datasets/classify_data/truth_data/20190712_classify_train/'
    in_bpath = '/data/datasets/truth_data/classify_data/201906-201907_checked/all/'
    in_bpath = '/data/results/temp/classify_instance/201908/done/0806_cls_all/'
    offline_bpath = '/data/haoran/t/se_resnext50_384_0814/offline_features/'
    # in_bpath = '/data/datasets/sync_data/classify_sync_instances/UE4_cls_0805/UE4_cls_0805/all/'
    file_list = glob(in_bpath + '*/*.jpg')
    f_len = len(file_list)
    device = 'cuda:0'
    random.shuffle(file_list)

    # net = SiameseNetwork(cfg)
    net = load_siamese_model(cfg)
    # net.eval()
    # print(net)
    net.eval()
    id_name_map = get_id_map(cfg.id_name_txt)

    with torch.no_grad():
        for now_file in tqdm(file_list):
            
            label_a = get_label_from_path(now_file)
            img_a = cv2.imread(now_file)

            t_img_a = data_transforms(img_a).unsqueeze(0)
            if not os.path.exists('%s%d.npy'%(offline_bpath, label_a)):
                continue
            t_img_b = torch.from_numpy(np.load('%s%d.npy'%(offline_bpath, label_a))).to(device)

            # t_img_a = torch.cat([t_img_a]*len(t_img_b), 0)
            
            

            ta = time.clock()
            feature_a, softmax_a = net(t_img_a)

            pred = F.softmax(softmax_a, 1)[0]
            p_b_cls = id_name_map[pred.argmax().cpu().item()]
            p_b_conf = pred[pred.argmax()].cpu().item()
            
            feature_a = torch.cat([feature_a]*len(t_img_b), 0)
            tb = time.clock()
            print(tb - ta)
            # print(softmax_a.shape, softmax_b.shape)

            distance = F.pairwise_distance(feature_a, t_img_b, p=2).detach().to('cpu').numpy()[0]
            cosin_dist = F.cosine_similarity(feature_a, t_img_b, dim=-1).to('cpu').numpy()[0]
            print('a img label : %d\tpred : %s\t confidence : %.2f\t distance : %.2f\t cosine distance: %2f'%(label_a, p_b_cls, p_b_conf, distance, cosin_dist))
            
            cv2.imshow('a img', img_a)
            # cv2.imshow('b img', img_b)
            if cv2.waitKey(10000)&0xFF == ord('q'):
                break

        

