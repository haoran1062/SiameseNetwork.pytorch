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
from random import shuffle

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

def get_batch_imgs(data_trans, img_list, st_iter, batch_size):
    data_len = len(img_list)
    if batch_size > data_len:
        st_iter = 0
        batch_size = data_len
    elif st_iter + batch_size > len(img_list):
        st_iter = len(img_list) - batch_size

    batch_img = []
    for now_path in img_list[st_iter:st_iter+batch_size]:
        img = cv2.imread(now_path)
        t_img = data_transforms(img).unsqueeze(0)
        batch_img.append(t_img)
    st_iter += batch_size
    return torch.cat(batch_img, 0), st_iter



if __name__ == "__main__":
    
    # in_bpath = '/data/datasets/truth_data/classify_data/201906-201907_checked/all/'
    # in_bpath_list = ['/dev/shm/datasets/201906-0807_all/all/', '/dev/shm/datasets/UE4_cls_0905/all/', '/dev/shm/datasets/cls_0808-12/all/', '/dev/shm/datasets/cls_0813/all/', '/dev/shm/datasets/cls_0820/all/', '/dev/shm/datasets/cls_0821/all/', '/dev/shm/datasets/cls_0822/all/', '/dev/shm/datasets/cls_0819_zh/all/', '/dev/shm/datasets/cls_0814-16/all/', '/dev/shm/datasets/0826-0829_all/all/', '/dev/shm/datasets/cls_0903/all/', '/dev/shm/datasets/cls_0905/all/', '/dev/shm/datasets/cls_0906/all/', '/dev/shm/datasets/cls_1_0909/all/']
    in_bpath_list = glob('/dev/shm/datasets/*/all/')
    # in_bpath_list = ['/data/datasets/truth_data/classify_data/hardcase_classify_datasets/all/']
    # in_bpath_list = ['/data/datasets/truth_data/classify_data/201906-201907_checked/all/', '/data/datasets/sync_data/classify_sync_instances/UE4_cls_0808/UE4_cls_0808/all/']
    # in_bpath = '/data/datasets/sync_data/classify_sync_instances/UE4_cls_0805/UE4_cls_0805/all/'
    # file_list = glob(in_bpath + '*/*.jpg')
    # folder_list = os.listdir(in_bpath)
    # folder_list = []
    folder_set = set()
    for i in in_bpath_list:
        folder_set = folder_set | set(os.listdir(i))
    folder_list = list(folder_set)
    out_bpath = '%s/offline_features/'%(cfg.model_bpath)
    if not os.path.exists(out_bpath):
        os.makedirs(out_bpath)

    # net = SiameseNetwork(cfg)
    net = load_siamese_model(cfg)
    batch_size = 400
    # print(net)
    net.eval()
    id_name_map = get_id_map(cfg.id_name_txt)
    
    with torch.no_grad():
        
        for now_folder_name in tqdm(folder_list[:]):
            if os.path.exists(out_bpath + '%s.npy'%(now_folder_name)):
                continue
            now_iter = 0
            total_feature_list = []
            file_list = []
            for now_bpath in in_bpath_list:
                file_list += glob(now_bpath + now_folder_name + '/*.jpg')
            # file_list = glob(in_bpath + now_folder_name + '/*.jpg')
            f_len = len(file_list)
            shuffle(file_list)
            while now_iter < f_len:
                # print(now_iter, f_len)
                img_a, now_iter = get_batch_imgs(data_transforms, file_list, now_iter, batch_size)
                # print(img_a.shape)
                # img_b, now_iter = get_batch_imgs(data_transforms, file_list, now_iter, batch_size)

                ta = time.clock()
                # feature_a, softmax_a, feature_b, softmax_b = net(img_a, img_b)
                feature_a, softmax_a = net(img_a)
                tb = time.clock()
                # print(feature_a.shape)
                total_feature_list.append(feature_a.to('cpu'))
                # total_feature_list.append(feature_b.cpu())
                # print(len(total_feature_list))
                break
            if len(total_feature_list) > 0:
                total_features = torch.cat(total_feature_list, 0)
                np_total_features = total_features.numpy()
                now_save_path = out_bpath + '%s.npy'%(now_folder_name)
                np.save(now_save_path, np_total_features)
            else:
                print(now_folder_name + ' fuck!!!')
        

