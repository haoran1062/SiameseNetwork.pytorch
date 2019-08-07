# encoding:utf-8
import os, sys, numpy as np, random, time, cv2
import torch

import jpeg4py as jpeg
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
from glob import glob
ia.seed(random.randint(1, 10000))


class ClassifyDataset(data.Dataset):
    def __init__(self, base_data_path, train, transform, id_name_path,  device, little_train=False, read_mode='jpeg4py', input_size=224, C=2048, test_mode=False):
        print('data init')
        
        self.train = train
        self.base_data_path=base_data_path
        self.transform=transform
        self.fnames = []
        self.resize = input_size
        self.little_train = little_train
        self.id_name_path = id_name_path
        self.C = C
        self.read_mode = read_mode
        self.device = device
        self._test = test_mode

        self.fnames = self.get_data_list(base_data_path)
        self.num_samples = len(self.fnames)

        self.get_id_map()
    
    def get_id_list(self):
        id_set = set()
        if isinstance(self.base_data_path, list):
            for i in self.base_data_path:
                id_tl = os.listdir(i)
                for j in id_tl:
                    id_set.add(j)
        else:
            id_tl = os.listdir(self.base_data_path)
            for j in id_tl:
                id_set.add(j)
        return list(id_set)


    def get_id_map(self):
        self.id_name_map = {}
        self.name_id_map = {}
        if not os.path.exists(self.id_name_path):
            id_list = self.get_id_list()
            with open(self.id_name_path, 'w') as f:
                for it, cls_name in enumerate(id_list):
                    self.name_id_map[cls_name] = it
                    self.id_name_map[it] = cls_name
                    f.write(cls_name+'\n')
        else:
            with open(self.id_name_path, 'r') as f:
                itt = 0
                for line in f:
                    self.name_id_map[line.strip()] = itt
                    self.id_name_map[itt] = line.strip()
                    itt += 1

    def get_data_list(self, base_data_path):
        cls_file_list = []
        if isinstance(base_data_path, list):
            for i in base_data_path:
                cls_file_list = cls_file_list + glob(i + '/*/*.jpg')
        else:
            cls_file_list = glob(base_data_path + '/*/*.jpg')
        if self.little_train:
            return cls_file_list[:self.little_train]
        return cls_file_list
    
    def get_label_from_path(self, in_path):
        t_str = in_path.split('/')[-2]
        return self.name_id_map[t_str]
    
    def get_img_from_path(self, in_path):
        if self.read_mode == 'cv2':
            img = cv2.imread(in_path)
        elif self.read_mode == 'jpeg4py':
            img = jpeg.JPEG(in_path).decode()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def __getitem__(self,idx):
        
        fname = self.fnames[idx]
        if self._test:
            print(fname)
        
        img1 = self.get_img_from_path(fname)
        assert img1 is not None, print(fname)
        label1 = self.get_label_from_path(fname)
        img1 = self.transform(img1)

        should_get_same_class = random.randint(0,1) 
        # print(should_get_same_class)
        if should_get_same_class:
            while True:
                fname = self.fnames[random.randint(0, self.num_samples-1)]
                label2 = self.get_label_from_path(fname)
                if label1==label2:
                    break
        else:
            while True:
                fname = self.fnames[random.randint(0, self.num_samples-1)]
                label2 = self.get_label_from_path(fname)
                if label1!=label2:
                    break
        img2 = self.get_img_from_path(fname)
        assert img2 is not None, print(fname)
        img2 = self.transform(img2)
        label2 = self.get_label_from_path(fname)
        
        if self.train:
            # add data augument
            pass
            # seq_det = self.augmentation.to_deterministic()
            # img = seq_det.augment_images([img])[0]

        return img1, img2, label1, label2, torch.from_numpy(np.array([int(label1 != label2)],dtype=np.float32))

    def __len__(self):
        return self.num_samples

    
