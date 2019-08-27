#encoding:utf-8
import os, sys, time, numpy as np, cv2, copy
import torch

import torchvision.transforms as transforms
from PIL import Image

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def padding_resize(img, resize=224):
    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    h, w, c = img.shape 
    pad_img = np.zeros((resize, resize, 3), np.uint8)
    pad_img[:, :, :] = 255
    if h > w:
        nh = resize
        rs = 1.0 * h / nh 
        nw = int(w / rs )
        img = cv2.resize(img, (nw, nh))
        # print('h > w: ', img.shape)
        pw = int((resize - nw)/2)
        pad_img[:, pw : pw+nw] = img

    else:
        nw = resize
        rs = 1.0 * w / nw
        nh = int(h / rs )
        img = cv2.resize(img, (nw, nh))
        # print('h < w: ', img.shape)
        ph = int((resize - nh)/2)
        pad_img[ph : ph+nh, :] = img
    
    return pad_img

def origin_resize(img, resize=224):
    if isinstance(img, Image.Image):
        img = img.resize((resize, resize), Image.ANTIALIAS)
    else:
        img = cv2.resize(img, (resize, resize))

    return img

def tensor2img(in_tensor, normal=False):
    if normal:
        in_tensor = un_normal_trans(in_tensor[0].float())
    in_tensor = in_tensor.permute(1, 2 ,0)
    in_tensor = in_tensor.mul(255).byte()
    img = in_tensor.cpu().numpy()
    return img