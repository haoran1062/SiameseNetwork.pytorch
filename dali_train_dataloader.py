# encoding:utf-8
import os, cv2, torch, time, numpy as np
from glob import glob 

from tqdm import tqdm
from random import shuffle, randint
import nvidia.dali.ops as ops 
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from train_config import Config
from nvidia.dali.plugin.pytorch import DALIGenericIterator


# root_dir = '/dev/shm/all'


class CustomSiameseIterator(object):
    def __init__(self, batch_size, root_dir, same_cate_prob=0.5, random_shuffle=False):
        self.images_dir = root_dir
        self.batch_size = batch_size
        self.same_cate_thresh = same_cate_prob
        self.files = self.get_data_list(self.images_dir)
        self.cls_path_map = self.get_cls_pathlist_map()
        if random_shuffle:
            shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def get_data_list(self, base_data_path):
        cls_file_list = []
        if isinstance(base_data_path, list):
            for i in base_data_path:
                cls_file_list = cls_file_list + glob(i + '/*/*.jpg')
        elif '|' in base_data_path:
            base_data_path = base_data_path.split('|')
            for i in base_data_path:
                cls_file_list = cls_file_list + glob(i + '/*/*.jpg')
        else:
            cls_file_list = glob(base_data_path + '/*/*.jpg')

        return cls_file_list

    def get_label_from_path(self, in_path):
        return int(in_path.split('/')[-2])

    def get_cls_pathlist_map(self):
        '''
            speedup choice special category sample


        '''
        ans_map = {}
        print('build choice map...')
        for now_path in tqdm(self.files):
            now_cls = self.get_label_from_path(now_path)
            if now_cls not in ans_map.keys():
                ans_map[now_cls] = [now_path]
            else:
                ans_map[now_cls].append(now_path)
        return ans_map

    def __next__(self):
        target_imgs = []
        target_labels = []
        cmp_imgs = []
        cmp_labels = []
        siamese_labels = []
        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i], self.get_label_from_path(self.files[self.i])
            f = open(jpeg_filename, 'rb')
            target_imgs.append(np.frombuffer(f.read(), dtype = np.uint8))
            target_labels.append(np.array([label], dtype = np.int64))
            if randint(0, 100) < self.same_cate_thresh:
                jpeg_filename = self.cls_path_map[label][randint(0, len(self.cls_path_map[label])-1)]
                f = open(jpeg_filename, 'rb')
                cmp_imgs.append(np.frombuffer(f.read(), dtype = np.uint8))
                cmp_labels.append(np.array([label], dtype = np.int64))
            else:
                ch_iter = randint(0, len(self.files)-1)
                jpeg_filename, label = self.files[ch_iter], self.get_label_from_path(self.files[ch_iter])
                f = open(jpeg_filename, 'rb')
                cmp_imgs.append(np.frombuffer(f.read(), dtype = np.uint8))
                cmp_labels.append(np.array([label], dtype = np.int64))
            siamese_labels.append(np.array([int(target_labels[-1] != cmp_labels[-1])],dtype=np.float32))
            self.i = (self.i + 1) % self.n
        return (target_imgs, target_labels, cmp_imgs, cmp_labels, siamese_labels)

    next = __next__

# csi = CustomSiameseIterator(batch_size, root_dir, same_cate_prob=same_cate_prob, random_shuffle=random_shuffle)
# iterator = iter(csi)

class SiamesePipeline(Pipeline):

    def __init__(self, cfg, root_dir, batch_size, num_threads, device_id=0):
        super(SiamesePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input_target_images = ops.ExternalSource()
        self.input_target_labels = ops.ExternalSource()
        self.input_cmp_images = ops.ExternalSource()
        self.input_cmp_labels = ops.ExternalSource()
        self.input_siamese_labels = ops.ExternalSource()
        self.dataset = CustomSiameseIterator(batch_size, root_dir, cfg.same_cate_prob, cfg.random_shuffle)
        self.iterator = iter(self.dataset)

        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.BGR)
        self.resize_op = ops.Resize(resize_longer=cfg.input_size)

        self.paste_ratio = ops.Uniform(range=(9, 12))
        self.paste = ops.Paste(device="gpu", fill_value=(255,255,255))
        # self.crop = ops.Crop(device ='gpu', crop=[224, 224])
        output_dtype = types.FLOAT16 if cfg.fp16_using else types.FLOAT
        # output_dtype = types.FLOAT
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            crop=(cfg.input_size, cfg.input_size),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=0,
            output_dtype=output_dtype,
            output_layout=types.NCHW,
            pad_output=False)

    def define_graph(self):
        self.target_jpegs = self.input_target_images()
        self.target_labels = self.input_target_labels()
        target_images = self.decode(self.target_jpegs)
        target_images = self.resize_op(target_images)

        ratio = self.paste_ratio()

        target_images = self.paste(target_images.gpu(), ratio = ratio)
        target_images = self.normalize(target_images)

        self.cmp_jpegs = self.input_cmp_images()
        self.cmp_labels = self.input_cmp_labels()
        cmp_images = self.decode(self.cmp_jpegs)
        cmp_images = self.resize_op(cmp_images)

        ratio = self.paste_ratio()

        cmp_images = self.paste(cmp_images.gpu(), ratio = ratio)
        cmp_images = self.normalize(cmp_images)

        self.siamese_labels =  self.input_siamese_labels()

        return (target_images, self.target_labels.gpu(), cmp_images, self.cmp_labels.gpu(), self.siamese_labels.gpu())
    
    def iter_setup(self):
        (target_imgs, target_labels, cmp_imgs, cmp_labels, siamese_labels) = self.iterator.next()
        self.feed_input(self.target_jpegs, target_imgs)
        self.feed_input(self.target_labels, target_labels)
        self.feed_input(self.cmp_jpegs, cmp_imgs)
        self.feed_input(self.cmp_labels, cmp_labels)
        self.feed_input(self.siamese_labels, siamese_labels)

def get_dataloader(pipiter, pipline, output_map, cfg, is_train):

    if is_train:
        root_dir = cfg.train_datasets_bpath
    else:
        root_dir = cfg.test_datasets_bpath
    now_pipline = pipline(cfg, root_dir, cfg.batch_size, cfg.worker_numbers)
    dataloader = DALIGenericIterator(now_pipline, output_map, now_pipline.dataset.n / len(cfg.gpu_ids), auto_reset=True)
    return dataloader

if __name__ == "__main__":
    # batch_size = 256
    # root_dir = '/data/datasets/truth_data/classify_data/201906-201907_checked/all'
    # root_dir = '/dev/shm/all'
    # file_list = 'train.txt'
    cfg = Config()
    # thread_number = cfg.worker_numbers
    # batch_size = cfg.batch_size
    # root_dir = cfg.train_datasets_bpath
    # # device_ids = cfg.gpu_ids
    # same_cate_prob = cfg.same_cate_prob
    # random_shuffle = cfg.random_shuffle
    # device_id = 0
    out_map = ['target_jpegs', 'target_labels', 'cmp_jpegs', 'cmp_labels', 'siamese_labels']
    test_n = 10

    dataloader = get_dataloader(CustomSiameseIterator, SiamesePipeline, out_map, cfg, True)

    # pipe = SiamesePipeline(batch_size, thread_number)
    # pipe.build()


    st = time.time()
    
    # for i in tqdm(range(test_n)):
    for it, data in enumerate(dataloader):
        img = data[0]['target_jpegs']
        label = data[0]['target_labels']
        img = data[0]['cmp_jpegs']
        label = data[0]['cmp_labels']
        s_label = data[0]['siamese_labels']

        print(img.shape, img.dtype, img.device, label.shape, label.dtype, label.device)
        # pipe_out = pipe.run()
        # img, label, img1, label1, s_label = pipe_out
        # np_label = np.array(label.as_tensor())
        # imgs = np.array(img.as_cpu().as_tensor())
        # np_label1 = np.array(label1.as_tensor())
        # imgs1 = np.array(img1.as_cpu().as_tensor())
        # siamese_l = s_label.as_tensor()
        # print(siamese_l)
        # print(imgs.shape)
        # print(img.as_cpu().at(1).shape)
        # cv2.imshow('img', imgs[0])
        # cv2.imshow('img1', imgs1[0])
        # if cv2.waitKey(10000)&0xFF == ord('q'):
            # break
        if it > test_n:
            break



        # print(imgs.shape)
    
    ed = time.time()
    print('cost %.2fs. load img speed: %.2f/s.'%(ed-st, (cfg.batch_size * test_n)/(ed-st)))
