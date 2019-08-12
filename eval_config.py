# encoding:utf-8

class Config(object):
    
    backbone_type = 'se-resnext50'
    input_size = 224

    feature_extract = False
    use_pre_train = False
    input_3x3 = True
    resume_from_path = '/data/haoran/t/se_resnext50_224_0810/best.pth'
    resume_epoch = 0
    epoch_num = 500
    model_bpath = '/data/haoran/t/se_resnext50_224_0810'
    id_name_txt = model_bpath + '/id.txt'

    gpu_ids = [0]
    batch_size = 1
    class_num = 1000
    dataLoader_util = 'cv2' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 4
    def __init__(self):
        super(Config, self).__init__()
        