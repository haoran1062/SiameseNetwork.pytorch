# encoding:utf-8

class Config(object):
    
    backbone_type = 'resnet18'
    input_size = 224

    feature_extract = False
    use_pre_train = False
    resume_from_path = '/data/haoran/t/res18_224_0807/best.pth'
    resume_epoch = 0
    epoch_num = 500
    model_bpath = '/data/haoran/t/res18_224_0807'
    id_name_txt = model_bpath + '/id.txt'

    gpu_ids = [0]
    batch_size = 1
    out_features = 1000
    dataLoader_util = 'jpeg4py' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 4
    def __init__(self):
        super(Config, self).__init__()
        