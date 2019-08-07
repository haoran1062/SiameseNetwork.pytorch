# encoding:utf-8

class Config(object):
    
    backbone_type = 'resnet18'
    input_size = 224
    train_datasets_bpath = ['/data/datasets/truth_data/classify_data/201908/20190801_am_cls']
    test_datasets_bpath = ['/data/datasets/truth_data/classify_data/201908/20190801_am_cls']
    
    feature_extract = False
    use_pre_train = True
    resume_from_path = None
    resume_epoch = 0
    epoch_num = 500
    model_bpath = 'saved_models/resnet18_test'
    id_name_txt = model_bpath + '/id.txt'
    log_name = 'train.log'
    vis_log = model_bpath + '/vis.log'

    gpu_ids = [0]
    batch_size = 128
    out_features = 1000
    dataLoader_util = 'jpeg4py' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 4
    def __init__(self):
        super(Config, self).__init__()
        