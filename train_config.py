# encoding:utf-8

class Config(object):
    
    backbone_type = 'se-resnext50'
    input_size = 224
    train_datasets_bpath = ['/data/datasets/truth_data/classify_data/201908/20190801_am_cls']
    test_datasets_bpath = ['/data/datasets/truth_data/classify_data/201908/20190801_am_cls']
    
    feature_extract = False
    use_pre_train = False
    input_3x3 = True
    resume_from_path = None
    resume_epoch = 0
    epoch_num = 24
    model_bpath = 'saved_models/se_resnext50_test'
    id_name_txt = model_bpath + '/id.txt'
    log_name = 'train.log'
    vis_log = model_bpath + '/vis.log'

    gpu_ids = [0]
    batch_size = 16
    class_num = 1000
    dataLoader_util = 'jpeg4py' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 4
    def __init__(self):
        super(Config, self).__init__()
        