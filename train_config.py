# encoding:utf-8

class Config(object):
    
    backbone_type = 'se-resnext50'
    input_size = 224
    train_datasets_bpath = ['/data/datasets/classify_data/checkout_cls_data/truth_data/201906-0807_train_test_data/train']# , '/data/datasets/classify_data/checkout_cls_data/sync_data/UE4_cls_0808/train']
    test_datasets_bpath = ['/data/datasets/classify_data/checkout_cls_data/truth_data/201906-0807_train_test_data/val']# , '/data/datasets/classify_data/checkout_cls_data/sync_data/UE4_cls_0808/val']
    
    feature_extract = False
    use_pre_train = False
    input_3x3 = True
    resume_from_path = None
    resume_epoch = 0
    epoch_num = 10
    model_bpath = '/data/train_models/siamese_models/se_resnext50_0810'
    id_name_txt = model_bpath + '/id.txt'
    log_name = 'train.log'
    vis_log = model_bpath + '/vis.log'

    gpu_ids = [0]
    batch_size = 28
    class_num = 1000
    dataLoader_util = 'cv2' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 8
    def __init__(self):
        super(Config, self).__init__()
        