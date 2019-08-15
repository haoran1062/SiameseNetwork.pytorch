# encoding:utf-8

class Config(object):
    
    backbone_type = 'se-resnext50'
    input_size = 224
    train_datasets_bpath = ['/dev/shm/datasets/201906-0807_all/train', '/dev/shm/datasets/UE4_cls_0814/train', '/dev/shm/datasets/cls_0808-12/train', '/dev/shm/datasets/cls_0813/train']
    test_datasets_bpath = ['/dev/shm/datasets/201906-0807_all/val', '/dev/shm/datasets/UE4_cls_0814/val', '/dev/shm/datasets/cls_0808-12/val', '/dev/shm/datasets/cls_0813/val']

    # train_datasets_bpath = ['/data/datasets/truth_data/classify_data/201907_checked/all']
    # test_datasets_bpath = ['/data/datasets/truth_data/classify_data/201907_checked/all']
    
    feature_extract = False
    use_pre_train = False
    input_3x3 = True
    resume_from_path = None
    fp16_using = True
    resume_epoch = 0
    epoch_num = 12
    model_bpath = '/data/train_models/classify_models/siamese_models/se_resnext50_%d_0814'%(input_size)
    id_name_txt = model_bpath + '/id.txt'
    log_name = 'train.log'
    vis_log = model_bpath + '/vis.log'

    gpu_ids = [0, 1, 2]
    batch_size = 128
    class_num = 1000
    dataLoader_util = 'cv2' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 8
    def __init__(self):
        super(Config, self).__init__()
        
