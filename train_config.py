# encoding:utf-8

class Config(object):
    
    backbone_type = 'se-resnext50'
    # backbone_type = 'mobilenet'
    # backbone_type = 'shufflenet'
    input_size = 224
    # train_datasets_bpath = ['/dev/shm/datasets/201906-0807_all/train', '/dev/shm/datasets/UE4_cls_0814/train', '/dev/shm/datasets/cls_0808-12/train', '/dev/shm/datasets/cls_0813/train']
    # test_datasets_bpath = ['/dev/shm/datasets/201906-0807_all/val', '/dev/shm/datasets/UE4_cls_0814/val', '/dev/shm/datasets/cls_0808-12/val', '/dev/shm/datasets/cls_0813/val']

    train_datasets_bpath = ['/data/datasets/truth_data/classify_data/hardcase_classify_datasets/train']
    test_datasets_bpath = ['/data/datasets/truth_data/classify_data/hardcase_classify_datasets/val']
    
    feature_extract = False
    use_pre_train = False
    input_3x3 = False
    resume_from_path = None
    fp16_using = True
    resume_epoch = 0
    epoch_num = 24
    model_bpath = '/data/train_models/classify_models/siamese_models/%s_%d_0820'%(backbone_type, input_size)
    id_name_txt = model_bpath + '/id.txt'
    log_name = 'train.log'
    vis_log = model_bpath + '/vis.log'

    gpu_ids = [0]
    batch_size = 64
    class_num = 3000
    dataLoader_util = 'cv2' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 8
    def __init__(self):
        super(Config, self).__init__()
        
