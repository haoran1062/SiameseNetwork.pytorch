# encoding:utf-8

class Config(object):
    
    backbone_type = 'se-resnext50'
    # backbone_type = 'mobilenet'
    # backbone_type = 'shufflenet'
    input_size = 448
    train_datasets_bpath = ['/dev/shm/datasets/201906-0807_all/train', '/dev/shm/datasets/UE4_cls_0905/train', '/dev/shm/datasets/cls_0808-12/train', '/dev/shm/datasets/cls_0813/train', '/dev/shm/datasets/cls_0820/train', '/dev/shm/datasets/cls_0821/train', '/dev/shm/datasets/cls_0822/train', '/dev/shm/datasets/cls_0819_zh/train', '/dev/shm/datasets/cls_0814-16/train', '/dev/shm/datasets/0826-0829_all/train', '/dev/shm/datasets/cls_0903/train', '/dev/shm/datasets/cls_0905/all']
    test_datasets_bpath = ['/dev/shm/datasets/201906-0807_all/val', '/dev/shm/datasets/UE4_cls_0905/val', '/dev/shm/datasets/cls_0808-12/val', '/dev/shm/datasets/cls_0813/val', '/dev/shm/datasets/cls_0820/val', '/dev/shm/datasets/cls_0821/val', '/dev/shm/datasets/cls_0822/val', '/dev/shm/datasets/cls_0819_zh/val', '/dev/shm/datasets/cls_0814-16/val', '/dev/shm/datasets/0826-0829_all/val', '/dev/shm/datasets/cls_0903/val']

    use_center_loss = True
    use_focal_loss = True
    feature_extract = False
    use_pre_train = False
    input_3x3 = True
    resume_from_path = "/data/train_models/classify_models/siamese_models/se-resnext50_448_0903/epoch_3.pth"
    fp16_using = True
    random_shuffle = True
    resume_epoch = 0
    epoch_num = 12
    model_bpath = '/data/train_models/classify_models/siamese_models/%s_%d_0903'%(backbone_type, input_size)
    id_name_txt = model_bpath + '/id.txt'
    log_name = 'train.log'
    vis_log = model_bpath + '/vis.log'

    gpu_ids = [0, 1, 2, 3]
    batch_size = 192
    class_num = 3000
    same_cate_prob = 0.5
    dataLoader_util = 'cv2' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 4
    def __init__(self):
        super(Config, self).__init__()
        
