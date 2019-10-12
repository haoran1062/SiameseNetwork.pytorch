# encoding:utf-8

class Config(object):
    
    backbone_type = 'se-resnext50'
    # backbone_type = 'mobilenet'
    # backbone_type = 'shufflenet'
    input_size = 448
    train_datasets_bpath = ['/dev/shm/datasets/201906-0807_all/train', '/dev/shm/datasets/UE4_cls_0926/train', '/dev/shm/datasets/cls_0808-12/train', '/dev/shm/datasets/cls_0813/train', '/dev/shm/datasets/cls_0820/train', '/dev/shm/datasets/cls_0821/train', '/dev/shm/datasets/cls_0822/train', '/dev/shm/datasets/cls_0819_zh/train', '/dev/shm/datasets/cls_0814-16/train', '/dev/shm/datasets/0826-0829_all/train', '/dev/shm/datasets/cls_0903/train', '/dev/shm/datasets/cls_0905/all', '/dev/shm/datasets/cls_0906/all', '/dev/shm/datasets/cls_1_0909/all', '/dev/shm/datasets/cls_2_0909/all', '/dev/shm/datasets/cls_3_0909/all', '/dev/shm/datasets/cls_1_0910/all', '/dev/shm/datasets/cls_1_0912/all', '/dev/shm/datasets/cls_1_0913/all', '/dev/shm/datasets/cls_1_0917/all', '/dev/shm/datasets/cls_1_0918/all', '/dev/shm/datasets/cls_1_0919/all', '/dev/shm/datasets/cls_1_0920/all', '/dev/shm/datasets/cls_1_0923/all', '/dev/shm/datasets/cls_1_0924/all']
    test_datasets_bpath = ['/dev/shm/datasets/201906-0807_all/val', '/dev/shm/datasets/UE4_cls_0926/val', '/dev/shm/datasets/cls_0808-12/val', '/dev/shm/datasets/cls_0813/val', '/dev/shm/datasets/cls_0820/val', '/dev/shm/datasets/cls_0821/val', '/dev/shm/datasets/cls_0822/val', '/dev/shm/datasets/cls_0819_zh/val', '/dev/shm/datasets/cls_0814-16/val', '/dev/shm/datasets/0826-0829_all/val', '/dev/shm/datasets/cls_0903/val', '/dev/shm/datasets/cls_0905/all', '/dev/shm/datasets/cls_0906/all', '/dev/shm/datasets/cls_1_0909/all', '/dev/shm/datasets/cls_2_0909/all', '/dev/shm/datasets/cls_3_0909/all', '/dev/shm/datasets/cls_1_0910/all', '/dev/shm/datasets/cls_1_0912/all', '/dev/shm/datasets/cls_1_0913/all', '/dev/shm/datasets/cls_1_0917/all', '/dev/shm/datasets/cls_1_0918/all', '/dev/shm/datasets/cls_1_0919/all', '/dev/shm/datasets/cls_1_0920/all', '/dev/shm/datasets/cls_1_0923/all', '/dev/shm/datasets/cls_1_0924/all']

    # train_datasets_bpath = ['/data/datasets/classify_data/trurh_data/cls_0903/train']# , '/data/datasets/classify_data/truth_data/20190905/train', '/data/datasets/classify_data/truth_data/20190906/train']
    # test_datasets_bpath = ['/data/datasets/classify_data/trurh_data/cls_0903/val']# , '/data/datasets/classify_data/truth_data/20190905/val', '/data/datasets/classify_data/truth_data/20190906/val']
    # use_center_loss = True
    additive_loss_type = 'COCOLoss&CenterLoss'# 'COCOLoss'
    use_focal_loss = True
    feature_extract = False
    use_pre_train = False
    input_3x3 = True
    model_bpath = '/data/train_models/classify_models/triplet_models/%s_%d_0924'%(backbone_type, input_size)
    # resume_from_path = '%s/best.pth'%(model_bpath) # "/data/train_models/classify_models/triplet_models/se-resnext50_448_0918/best.pth"
    resume_from_path = '/data/train_models/classify_models/triplet_models/se-resnext50_448_0923/best.pth'
    fp16_using = True
    dist_training = True
    random_shuffle = True
    resume_epoch = 0
    epoch_num = 18
    # model_bpath = '/data/train_models/classify_models/triplet_models/%s_%d_0923'%(backbone_type, input_size)
    id_name_txt = model_bpath + '/id.txt'
    log_name = 'train.log'
    vis_log = model_bpath + '/vis.log'

    gpu_ids = [0, 1, 2]
    batch_size = 18
    class_num = 8000
    same_cate_prob = 0.5
    dataLoader_util = 'cv2' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 6
    def __init__(self):
        super(Config, self).__init__()
        
