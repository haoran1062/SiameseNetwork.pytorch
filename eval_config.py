# encoding:utf-8

class Config(object):
    
    backbone_type = 'se-resnext50'
    input_size = 448
    
    # backbone_type = 'mobilenet'
    # backbone_type = 'shufflenet'
    input_size = 224

    feature_extract = False
    use_pre_train = False
    input_3x3 = True
    additive_loss_type = 'COCOLoss&CenterLoss'
    # resume_from_path = '/data/haoran/t/se_resnext50_384_0814/best.pth'
    
    resume_from_path = '/data/train_models/classify_models/siamese_models/%s_%d_0903/best.pth'%(backbone_type, input_size)
    crit_resume_path = '/data/train_models/classify_models/siamese_models/%s_%d_0903/best_COCOCrit.pth'%(backbone_type, input_size)
    resume_epoch = 0
    epoch_num = 500
    # model_bpath = '/data/haoran/t/se_resnext50_384_0814'
    
    model_bpath = '/data/train_models/classify_models/siamese_models/%s_%d_0903'%(backbone_type, input_size)
    id_name_txt = model_bpath + '/id.txt'

    gpu_ids = [0]
    batch_size = 1
    class_num = 3000
    dataLoader_util = 'cv2' # cv2, PIL or jpeg4py (jpeg4py is faster) 
    worker_numbers = 4
    def __init__(self):
        super(Config, self).__init__()
        
