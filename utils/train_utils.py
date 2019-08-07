# encoding:utf-8
import os, numpy as np, random, cv2, logging, json, torch



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)
            
    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

def get_config_map(file_path):
    print(file_path)
    config_map = json.loads(open(file_path).read())
    
    config_map['batch_size'] *= len(config_map['gpu_ids'])
    return config_map

def create_logger(base_path, log_name):

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    fhander = logging.FileHandler('%s/%s.log'%(base_path, log_name))
    fhander.setLevel(logging.INFO)

    shander = logging.StreamHandler()
    shander.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    fhander.setFormatter(formatter) 
    shander.setFormatter(formatter) 

    logger.addHandler(fhander)
    logger.addHandler(shander)

    return logger

def get_show_result_img(gt_label, pred_label):
    img = np.zeros((100, 500, 3), np.uint8)
    str_input = 'gt: %.2f, pred : %.2f'%(float(gt_label), float(pred_label))
    cv2.putText(img, str_input, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2)
    return img

def convert_show_cls_bar_data(acc_map, rename_map=None):
    mAP = 0.
    kl = acc_map.keys()
    name_l = [str(i) for i in kl]
    if rename_map:
        name_l = [str(rename_map[i]) for i in kl]
    acc_np = np.zeros((len(kl), 2), np.int32)


    for it, k in enumerate(kl):
        acc_np[it, :] = acc_map[k]
        print('now cls id: %5s, total : %5d, right: %5d, wrong: %5d, Acc %.3f'%(name_l[it], acc_map[k][0] + acc_map[k][1], acc_map[k][0], acc_map[k][1], acc_map[k][0]/(acc_map[k][0] + acc_map[k][1])))
        mAP += acc_map[k][0]/(acc_map[k][0] + acc_map[k][1])
    mAP /= len(kl)
    print('*'*20, 'mAP is : %.5f'%(mAP), '*'*20)
    leg_l = ['right', 'wrong']
    return acc_np, leg_l, name_l
