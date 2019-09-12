#encoding:utf-8
import os, sys, time, numpy as np, cv2, copy, argparse
from glob import glob 

import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from utils.data_utils import *
from utils.train_utils import *
from utils.visual import Visual
from torchsummary import summary
from dataLoader import ClassifyDataset
from SiameseNet import SiameseNetwork
from ContrastiveLoss import ContrastiveLoss
from CenterLoss import CenterLoss
from COCOLoss import COCOLoss
from train_config import Config
import torch.nn.functional as F

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    fp16_using = True
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    fp16_using = False

train_cfg = Config()

if fp16_using:
    fp16_using = train_cfg.fp16_using
print('using FP16 Mixed : ', fp16_using)
    

parser = argparse.ArgumentParser(
    description='Simaese Network Training params')

args = parser.parse_args()


if not os.path.exists(train_cfg.model_bpath):
    os.makedirs(train_cfg.model_bpath)

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params


def train_model(model, dataloaders, criterion, addCrit, optimizer, train_cfg, save_step=1000, logger=None, vis=None, rename_map=None, id_name_map=None):
    since = time.time()
    
    num_epochs=train_cfg.epoch_num
    epoch_start=train_cfg.resume_epoch
    save_base_path=train_cfg.model_bpath
    # cosin_lr = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs // 10)+1)
    adjust_lr = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.8, verbose=1, patience=2)
    best_model_wts = copy.deepcopy(model.state_dict())
    if  train_cfg.additive_loss_type == 'COCOLoss':
        best_coco_crit_w = copy.deepcopy(addCrit.state_dict())
    elif train_cfg.additive_loss_type == 'COCOLoss&CenterLoss':
        best_coco_crit_w = copy.deepcopy(addCrit[0].state_dict())

    best_acc = 0.0

    for epoch in range(epoch_start, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        acc_map = {}
        # cosin_lr.step(epoch)
        my_vis.plot('lr', optimizer.param_groups[0]['lr'])

        # Each epoch has a training and validation phase
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            prefetcher = data_prefetcher(dataloaders[phase])
            data = prefetcher.next()
            it = 0
            
            # Iterate over data.
            # for it, temp in enumerate(dataloaders[phase]):
            while data is not None:
                img1, img2, label1, label2, sim_labels = data
                # inputs, labels = temp
                img1 = img1.to(device)
                img2 = img2.to(device)
                label1 = label1.to(device)
                label2 = label2.to(device)
                sim_labels = sim_labels.to(device)
                
                st = time.clock()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    
                    feature1, output1, feature2, output2 = model(img1, img2)
                    

                    if train_cfg.additive_loss_type is None or train_cfg.additive_loss_type == '':
                        sim_loss, softmax_loss1, softmax_loss2 = criterion(feature1, feature2, output1, output2, label1, label2, sim_labels)
                        now_total_loss = 0.01 * sim_loss + 1. * softmax_loss1 + 1. * softmax_loss2
                    
                    elif train_cfg.additive_loss_type == 'CenterLoss':
                        center_loss = addCrit(feature1, label1) + addCrit(feature2, label2)
                        sim_loss, softmax_loss1, softmax_loss2 = criterion(feature1, feature2, output1, output2, label1, label2, sim_labels)
                        now_total_loss = 0.01 * sim_loss + 1. * softmax_loss1 + 1. * softmax_loss2 + 0.001 * center_loss
                    
                    elif train_cfg.additive_loss_type == 'COCOLoss':
                        output1 = addCrit(feature1)
                        output2 = addCrit(feature2)
                        sim_loss, softmax_loss1, softmax_loss2 = criterion(feature1, feature2, output1, output2, label1, label2, sim_labels)
                        now_total_loss = 0.01 * sim_loss + 1. * softmax_loss1 + 1. * softmax_loss2

                    elif train_cfg.additive_loss_type == 'COCOLoss&CenterLoss':
                        output1 = addCrit[0](feature1)
                        output2 = addCrit[0](feature2)
                        center_loss = addCrit[1](feature1, label1) + addCrit[1](feature2, label2)
                        sim_loss, softmax_loss1, softmax_loss2 = criterion(feature1, feature2, output1, output2, label1, label2, sim_labels)
                        now_total_loss = 0.01 * sim_loss + 1. * softmax_loss1 + 1. * softmax_loss2 + 0.001 * center_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if fp16_using:
                            with amp.scale_loss(now_total_loss, optimizer) as bp_loss:
                                bp_loss.backward()
                        else:
                            now_total_loss.backward()

                        optimizer.step()
                

                # statistics
                now_loss = now_total_loss.item() # * img1.size(0)
                running_loss += now_loss

                _, preds1 = torch.max(output1, 1)
                _, preds2 = torch.max(output2, 1)
                now_correct = torch.sum(preds1 == label1.data) + torch.sum(preds2 == label2.data)
                # print(now_correct)
                running_corrects += now_correct

                if phase == 'test':
                    p_l = preds1.data.tolist()
                    gt_l = label1.data.tolist()
                    for tij in range(len(p_l)):
                        t_gt = gt_l[tij]
                        if t_gt not in acc_map.keys():
                            acc_map[t_gt] = [0, 0]
                        if t_gt == p_l[tij]:
                            acc_map[t_gt][0] += 1
                        else:
                            acc_map[t_gt][1] += 1

                ed = time.clock()
                it_cost_time = ed - st
                
                if it % 10 == 0:
                    # convert_show_cls_bar_data(acc_map, rename_map=rename_map)
                    now_acc = float(now_correct) / (len(preds1)*2.0)

                    if phase == 'train':
                        logger.info('Epoch [{}/{}], Iter [{}/{}] expect end in {:4f} min.  average_loss: {:2f}, acc: {:2f}'.format(
                            epoch, 
                            int(num_epochs),
                            it, 
                            len(dataloaders[phase]), 
                            it_cost_time * (len(dataloaders[phase]) - it+1) / 60, 
                            running_loss / (it+1),
                            now_acc ) )

                    img_1 = tensor2img(img1, normal=True)
                    vis.img('pred1 img', img_1)
                    if id_name_map:
                        t_pred = F.softmax(output1, 1)[0]
                        show_id = t_pred.argmax().cpu().item()
                        conf = t_pred[t_pred.argmax()].cpu().item()


                        # conf, _ = torch.max(output1, 1)
                        # conf = conf.cpu()[0]

                        # show_id = preds1.to('cpu').numpy()[0]
                        # conf = output1[0][show_id].cpu().item()
                        if show_id in id_name_map.keys():
                            show_id = id_name_map[show_id]
                        vis.img('pred1 result', get_show_result_img(id_name_map[label1.to('cpu').numpy()[0]], show_id, conf))
                    else:
                        vis.img('pred1 result', get_show_result_img(label1.to('cpu').numpy()[0], preds1.to('cpu').numpy()[0], conf))

                    img_2 = tensor2img(img2, normal=True)
                    vis.img('pred2 img', img_2)
                    if id_name_map:
                        t_pred = F.softmax(output2, 1)[0]
                        show_id = t_pred.argmax().cpu().item()
                        conf = t_pred[t_pred.argmax()].cpu().item()
                        # conf, _ = torch.max(output1, 1)
                        # conf = conf.cpu()[0]

                        # show_id = preds2.to('cpu').numpy()[0]
                        # conf = output2[0][show_id].cpu().item()
                        if show_id in id_name_map.keys():
                            show_id = id_name_map[show_id]
                        vis.img('pred2 result', get_show_result_img(id_name_map[label2.to('cpu').numpy()[0]], show_id, conf))
                    else:
                        vis.img('pred2 result', get_show_result_img(label2.to('cpu').numpy()[0], preds2.to('cpu').numpy()[0], conf))
                    
                    # print(feature1.shape)
                    vis.img('like result', get_show_result_img(sim_labels.to('cpu').numpy()[0], F.pairwise_distance(feature1[0].unsqueeze(0), feature2[0].unsqueeze(0)).detach().to('cpu').numpy()[0]))

                if it % save_step == 0 and phase == 'train':
                    if not os.path.exists('%s'%(save_base_path)):
                        os.mkdir('%s'%(save_base_path))
                    save_checkpoint(model, optimizer, epoch, '%s/epoch_%d.pth'%(save_base_path, epoch))
                    if  train_cfg.additive_loss_type == 'COCOLoss':
                        torch.save(addCrit.state_dict(), '%s/epoch_%d_COCOCrit.pth'%(save_base_path, epoch))

                    elif train_cfg.additive_loss_type == 'COCOLoss&CenterLoss':
                        torch.save(addCrit[0].state_dict(), '%s/epoch_%d_COCOCrit.pth'%(save_base_path, epoch))
                    # torch.save(model.state_dict(), '%s/epoch_%d.pth'%(save_base_path, epoch))
                
                data = prefetcher.next()
                it += 1
                if it == len(dataloaders[phase]):
                    it = 0
                    break

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset) * 2)

            adjust_lr.step(epoch_loss)

            if phase == 'train':
                my_vis.plot('train loss', epoch_loss)
                my_vis.plot('train acc', epoch_acc.item())

            elif phase == 'test':
                my_vis.plot('test loss', epoch_loss)
                my_vis.plot('test acc', epoch_acc.item())

                acc_x, leg_l, name_l = convert_show_cls_bar_data(acc_map, save_base_path+'/meanAcc.txt', rename_map=rename_map)
                my_vis.multi_cls_bar('every class Acc', acc_x, leg_l, name_l)


            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                if  train_cfg.additive_loss_type == 'COCOLoss':
                    best_coco_crit_w = copy.deepcopy(addCrit.state_dict())
                elif train_cfg.additive_loss_type == 'COCOLoss&CenterLoss':
                    best_coco_crit_w = copy.deepcopy(addCrit[0].state_dict())
                
                # model.load_state_dict(best_model_wts)
                # torch.save(model.state_dict(), '%s/best.pth'%(save_base_path))
                torch.save(best_coco_crit_w, '%s/best_COCOCrit.pth'%(save_base_path))
                save_checkpoint(model, optimizer, epoch, '%s/best.pth'%(save_base_path))

                best_acc = epoch_acc
                acc_x, leg_l, name_l = convert_show_cls_bar_data(acc_map, save_base_path+'/best_meanAcc.txt', rename_map=rename_map)
            if phase == 'test':
                pass

    time_elapsed = time.time() - since
    logger.info('finish training using %.2fs'%(time_elapsed))
    # load best model weights
    

def save_checkpoint(model, optimizer, epoch, save_path):
    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(state, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    return model, optimizer, start_epoch
    



if __name__ == "__main__":

    fine_turn = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    img_input_size = train_cfg.input_size

    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.Lambda(lambda img: origin_resize(img)),
            transforms.Lambda(lambda img: padding_resize(img, resize=img_input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.Lambda(lambda img: origin_resize(img)),
            transforms.Lambda(lambda img: padding_resize(img, resize=img_input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.Lambda(lambda img: origin_resize(img)),
            transforms.Lambda(lambda img: padding_resize(img, resize=img_input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Initialize the model for this run
    model_ft = SiameseNetwork(train_cfg).to(device)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1 * train_cfg.batch_size * 2 / 256.0, momentum=0.9)
    if fp16_using:
        model_ft, optimizer_ft = amp.initialize(model_ft, optimizer_ft, opt_level='O1', loss_scale=128.0)
    # model_ft.load_state_dict(torch.load(config_map['resume_from_path']))
    model_p = nn.DataParallel(model_ft, device_ids=train_cfg.gpu_ids)
    
    if train_cfg.resume_from_path:
        print("resume from %s"%(train_cfg.resume_from_path))
        # model_p.load_state_dict(torch.load(train_cfg.resume_from_path))
        model_p, optimizer_ft, train_cfg.resume_epoch = load_checkpoint(model_p, optimizer_ft, train_cfg.resume_from_path)

    logger = create_logger(train_cfg.model_bpath, train_cfg.log_name)

    my_vis = Visual(train_cfg.model_bpath, log_to_file=train_cfg.vis_log)   

    # Observe that all parameters are being optimized
    
    # optimizer_ft = optim.RMSprop(params_to_update, momentum=0.9)
    # optimizer_ft = optim.Adam(model_p.parameters(), lr=1e-2, eps=1e-8, betas=(0.9, 0.99), weight_decay=0.)
    # optimizer_ft = optim.Adadelta(params_to_update, lr=1)

    # Setup the loss fxn
    # criterion = nn.CrossEntropyLoss()
    criterion = ContrastiveLoss(train_cfg.use_focal_loss)

    add_crit = None
    if train_cfg.additive_loss_type == 'CenterLoss':
        add_crit = CenterLoss(train_cfg.class_num)
    elif train_cfg.additive_loss_type == 'COCOLoss':
        add_crit = COCOLoss(train_cfg.class_num)
    elif train_cfg.additive_loss_type == 'COCOLoss&CenterLoss':
        add_crit = [COCOLoss(train_cfg.class_num), CenterLoss(train_cfg.class_num)]


    dataloaders = {}
    train_dataset = ClassifyDataset(base_data_path=train_cfg.train_datasets_bpath, train=True, transform = data_transforms['train'], read_mode=train_cfg.dataLoader_util, id_name_path=train_cfg.id_name_txt, device=device, little_train=False)
    train_loader = DataLoader(train_dataset,batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.worker_numbers, pin_memory=True)
    test_dataset = ClassifyDataset(base_data_path=train_cfg.test_datasets_bpath, train=False,transform = data_transforms['val'], read_mode=train_cfg.dataLoader_util, id_name_path=train_cfg.id_name_txt, device=device, little_train=False)
    test_loader = DataLoader(test_dataset,batch_size=train_cfg.batch_size,shuffle=False, num_workers=train_cfg.worker_numbers, pin_memory=True)
    id_name_map = train_dataset.id_name_map
    data_len = int(len(test_dataset) / train_cfg.batch_size)
    logger.info('the dataset has %d images' % (len(train_dataset)))
    logger.info('the batch_size is %d' % (train_cfg.batch_size))

    dataloaders['train']=train_loader
    dataloaders['test']=test_loader

    model_p.train()
    # Train and evaluate
    train_model(model_p, dataloaders, criterion, add_crit, optimizer_ft, train_cfg, logger=logger, vis=my_vis, rename_map=id_name_map, id_name_map=id_name_map)

