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
from TripletDataLoader import ClassifyDataset
from SiameseNet import SiameseNetwork
from TripletLoss import TripletLoss
from CenterLoss import CenterLoss
from COCOLoss import COCOLoss
from triplet_train_config import Config
import torch.nn.functional as F
from dali_train_dataloader import *

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
    
parser = argparse.ArgumentParser(description='Triplet Network Training params')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--world-size", default=1, type=int)


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
    best_coco_crit_w = None
    if  train_cfg.additive_loss_type == 'COCOLoss':
        best_coco_crit_w = copy.deepcopy(addCrit.state_dict())
    elif train_cfg.additive_loss_type == 'COCOLoss&CenterLoss':
        best_coco_crit_w = copy.deepcopy(addCrit[0].state_dict())

    best_acc = 0.0

    for epoch in range(epoch_start, num_epochs):
        # if train_cfg.dist_training:
        #     for phase in datasamplers.keys():
        #         datasamplers[phase].set_epoch(epoch)
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

            now_dataloader = dataloaders[phase]
            stop_it = now_dataloader._size
            # it = 0
            
            # Iterate over data.
            # for it, temp in enumerate(dataloaders[phase]):
            for it, data in enumerate(now_dataloader):

                img1 = data[0]['target_jpegs']
                label1 = data[0]['target_labels'].squeeze()
                img2 = data[0]['pos_jpegs']
                label2 = data[0]['pos_labels'].squeeze()
                img3 = data[0]['neg_jpegs']
                label3 = data[0]['neg_labels'].squeeze()
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
                    
                    feature1, output1, feature2, output2, feature3, output3 = model(img1, img2, img3)
                    

                    if train_cfg.additive_loss_type is None or train_cfg.additive_loss_type == '':
                        trip_loss, softmax_loss1, softmax_loss2, softmax_loss3 = criterion(feature1, feature2, feature3, output1, output2, output3, label1, label2, label3)
                        now_total_loss = 0.01 * trip_loss + 1. * softmax_loss1 + 1. * softmax_loss2 + 1. * softmax_loss3
                    
                    elif train_cfg.additive_loss_type == 'CenterLoss':
                        center_loss = addCrit(feature1, label1) + addCrit(feature2, label2) + addCrit(feature3, label3)
                        trip_loss, softmax_loss1, softmax_loss2, softmax_loss3 = criterion(feature1, feature2, feature3, output1, output2, output3, label1, label2, label3)
                        now_total_loss = 0.01 * trip_loss + 1. * softmax_loss1 + 1. * softmax_loss2 + 1. * softmax_loss3 + 0.001 * center_loss
                    
                    elif train_cfg.additive_loss_type == 'COCOLoss':
                        output1 = addCrit(feature1)
                        output2 = addCrit(feature2)
                        output3 = addCrit(feature3)
                        trip_loss, softmax_loss1, softmax_loss2, softmax_loss3 = criterion(feature1, feature2, feature3, output1, output2, output3, label1, label2, label3)
                        now_total_loss = 0.01 * trip_loss + 1. * softmax_loss1 + 1. * softmax_loss2 + 1. * softmax_loss3

                    elif train_cfg.additive_loss_type == 'COCOLoss&CenterLoss':
                        output1 = addCrit[0](feature1)
                        output2 = addCrit[0](feature2)
                        output3 = addCrit[0](feature3)
                        center_loss = addCrit[1](feature1, label1) + addCrit[1](feature2, label2) + addCrit[1](feature3, label3)
                        trip_loss, softmax_loss1, softmax_loss2, softmax_loss3 = criterion(feature1, feature2, feature3, output1, output2, output3, label1, label2, label3)
                        now_total_loss = 0.01 * trip_loss + 1. * softmax_loss1 + 1. * softmax_loss2 + 1. * softmax_loss3 + 0.001 * center_loss

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
                _, preds3 = torch.max(output3, 1)
                now_correct = torch.sum(preds1 == label1.data) + torch.sum(preds2 == label2.data) + torch.sum(preds3 == label3.data)
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
                    now_acc = float(now_correct) / (len(preds1)*3.0)

                    if phase == 'train':
                        logger.info('Epoch [{}/{}], Iter [{}/{}] expect end in {:4f} min.  average_loss: {:2f}, acc: {:2f}'.format(
                            epoch, 
                            int(num_epochs),
                            it, 
                            dataloaders[phase]._size // train_cfg.batch_size, 
                            it_cost_time * (dataloaders[phase]._size // train_cfg.batch_size - it+1) / 60, 
                            running_loss / (it+1),
                            now_acc ) )
                    
                    img_1 = tensor2img(img1, normal=True)
                    vis.img('pred1 img', img_1)

                    t_pred = F.softmax(output1, 1)[0]
                    show_id = t_pred.argmax().cpu().item()
                    conf = t_pred[t_pred.argmax()].cpu().item()

                    if id_name_map:
                        
                        if show_id in id_name_map.keys():
                            show_id = id_name_map[show_id]
                        vis.img('pred1 result', get_show_result_img(id_name_map[label1.to('cpu').numpy()[0]], show_id, conf))
                    else:
                        vis.img('pred1 result', get_show_result_img(label1.to('cpu').numpy()[0], preds1.to('cpu').numpy()[0], conf))

                    img_2 = tensor2img(img2, normal=True)
                    vis.img('pred2 img', img_2)

                    t_pred = F.softmax(output2, 1)[0]
                    show_id = t_pred.argmax().cpu().item()
                    conf = t_pred[t_pred.argmax()].cpu().item()
                    if id_name_map:
                        
                        if show_id in id_name_map.keys():
                            show_id = id_name_map[show_id]
                        vis.img('pred2 result', get_show_result_img(id_name_map[label2.to('cpu').numpy()[0]], show_id, conf))
                    else:
                        vis.img('pred2 result', get_show_result_img(label2.to('cpu').numpy()[0], preds2.to('cpu').numpy()[0], conf))
                    
                    img_3 = tensor2img(img3, normal=True)
                    vis.img('pred3 img', img_3)

                    t_pred = F.softmax(output3, 1)[0]
                    show_id = t_pred.argmax().cpu().item()
                    conf = t_pred[t_pred.argmax()].cpu().item()
                    if id_name_map:
                        
                        if show_id in id_name_map.keys():
                            show_id = id_name_map[show_id]
                        vis.img('pred3 result', get_show_result_img(id_name_map[label3.to('cpu').numpy()[0]], show_id, conf))
                    else:
                        vis.img('pred3 result', get_show_result_img(label3.to('cpu').numpy()[0], preds3.to('cpu').numpy()[0], conf))
                    
                    # print(feature1.shape)  .pow(2).sum(1)
                    vis.img('pos distance', get_show_result_img(0, (feature1[0].unsqueeze(0) - feature2[0].unsqueeze(0)).pow(2).sum(1).detach().to('cpu').numpy()[0]))
                    vis.img('neg distance', get_show_result_img(1, (feature1[0].unsqueeze(0) - feature3[0].unsqueeze(0)).pow(2).sum(1).detach().to('cpu').numpy()[0]))

                    # vis.img('pos distance', get_show_result_img(0, F.pairwise_distance(feature1[0].unsqueeze(0), feature2[0].unsqueeze(0)).detach().to('cpu').numpy()[0]))
                    # vis.img('neg distance', get_show_result_img(1, F.pairwise_distance(feature1[0].unsqueeze(0), feature3[0].unsqueeze(0)).detach().to('cpu').numpy()[0]))

                if it == stop_it and phase == 'train':
                    if not os.path.exists('%s'%(save_base_path)):
                        os.mkdir('%s'%(save_base_path))
                    save_checkpoint(model, optimizer, epoch, '%s/epoch_%d.pth'%(save_base_path, epoch))
                    if  train_cfg.additive_loss_type == 'COCOLoss':
                        torch.save(addCrit.state_dict(), '%s/epoch_%d_COCOCrit.pth'%(save_base_path, epoch))

                    elif train_cfg.additive_loss_type == 'COCOLoss&CenterLoss':
                        torch.save(addCrit[0].state_dict(), '%s/epoch_%d_COCOCrit.pth'%(save_base_path, epoch))
                    # torch.save(model.state_dict(), '%s/epoch_%d.pth'%(save_base_path, epoch))
                    break


            epoch_loss = running_loss / (dataloaders[phase]._size)
            epoch_acc = running_corrects.double() / (dataloaders[phase]._size * 3)

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
                if best_coco_crit_w is not None:
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    img_input_size = train_cfg.input_size

    if train_cfg.dist_training:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(args.local_rank)
        device = "cuda:%d"%(args.local_rank) if torch.cuda.is_available() else "cpu"
        # world_size = torch.distributed.get_world_size()
        torch.distributed.init_process_group(backend='nccl', init_method='env://')# , rank=args.local_rank)
        args.world_size = torch.distributed.get_world_size()
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
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1 * train_cfg.batch_size * 3 / 256.0, momentum=0.9)
    if fp16_using:
        model_ft, optimizer_ft = amp.initialize(model_ft, optimizer_ft, opt_level='O1', loss_scale=128.0)
        if train_cfg.dist_training:
            model_p = DDP(model_ft, delay_allreduce=True)
        else:
            model_p = nn.DataParallel(model_ft, device_ids=train_cfg.gpu_ids)
    # model_ft.load_state_dict(torch.load(config_map['resume_from_path']))
    else:
        if train_cfg.dist_training:
            pass
        else:
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
    criterion = TripletLoss(train_cfg.use_focal_loss)

    add_crit = None
    if train_cfg.additive_loss_type == 'CenterLoss':
        add_crit = CenterLoss(train_cfg.class_num)
    elif train_cfg.additive_loss_type == 'COCOLoss':
        add_crit = COCOLoss(train_cfg.class_num)
    elif train_cfg.additive_loss_type == 'COCOLoss&CenterLoss':
        add_crit = [COCOLoss(train_cfg.class_num), CenterLoss(train_cfg.class_num)]

    out_map = ['target_jpegs', 'target_labels', 'pos_jpegs', 'pos_labels', 'neg_jpegs', 'neg_labels']
    dataloaders = {}
    # data_samplers = {}
    train_loader = get_dataloader(CustomTripletIterator, TripletPipeline, out_map, train_cfg, True, args.local_rank)
    test_loader = get_dataloader(CustomTripletIterator, TripletPipeline, out_map, train_cfg, False, args.local_rank)
    logger.info('the dataset has %d images' % (train_loader._pipes[0].dataset.n))
    logger.info('the batch_size is %d' % (train_cfg.batch_size))

    dataloaders['train']=train_loader
    dataloaders['test']=test_loader
    model_p.train()
    # Train and evaluate
    train_model(model_p, dataloaders, criterion, add_crit, optimizer_ft, train_cfg, logger=logger, vis=my_vis)

       
    