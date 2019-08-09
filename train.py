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
from train_config import Config
import torch.nn.functional as F

parser = argparse.ArgumentParser(
    description='Simaese Network Training params')

args = parser.parse_args()

train_cfg = Config()
if not os.path.exists(train_cfg.model_bpath):
    os.makedirs(train_cfg.model_bpath)

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, epoch_start=0, save_base_path='./', save_step=500, logger=None, vis=None, rename_map=None, id_name_map=None):
    since = time.time()
    
    # cosin_lr = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs // 10)+1)
    adjust_lr = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.8, verbose=1, patience=2)
    best_model_wts = copy.deepcopy(model.state_dict())
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
                    _, preds1 = torch.max(output1, 1)
                    _, preds2 = torch.max(output2, 1)
                    
                    sim_loss, softmax_loss1, softmax_loss2 = criterion(feature1, feature2, output1, output2, label1, label2, sim_labels)
                    now_total_loss = 0.01 * sim_loss + 1. * softmax_loss1 + 1. * softmax_loss2 # sim_loss + 

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        now_total_loss.backward()
                        # softmax_loss1.backward()
                        # sim_loss.backward(retain_graph=True)
                        # softmax_loss1.backward(retain_graph=True)
                        # softmax_loss2.backward(retain_graph=True)
                        optimizer.step()
                

                # statistics
                now_loss = now_total_loss.item() # * img1.size(0)
                running_loss += now_loss
                
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
                        show_id = preds1.to('cpu').numpy()[0]
                        if show_id in id_name_map.keys():
                            show_id = id_name_map[show_id]
                        vis.img('pred1 result', get_show_result_img(id_name_map[label1.to('cpu').numpy()[0]], show_id))
                    else:
                        vis.img('pred1 result', get_show_result_img(label1.to('cpu').numpy()[0], preds1.to('cpu').numpy()[0]))

                    img_2 = tensor2img(img2, normal=True)
                    vis.img('pred2 img', img_2)
                    if id_name_map:
                        show_id = preds2.to('cpu').numpy()[0]
                        if show_id in id_name_map.keys():
                            show_id = id_name_map[show_id]
                        vis.img('pred2 result', get_show_result_img(id_name_map[label2.to('cpu').numpy()[0]], show_id))
                    else:
                        vis.img('pred2 result', get_show_result_img(label2.to('cpu').numpy()[0], preds2.to('cpu').numpy()[0]))
                    
                    # print(feature1.shape)
                    vis.img('like result', get_show_result_img(sim_labels.to('cpu').numpy()[0], F.pairwise_distance(feature1[0].unsqueeze(0), feature2[0].unsqueeze(0)).detach().to('cpu').numpy()[0]))

                if it % save_step == 0 and phase == 'train':
                    if not os.path.exists('%s'%(save_base_path)):
                        os.mkdir('%s'%(save_base_path))
                    torch.save(model.state_dict(), '%s/epoch_%d.pth'%(save_base_path, epoch))
                
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

                acc_x, leg_l, name_l = convert_show_cls_bar_data(acc_map, rename_map=rename_map)
                my_vis.multi_cls_bar('every class Acc', acc_x, leg_l, name_l)


            # deep copy the model
            if phase == 'test' and epoch_loss > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_loss
            if phase == 'test':
                pass

    time_elapsed = time.time() - since
    logger.info('finish training using %.2fs'%(time_elapsed))
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), '%s/best.pth'%(save_base_path))



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
    model_ft = SiameseNetwork(train_cfg)
    
    # model_ft.load_state_dict(torch.load(config_map['resume_from_path']))
    model_p = nn.DataParallel(model_ft.to(device), device_ids=train_cfg.gpu_ids)
    if train_cfg.resume_from_path:
        print("resume from %s"%(train_cfg.resume_from_path))
        model_p.load_state_dict(torch.load(train_cfg.resume_from_path))
        
   
    # Print the model we just instantiated
    # summary(model_p, (3, img_input_size, img_input_size))

    # Send the model to GPU

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are 
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.

    logger = create_logger(train_cfg.model_bpath, train_cfg.log_name)

    my_vis = Visual(train_cfg.model_bpath, log_to_file=train_cfg.vis_log)   

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_p.parameters(), lr=0.1 * train_cfg.batch_size / 256.0, momentum=0.9)
    # optimizer_ft = optim.RMSprop(params_to_update, momentum=0.9)
    # optimizer_ft = optim.Adam(model_p.parameters(), lr=1e-2, eps=1e-8, betas=(0.9, 0.99), weight_decay=0.)
    # optimizer_ft = optim.Adadelta(params_to_update, lr=1)

    # Setup the loss fxn
    # criterion = nn.CrossEntropyLoss()
    criterion = ContrastiveLoss()

    dataloaders = {}
    train_dataset = ClassifyDataset(base_data_path=train_cfg.train_datasets_bpath, train=True, transform = data_transforms['train'], id_name_path=train_cfg.id_name_txt, device=device, little_train=False)
    train_loader = DataLoader(train_dataset,batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.worker_numbers, pin_memory=True)
    test_dataset = ClassifyDataset(base_data_path=train_cfg.test_datasets_bpath, train=False,transform = data_transforms['val'], id_name_path=train_cfg.id_name_txt, device=device, little_train=False)
    test_loader = DataLoader(test_dataset,batch_size=train_cfg.batch_size,shuffle=True, num_workers=train_cfg.worker_numbers, pin_memory=True)
    id_name_map = train_dataset.id_name_map
    data_len = int(len(test_dataset) / train_cfg.batch_size)
    logger.info('the dataset has %d images' % (len(train_dataset)))
    logger.info('the batch_size is %d' % (train_cfg.batch_size))

    dataloaders['train']=train_loader
    dataloaders['test']=test_loader

    model_p.train()
    # Train and evaluate
    train_model(model_p, dataloaders, criterion, optimizer_ft, num_epochs=train_cfg.epoch_num, epoch_start=train_cfg.resume_epoch, save_base_path=train_cfg.model_bpath, logger=logger, vis=my_vis, rename_map=id_name_map, id_name_map=id_name_map)

