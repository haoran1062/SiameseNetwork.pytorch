# Siamese-Network
## 
### how to run
* modify `train_config.py` and `eval_config.py`: 
  * change `backbone_type` you want use [`se-resnext50`, `vgg`, `resnet50`, `resnext50`, `resnext101`] you can add your backbone in `backbones`, or add torchvision supported model by modify `utils/model_utils.py`
  * **train_datasets_bpath** = `'data/to/your/path'` and same to **test_datasets_bpath**
  * change dataLoader_util to you want use(cv2 by default) [`cv2`/`PIL`/`jpeg4py`([jpeg4py](https://github.com/ajkxyz/jpeg4py) need compile )]
  * change `batch_size` and `gpu_ids` to you want use
  * if you wanna use **FP16 Mixed Precision**, [apex](https://github.com/NVIDIA/apex) requirement, then change `fp16_using` = True
  * if train with center loss, change `additive_loss_type` = `'CenterLoss'`
  * if train with COCO loss, change `additive_loss_type` = `'COCOLoss'`
  * if train with center loss and COCO loss, change `additive_loss_type` = `'COCOLoss&CenterLoss'`
  * if train only use softmax, change `additive_loss_type` = `None` or `''`
  * if train with focal loss, change `use_focal_loss` = `True`
* run `nohup python -m visdom.server &` on linux shell then goto `localhost:8097` to see your model visual output
* then run `python train.py` to train ~~or run `fast_train.py` to train with `DALI` that `3x~20x`(still debuging only support single GPU now) faster than `pytorch dataloader`~~

### requirement
* pytorch >= 1.0
* opencv
* PIL
* visdom
* [jpeg4py](https://github.com/ajkxyz/jpeg4py)
* tqdm
* [apex](https://github.com/NVIDIA/apex)
* [DALI](https://github.com/NVIDIA/DALI.git)

### TODO
* support multi-GPU DALI speedup