# Siamese-Network
## 
### how to run
* modify `train_config.py` and `eval_config.py`: 
  * change `backbone_type` you want use [`se-resnext50`, `vgg`, `resnet50`, `resnext50`, `resnext101`] you can add your backbone in `backbones`, or add torchvision supported model by modify `utils/model_utils.py`
  * **train_datasets_bpath** = `'data/to/your/path'` and same to **test_datasets_bpath**
  * change dataLoader_util to you want use(cv2 by default) [`cv2`/`PIL`/`jpeg4py`([jpeg4py](https://github.com/ajkxyz/jpeg4py) need compile )]
  * change `batch_size` and `gpu_ids` to you want use
  * if you wanna use **FP16 Mixed Precision**, [apex](https://github.com/NVIDIA/apex) requirement, then change `fp16_using` = True
* run `nohup python -m visdom.server &` on linux shell then goto `localhost:8097` to see your model visual output
* then run `python train.py` to train

### requirement
* pytorch >= 1.0
* opencv
* PIL
* visdom
* [jpeg4py](https://github.com/ajkxyz/jpeg4py)
* tqdm
* [apex](https://github.com/NVIDIA/apex)

