# Rotated Deformable DETR (ICLR 2021)


> [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)


<!-- [ALGORITHM] -->

## Abstract

DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach.


<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143877617-ad9b24fd-77ce-46aa-9689-1a44b5594132.png" width="800"/>
</div>


### DOTA-v1.0

|  Model  | bs |  Aug  | mAP  | AP50 | AP75 | Params | lr schd | lr | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Deformable<br>DETR  |    R-50   |  -   | 39.63 | 68.65 | 39.83 |   40.41M  | 50e |  1e-4   |    4=2gpu*<br>2img/gpu     |        [config](./configs/rotated_deformable-detr_r50_2xb2-50e_dota.py)   |  [last epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_r50_2xb2-50e_dota/epoch_50.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_r50_2xb2-50e_dota/20250909_215737/20250909_215737.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_r50_2xb2-50e_dota/Task1.zip) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.6864634948018359  
ap of each class: plane:0.7885114635406556, baseball-diamond:0.713985806913668, bridge:0.4586041590186169, ground-track-field:0.6259535689632941, small-vehicle:0.708116156118346, large-vehicle:0.7373949971632505, ship:0.847153797544194, tennis-court:0.9051880089836443, basketball-court:0.7866082155611046, storage-tank:0.6569995167778878, soccer-ball-field:0.546815813170951, roundabout:0.5987699946670513, harbor:0.655005767639977, swimming-pool:0.7088824014122034, helicopter:0.558962754552694  
COCO style result:  
AP50: 0.6864634948018359  
AP75: 0.3982598761971252  
mAP: 0.39634542377772963

|  Model  | bs |  Aug  | mAP  | AP50 | AP75 | Param | lr schd | lr | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Deformable<br>DETR +refine  |    R-50   |  -   | 38.60 | 67.55 | 39.00 |  - | 50e |  1e-4   |    4=2gpu*<br>2img/gpu     |        [config](./configs/rotated_deformable-detr_refine_r50_2xb2-50e_dota.py)   |  [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_refine_r50_2xb2-50e_dota/epoch_50.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_refine_r50_2xb2-50e_dota/20250911_104139/20250911_104139.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_refine_r50_2xb2-50e_dota/Task1.zip) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.6755119657279754  
ap of each class: plane:0.7829868044925622, baseball-diamond:0.6666981427431674, bridge:0.417807010717146, ground-track-field:0.59806353394589, small-vehicle:0.697294859807839, large-vehicle:0.7362915837309918, ship:0.8465725513223994, tennis-court:0.8993336551138541, basketball-court:0.7868580476993534, storage-tank:0.7071923681260949, soccer-ball-field:0.5268902759769157, roundabout:0.5750922608428727, harbor:0.629763935872444, swimming-pool:0.6816054790418768, helicopter:0.5802289764862243  
COCO style result:  
AP50: 0.6755119657279754  
AP75: 0.39003965791712775  
mAP: 0.3859510486962555


|  Model  | bs |  Aug  | mAP  | AP50 | AP75 | Param | lr schd | lr | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Deformable<br>DETR ++two-stage  |    R-50   |  -   | 37.79 | 69.55 | 40.60 |  - | 50e |  1e-4   |    4=2gpu*<br>2img/gpu     |        [config](./configs/rotated_deformable-detr_refine_twostage_r50_2xb2-50e_dota.py)   |  [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_refine_twostage_r50_2xb2-50e_dota/epoch_50.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_refine_twostage_r50_2xb2-50e_dota/20250911_113406/20250911_113406.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_refine_twostage_r50_2xb2-50e_dota/Task1.zip) |


This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.6955358764530221  
ap of each class: plane:0.8469243661849398, baseball-diamond:0.7146252093523335, bridge:0.4341599847019934, ground-track-field:0.5987553014820587, small-vehicle:0.7401067283502978, large-vehicle:0.772134456656225, ship:0.8691254418541476, tennis-court:0.9084055873404219, basketball-court:0.8289222958987396, storage-tank:0.758888564867986, soccer-ball-field:0.5431705135906438, roundabout:0.5784628948935202, harbor:0.6393071919941057, swimming-pool:0.6972565992886282, helicopter:0.5027930103392865  
COCO style result:  
AP50: 0.6955358764530221  
AP75: 0.406012015029598  
mAP: 0.3978708527318947


**Train**:

```
 bash tools/dist_train.sh projects/rotated_deformable_detr/configs/rotated_deformable-detr_r50_2xb2-50e_dota.py 2

 bash tools/dist_train.sh projects/rotated_deformable_detr/configs/rotated_deformable-detr_refine_r50_2xb2-50e_dota.py 2

 bash tools/dist_train.sh projects/rotated_deformable_detr/configs/rotated_deformable-detr_refine_twostage_r50_16xb2-50e_dota.py 2
```

**Test**:

```
bash tools/dist_test.sh projects/rotated_deformable_detr/configs/rotated_deformable-detr_r50_2xb2-50e_dota.py work_dirs/rotated_deformable-detr_r50_2xb2-50e_dota/epoch_50.pth 2

bash tools/dist_test.sh projects/rotated_deformable_detr/configs/rotated_deformable-detr_refine_r50_2xb2-50e_dota.py work_dirs/rotated_deformable-detr_refine_r50_2xb2-50e_dota/epoch_50.pth 2

bash tools/dist_test.sh projects/rotated_deformable_detr/configs/rotated_deformable-detr_refine_twostage_r50_16xb2-50e_dota.py work_dirs/rotated_deformable-detr_refine_twostage_r50_16xb2-50e_dota/epoch_50.pth 2
```

**Get Params and FLOPS**:

```
python tools/analysis_tools/get_flops.py projects/rotated_deformable_detr/configs/rotated_deformable-detr_r50_2xb2-50e_dota.py
```

### RSAR

**NOTE: the mAP, AP50, and AP75 are reported on test set, not val set !!!**

|      Backbone      |        Model        |  mAP  |  AP50 | AP75 | Angle  |  lr schd  |  BS  | Config | Download |
| :----------: | :------------: | :---: | :----: | :----: | :----: |:-------: | :--: | :-----: | :---------------: |
| ResNet50<br> (800,800) |   Deformable DETR   | 33.05 | 66.20 | 28.20 | `le90` | `3x` |  4=2gpu*<br>2img/gpu   | [config](./configs/rotated_deformable-detr_refine_twostage_r50_2xb2-3x_rsar.py) | [last ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_refine_twostage_r50_2xb2-3x_rsar/epoch_36.pth) \| <br> [all ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| <br> [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_refine_twostage_r50_2xb2-3x_rsar/20251002_193347/20251002_193347.log) \| [result](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_deformable-detr/rotated_deformable-detr_refine_twostage_r50_2xb2-3x_rsar/20251007_133843/20251007_133843.log) |

## Citation


```latex
@inproceedings{
zhu2021deformable,
title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
author={Xizhou Zhu and Weijie Su and Lewei Lu and Bin Li and Xiaogang Wang and Jifeng Dai},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=gZ9hCDWe6ke}
}
```


## Acknowledgement


