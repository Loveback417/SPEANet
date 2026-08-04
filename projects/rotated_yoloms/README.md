# YOLO-MS (TPAMI 2025)

> IEEE Link [YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection](https://ieeexplore.ieee.org/document/10872821)

> ArXiv Link [YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection](https://arxiv.org/abs/2308.05480)

> Official Code [Link](https://github.com/FishAndWasabi/YOLO-MS)

<!-- [ALGORITHM] -->

## Abstract

We aim at providing the object detection community with an efficient and performant object detector, termed YOLO-MS. The core design is based on a series of investigations on how multi-branch features of the basic block and convolutions with different kernel sizes affect the detection performance of objects at different scales. The outcome is a new strategy that can significantly enhance multi-scale feature representations of real-time object detectors. To verify the effectiveness of our work, we train our YOLO-MS on the MS COCO dataset from scratch without relying on any other large-scale datasets, like ImageNet or pre-trained weights. Without bells and whistles, our YOLO-MS outperforms the recent state-of-the-art real-time object detectors, including YOLO-v7, RTMDet, and YOLO-v8. Taking the XS version of YOLO-MS as an example, it can achieve an AP score of 42+% on MS COCO, which is about 2% higher than RTMDet with the same model size. Furthermore, our work can also serve as a plug-and-play module for other YOLO models. Typically, our method significantly advances the APs, APl, and AP of YOLOv8-N from 18%+, 52%+, and 37%+ to 20%+, 55%+, and 40%+, respectively, with even fewer parameters and MACs.

<div align=center>
<img src="https://i-blog.csdnimg.cn/img_convert/a6ae33d000432d002f498bc79f75cc2b.png" height="360"/>
</div>


## YOLO-MS Previous Version

### DOTA-v1.0

|  Model  | pretrain |  Aug  | mAP  | AP50 | AP75 | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLO-MS-XS<br>previous |    COCO   |  RR   | 49.30 | 75.88 | 53.81 |   4.48  |  22.24   |    8=2gpu*<br>4img/gpu     |        [config](./configs/yoloms-xs_syncbn_fast_2xb4-36e_dota_previous.py)   |  [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-xs_syncbn_fast_2xb4-36e_dota_previous/epoch_36.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-xs_syncbn_fast_2xb4-36e_dota_previous/20250904_152409/20250904_152409.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-xs_syncbn_fast_2xb4-36e_dota_previous/Task1.zip) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.758775701719842  
ap of each class: plane:0.8944230339878622, baseball-diamond:0.842146703095921, bridge:0.5263208048478161, ground-track-field:0.7407278936319676, small-vehicle:0.8176438257248919, large-vehicle:0.8251346237459852, ship:0.8870831668864025, tennis-court:0.9090233186887464, basketball-court:0.8765827014695793, storage-tank:0.8650064504374552, soccer-ball-field:0.6304304519950606, roundabout:0.5883983913819532, harbor:0.7650530632537562, swimming-pool:0.6918867394821042, helicopter:0.5217743571681305  
COCO style result:  
AP50: 0.758775701719842  
AP75: 0.5380872925688465  
mAP: 0.49298488279895897

|  Model  | pretrain |  Aug  | mAP  | AP50 | AP75 | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLO-MS-S<br>previous |    COCO   |  RR   | 50.46 | 76.55 | 55.65 |   8.03    |  39.68   |    8=2gpu*<br>4img/gpu     |        [config](./configs/yoloms-s_syncbn_fast_2xb4-36e_dota_previous.py)        |                         [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-s_syncbn_fast_2xb4-36e_dota_previous/epoch_36.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-s_syncbn_fast_2xb4-36e_dota_previous/20250903_223206/20250903_223206.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-s_syncbn_fast_2xb4-36e_dota_previous/Task1.zip)                           |


This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.765460659633011  
ap of each class: plane:0.8936621878842959, baseball-diamond:0.8392404068426194, bridge:0.5326113192062201, ground-track-field:0.7353638093780249, small-vehicle:0.8174603067689339, large-vehicle:0.8494447803894849, ship:0.889119814337507, tennis-court:0.90887829545426, basketball-court:0.8833521616607203, storage-tank:0.8785790124897477, soccer-ball-field:0.619967107911772, roundabout:0.5697002557520482, harbor:0.7753673808345287, swimming-pool:0.7565379369479264, helicopter:0.5326251186370747  
COCO style result:  
AP50: 0.765460659633011  
AP75: 0.5564636731501145  
mAP: 0.5045991125776231



|  Model  | pretrain |  Aug  | mAP  | AP50 | AP75 | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLO-MS<br>previous |    COCO   |  RR   | - | - | - | 21.96   | 102.34  |    8=2gpu*<br>4img/gpu     |        [config](./configs/yoloms_syncbn_fast_2xb4-36e_dota_previous.py)   |  last epoch \| log \|<br> all epoch \| result |

Sorry, I don't have enough GPU resources to run `yoloms_syncbn_fast_2xb4-36e_dota_previous.py`.

**Train**:

```
bash tools/dist_train.sh projects/rotated_yoloms/configs/yoloms-xs_syncbn_fast_2xb4-36e_dota_previous.py 2

bash tools/dist_train.sh projects/rotated_yoloms/configs/yoloms-s_syncbn_fast_2xb4-36e_dota_previous.py 2

bash tools/dist_train.sh projects/rotated_yoloms/configs/yoloms_syncbn_fast_2xb4-36e_dota_previous.py 2
```

**Test**:

```
bash tools/dist_test.sh projects/rotated_yoloms/configs/yoloms-xs_syncbn_fast_2xb4-36e_dota_previous.py work_dirs/yoloms-xs_syncbn_fast_2xb4-36e_dota_previous/epoch_36.pth 2

bash tools/dist_test.sh projects/rotated_yoloms/configs/yoloms-s_syncbn_fast_2xb4-36e_dota_previous.py work_dirs/yoloms-s_syncbn_fast_2xb4-36e_dota_previous/epoch_36.pth 2

bash tools/dist_test.sh projects/rotated_yoloms/configs/yoloms_syncbn_fast_2xb4-36e_dota_previous.py work_dirs/yoloms_syncbn_fast_2xb4-36e_dota_previous/epoch_36.pth 2
```

**Get Params and FLOPS**:

```
python tools/analysis_tools/get_flops.py projects/rotated_yoloms/configs/yoloms-xs_syncbn_fast_2xb4-36e_dota_previous.py

python tools/analysis_tools/get_flops.py projects/rotated_yoloms/configs/yoloms-s_syncbn_fast_2xb4-36e_dota_previous.py

python tools/analysis_tools/get_flops.py projects/rotated_yoloms/configs/yoloms_syncbn_fast_2xb4-36e_dota_previous.py
```


### DIOR-R

|  Model  | pretrain |  Aug  | mAP  | AP50 | AP75 | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLO-MS-XS<br>previous |    COCO   |  -   | 45.42 | 66.65 | 47.83 |   4.48  |  22.24   |    8=2gpu*<br>4img/gpu     |        [config](./configs/yoloms-xs_syncbn_fast_2xb4-3x_dior_previous.py)   |  [last epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-xs_syncbn_fast_2xb4-3x_dior_previous/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-xs_syncbn_fast_2xb4-3x_dior_previous/20251107_233434/20251107_233434.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files)  |



## YOLO-MS


### DOTA-v1.0

|  Model  | pretrain |  Aug  | mAP  | AP50 | AP75 | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLO-MS-XS |    COCO   |  RR   |  49.16 | 76.65 | 52.41 |  5.09  |  22.02  |    8=2gpu*<br>4img/gpu     |        [config](./configs/yoloms-xs_syncbn_fast_2xb4-36e_dota.py)   |  [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-xs_syncbn_fast_2xb4-36e_dota/epoch_36.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-xs_syncbn_fast_2xb4-36e_dota/20250906_215214/20250906_215214.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-xs_syncbn_fast_2xb4-36e_dota/Task1.zip) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7665371340189844  
ap of each class: plane:0.8916935332940078, baseball-diamond:0.814094312395071, bridge:0.5084526408289133, ground-track-field:0.7269527793045261, small-vehicle:0.8115511649453246, large-vehicle:0.8409107048135034, ship:0.8869555205944052, tennis-court:0.9089586776859506, basketball-court:0.8770591478285379, storage-tank:0.8630198381343265, soccer-ball-field:0.6132339176153921, roundabout:0.6506499435928538, harbor:0.7620437751544, swimming-pool:0.811402535938293, helicopter:0.5310785181592597  
COCO style result:  
AP50: 0.7665371340189844  
AP75: 0.5240587149252313  
mAP: 0.49159514215398037

|  Model  | pretrain |  Aug  | mAP  | AP50 | AP75 | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLO-MS-S |    COCO   |  RR   | 50.43  | 77.22 | 54.13 |  8.72  |  38.07  |    8=2gpu*<br>4img/gpu     |        [config](./configs/yoloms-s_syncbn_fast_2xb4-36e_dota.py)   |  [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-s_syncbn_fast_2xb4-36e_dota/epoch_36.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-s_syncbn_fast_2xb4-36e_dota/20250907_125043/20250907_125043.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-s_syncbn_fast_2xb4-36e_dota/Task1.zip) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7722389970489156  
ap of each class: plane:0.8950727591139864, baseball-diamond:0.8289477391782069, bridge:0.5280990505176452, ground-track-field:0.747512721794236, small-vehicle:0.8115115979676332, large-vehicle:0.837990844483579, ship:0.8858722167760349, tennis-court:0.9089555269748868, basketball-court:0.8739596275006111, storage-tank:0.8621823159844082, soccer-ball-field:0.6385742669304498, roundabout:0.6627481300027679, harbor:0.7738014402951015, swimming-pool:0.804580969590782, helicopter:0.523775748623405  
COCO style result:  
AP50: 0.7722389970489156  
AP75: 0.5412921155184615  
mAP: 0.5043088492422652


|  Model  | pretrain |  Aug  | mAP  | AP50 | AP75 | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLO-MS |    COCO   |  RR   | -  | - | - |  23.12  |  98.47  |    8=2gpu*<br>4img/gpu     |        [config](./configs/yoloms_syncbn_fast_2xb4-36e_dota.py)   |  last epoch \| log \|<br> all epoch \| result |

Sorry, I don't have enough GPU resources to run `yoloms_syncbn_fast_2xb4-36e_dota.py`

**Train**:

```
bash tools/dist_train.sh projects/rotated_yoloms/configs/yoloms-xs_syncbn_fast_2xb4-36e_dota.py 2

bash tools/dist_train.sh projects/rotated_yoloms/configs/yoloms-s_syncbn_fast_2xb4-36e_dota.py 2

bash tools/dist_train.sh projects/rotated_yoloms/configs/yoloms_syncbn_fast_2xb4-36e_dota.py 2
```

**Test**:

```
bash tools/dist_test.sh projects/rotated_yoloms/configs/yoloms-xs_syncbn_fast_2xb4-36e_dota.py work_dirs/yoloms-xs_syncbn_fast_2xb4-36e_dota/epoch_36.pth 2

bash tools/dist_test.sh projects/rotated_yoloms/configs/yoloms-s_syncbn_fast_2xb4-36e_dota.py work_dirs/yoloms-s_syncbn_fast_2xb4-36e_dota/epoch_36.pth 2

bash tools/dist_test.sh projects/rotated_yoloms/configs/yoloms_syncbn_fast_2xb4-36e_dota.py work_dirs/yoloms_syncbn_fast_2xb4-36e_dota/epoch_36.pth 2
```

**Get Params and FLOPS**:

```
python tools/analysis_tools/get_flops.py  projects/rotated_yoloms/configs/yoloms-xs_syncbn_fast_2xb4-36e_dota.py

python tools/analysis_tools/get_flops.py projects/rotated_yoloms/configs/yoloms-s_syncbn_fast_2xb4-36e_dota.py

python tools/analysis_tools/get_flops.py projects/rotated_yoloms/configs/yoloms_syncbn_fast_2xb4-36e_dota.py
```

**Note**:

1. We follow the latest metrics from the DOTA evaluation server, original voc format mAP is now mAP50.
2. `IN` means ImageNet pretrain, `COCO` means COCO pretrain.
3. By default, DOTA-v1.0 dataset trained with 3x schedule and image size 1024\*1024.

## Citation

```
@article{Chen2025,
  title = {YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-time Object Detection},
  ISSN = {1939-3539},
  url = {http://dx.doi.org/10.1109/TPAMI.2025.3538473},
  DOI = {10.1109/tpami.2025.3538473},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Chen, Yuming and Yuan, Xinbin and Wang, Jiabao and Wu, Ruiqi and Li, Xiang and Hou, Qibin and Cheng, Ming-Ming},
  year = {2025},
  pages = {1–14}
}
```
