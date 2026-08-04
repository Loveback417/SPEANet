# Attention-Based Mean-Max Balance Assignment for Oriented Object Detection in Optical Remote Sensing Images (TGRS 2025)

> [Attention-Based Mean-Max Balance Assignment for Oriented Object Detection in Optical Remote Sensing Images](https://ieeexplore.ieee.org/document/10852329)

> Official Code [Link](https://github.com/promisekoloer/AMMBA)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://i-blog.csdnimg.cn/direct/e6cfdcec0842425cbf84be389dba0aed.png" width="800"/>
</div>

For objects with arbitrary angles in optical remote sensing (RS) images, the oriented bounding box regression task often faces the problem of ambiguous boundaries between positive and negative samples. The statistical analysis of existing label assignment strategies reveals that anchors with low Intersection over Union (IoU) between ground truth (GT) may also accurately surround the GT after decoding. Therefore, this article proposes an attention-based mean-max balance assignment (AMMBA) strategy, which consists of two parts: mean-max balance assignment (MMBA) strategy and balance feature pyramid with attention (BFPA). MMBA employs the mean-max assignment (MMA) and balance assignment (BA) to dynamically calculate a positive threshold and adaptively match better positive samples for each GT for training. Meanwhile, to meet the need of MMBA for more accurate feature maps, we construct a BFPA module that integrates spatial and scale attention mechanisms to promote global information propagation. Combined with S2ANet, our AMMBA method can effectively achieve state-of-the-art performance, with a precision of 80.91% on the DOTA dataset in a simple plug-and-play fashion. Extensive experiments on three challenging optical RS image datasets (DOTA-v1.0, HRSC, and DIOR-R) further demonstrate the balance between precision and speed in single-stage object detectors. Our AMMBA has enough potential to assist all existing RS models in a simple way to achieve better detection performance.


## Results and models


**DOTA-v1.0**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | lr | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :---: | :--------: | :---------------------------------------------: | :-------------------------------: |
| RetinaNet + AMMBA <br> (1024,1024,200) | 42.94 | 72.92  |  44.48  |   le90   |  1x  | -  | 2.5e-3 | 2=1gpu*<br>2img/gpu      | [ammba-le90_r50_fpn_1x_dota.py](./configs/ammba-le90_r50_fpn_1x_dota.py) | [last epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/AMMBA/ammba-le90_r50_fpn_1x_dota/epoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/AMMBA/ammba-le90_r50_fpn_1x_dota/20260114_211741/20260114_211741.log) \| <br> [all epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/AMMBA/ammba-le90_r50_fpn_1x_dota) \| [result](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/AMMBA/ammba-le90_r50_fpn_1x_dota/Task1.zip)|
| S2ANet + AMMBA <br> (1024,1024,200) | 43.47 | 74.57  |  45.12  |   le90   |  1x  | -  | 2.5e-3 | 2=1gpu*<br>2img/gpu      | [s2anet-ammba-le90_r50_fpn_1x_dota.py](./configs/s2anet-ammba-le90_r50_fpn_1x_dota.py) | [last epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/AMMBA/s2anet-ammba-le90_r50_fpn_1x_dota/epoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/AMMBA/s2anet-ammba-le90_r50_fpn_1x_dota/20260114_223648/20260114_223648.log) \| <br> [all epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/AMMBA/s2anet-ammba-le90_r50_fpn_1x_dota) \| [result](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/AMMBA/s2anet-ammba-le90_r50_fpn_1x_dota/Task1.zip)|

Note: This is the **unofficial** checkpoint.

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.729158804477687  
ap of each class: plane:0.8932147923984427, baseball-diamond:0.751149120505701, bridge:0.49859668116761274, ground-track-field:0.6752543913878198, small-vehicle:0.8017523713632771, large-vehicle:0.7861681383781977, ship:0.872651401160053, tennis-court:0.9090236188681786, basketball-court:0.8195624728315555, storage-tank:0.8519478694984839, soccer-ball-field:0.6328976604610127, roundabout:0.633209510994192, harbor:0.657088523402398, swimming-pool:0.6783472934462506, helicopter:0.4765182213021302  
COCO style result:  
AP50: 0.729158804477687  
AP75: 0.4447806538407711  
mAP: 0.4294407961875522  

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7457400869020935  
ap of each class: plane:0.8854353689516588, baseball-diamond:0.7997321018912741, bridge:0.5117294606789453, ground-track-field:0.7209230598521081, small-vehicle:0.7962897706144758, large-vehicle:0.8061758524588448, ship:0.8750618173613448, tennis-court:0.9090909090909093, basketball-court:0.8243371357870869, storage-tank:0.8483852537704649, soccer-ball-field:0.6226729868491679, roundabout:0.6664725396832714, harbor:0.6991668374170186, swimming-pool:0.691249613523133, helicopter:0.5293785956016964  
COCO style result:  
AP50: 0.7457400869020935  
AP75: 0.45118847177150495  
mAP: 0.43467739965832985


**Train**

```
bash tools/dist_train.sh projects/AMMBA/configs/ammba-le90_r50_fpn_1x_dota.py 1
``` 


```
bash tools/dist_train.sh projects/AMMBA/configs/s2anet-ammba-le90_r50_fpn_1x_dota.py 1
```


**Test**
```
bash tools/dist_test.sh projects/AMMBA/configs/ammba-le90_r50_fpn_1x_dota.py work_dirs/ammba-le90_r50_fpn_1x_dota/epoch_12.pth 1
```  


```
bash tools/dist_test.sh projects/AMMBA/configs/s2anet-ammba-le90_r50_fpn_1x_dota.py work_dirs/s2anet-ammba-le90_r50_fpn_1x_dota/epoch_12.pth 1
```


## Citation

```
@ARTICLE{10852329,
  author={Lin, Qifeng and Chen, Nuo and Huang, Haibin and Zhu, Daoye and Fu, Gang and Chen, Chuanxi and Yu, Yuanlong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Attention-Based Mean-Max Balance Assignment for Oriented Object Detection in Optical Remote Sensing Images}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Remote sensing;Detectors;Feature extraction;Object detection;Training;Semantics;Location awareness;Accuracy;Shape;Optical scattering;Attention feature fusion;label assignment;optical remote sensing (RS) images;oriented object detection},
  doi={10.1109/TGRS.2025.3533553}}
```
