# YOLOX

> [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

<!-- [ALGORITHM] -->

## Abstract

In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144001736-9fb303dd-eac7-46b0-ad45-214cfa51e928.png" width="800"/>
</div>

## Results and models

**DOTA**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Scale | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  YOLOX-t <br> (1024,1024,200) | 44.96 | 73.38  |  46.58  |   `le90`   |      `300e`      |  single <br> scale  | 8=2gpu*<br>4img/gpu   | [rotated_yolox_t_300e_dota_le90.py](./configs/rotated_yolox_t_300e_dota_le90.py) | [all ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_t_300e_dota_le90/20250815_193736/20250815_193736.log) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_t_300e_dota_le90/Task1.zip) \| [last ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_t_300e_dota_le90/epoch_300.pth) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7338344506577592  
ap of each class: plane:0.8695257026303225, baseball-diamond:0.8129688706912487, bridge:0.4631930789555658, ground-track-field:0.6506972221957175, small-vehicle:0.7812533539001271, large-vehicle:0.8500794894166496, ship:0.8893545723380502, tennis-court:0.9084604418159161, basketball-court:0.8649783554121606, storage-tank:0.8599190783625095, soccer-ball-field:0.546212255486449, roundabout:0.6147867938360598, harbor:0.667000550182095, swimming-pool:0.7090841981421707, helicopter:0.5200027965013474  
COCO style result:  
AP50: 0.7338344506577592  
AP75: 0.46576575276691484  
mAP: 0.44955888646925707

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Scale | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  YOLOX-s <br> (1024,1024,200) | 46.27 | 74.94  |   49.49    |   `le90`   |      `300e`      |  single <br> scale  | 8=2gpu*<br>4img/gpu   | [rotated_yolox_s_300e_dota_le90.py](./configs/rotated_yolox_s_300e_dota_le90.py) | [all ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_s_300e_dota_le90/20250809_224227.log) \| [result](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_s_300e_dota_le90/Task1.zip) \| [last ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_s_300e_dota_le90/epoch_300.pth) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.749433522358599  
ap of each class: plane:0.8642951917278126, baseball-diamond:0.8293671917667135, bridge:0.4792212695427707, ground-track-field:0.6550645193578333, small-vehicle:0.7802050660379458, large-vehicle:0.8497052490370042, ship:0.8906889639787817, tennis-court:0.9083821116390618, basketball-court:0.8798755073052649, storage-tank:0.8600542432660736, soccer-ball-field:0.5829021529662977, roundabout:0.6362592197722562, harbor:0.7323143096770864, swimming-pool:0.6980757951055564, helicopter:0.5950920441985256  
COCO style result:  
AP50: 0.749433522358599  
AP75: 0.4948977689064668  
mAP: 0.46267007542788247


|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Scale | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  YOLOX-m <br> (1024,1024,200) | 49.51 | 76.70 | 54.95 |   `le90`   |      `300e`      |  single <br> scale  | 8=2gpu*<br>4img/gpu   | [rotated_yolox_m_300e_dota_le90.py](./configs/rotated_yolox_m_300e_dota_le90.py) | [all ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_m_300e_dota_le90/20251128_095408/20251128_095408.log) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_m_300e_dota_le90/Task1.zip) \| [last ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_m_300e_dota_le90/epoch_300.pth) |


This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7669546721225817  
ap of each class: plane:0.8684170849180731, baseball-diamond:0.8639517728968095, bridge:0.5427458156752936, ground-track-field:0.7003597644813377, small-vehicle:0.8103007833301753, large-vehicle:0.8567521184687866, ship:0.8927121951978374, tennis-court:0.907139950954836, basketball-court:0.8632436842044986, storage-tank:0.8617119349667725, soccer-ball-field:0.603056391874901, roundabout:0.6552074291399157, harbor:0.7566053782569716, swimming-pool:0.7267901299600665, helicopter:0.5953256475124526  
COCO style result:  
AP50: 0.7669546721225817  
AP75: 0.5494557409860242  
mAP: 0.4951261405231623

**Note**: Since we configured `auto_scale_lr = dict(base_batch_size=64, enable=True)` in `rotated_yolox_s_300e_dota_le90.py`, the learning rate will be automatically scaled. Specifically, with our actual batch size of `8=2gpu*4img/gpu`, the learning rate is automatically reduced to 1/8 of the original value.

**DIOR**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Scale | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  YOLOX-s <br> (800, 800) | - | 64.38  | 44.38 |   `le90`   |      `300e`      |  single <br> scale  | 8=2gpu*<br>4img/gpu   | [rotated_yolox_s_300e_dior_le90.py](./configs/rotated_yolox_s_300e_dior_le90.py) | [all ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_s_300e_dior_le90/20250809_212335.log)  \| [best ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_s_300e_dior_le90/epoch_290.pth) |

**Note**: The results are reported by `epoch_290.pth`

**Note**: Since we configured `auto_scale_lr = dict(base_batch_size=64, enable=True)` in `rotated_yolox_s_300e_dior_le90.py`, the learning rate will be automatically scaled. Specifically, with our actual batch size of `8=2gpu*4img/gpu`, the learning rate is automatically reduced to 1/8 of the original value.

## Citation

```latex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
