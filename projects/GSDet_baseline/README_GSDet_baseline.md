# GSDet: Gaussian Splatting for Oriented Object Detection

> [GSDet: Gaussian Splatting for Oriented Object Detection](https://www.ijcai.org/proceedings/2025/101)

<!-- [ALGORITHM] -->

## Appendix 
[GSDet appendix](https://github.com/wokaikaixinxin/GSDet/blob/main/GSDet_appendix.pdf)
## Insight

<!--
- 模型能将随机输入学习得到目标。随机输入都可行，其他类型的输入大概率也是可行的。
- 旋转框可以表征为概率分布，如高斯分布。万物都可以表征为不同的概率分布。
- 扩散模型、高斯泼溅等都涉及概率分布，都涉及随机性。
- 架构是decoder-only型，由多层decoder layer堆叠。代码借用了openmmlab 中two-stage类作为父类，事实上，将one-stage或transformer作为父类都可以。
- 这是GSDet的baseline。
-->

- The model can learn targets from random inputs. If random inputs work, other types of inputs are likely feasible as well.  
- Oriented boxes can be represented as probability distributions, such as Gaussian distributions. Everything can be characterized by different probability distributions.  
- Diffusion models, Gaussian splatting, etc., all involve probability distributions and randomness.  
- The architecture is decoder-only, stacked with multiple decoder layers. The code borrows the two-stage class from OpenMMLab as the parent class, though using one-stage or Transformer as the parent is also possible.
- This is the baseline of GSDet.

## New

<!-- &#x2705; 旋转卡壳算法-求点集的最小外接矩形 --> 
&#x2705; Rotating Calipers Algorithm - Finding the Minimum Bounding Rectangle of a Point Set

## Abstract

<div align=center>
<img src="https://github.com/wokaikaixinxin/GSDet/raw/main/GSDet_overview.png" width="800"/>
</div>

Oriented object detection has advanced with the development of convolutional neural networks (CNNs) and transformers. However, modern detectors still rely on predefined object candidates, such as anchors in CNN-based methods or queries in transformer-based methods, which struggle to capture spatial information effectively. To address the limitations, we propose GSDet, a novel framework that formulates oriented object detection as Gaussian splatting. Specifically, our approach performs detection within a 3D feature space constructed from image features, where 3D Gaussians are employed to represent oriented objects. These 3D Gaussians are projected onto the image plane to form 2D Gaussians, which are then transformed into oriented boxes. Furthermore, we optimize the mean, anisotropic covariance, and confidence scores of these randomly initialized 3D Gaussians, using a decoder that incorporates 3D Gaussian sampling. Moreover, our method exhibits flexibility, enabling adaptive control and a dynamic number of Gaussians during inference. Experiments on 3 datasets indicate that GSDet achieves AP50 gains of 0.7% on DIOR-R, 0.3% on DOTA-v1.0, and 0.55% on DOTA-v1.5 when evaluated with adaptive control and outperforms mainstream detectors.

## Results and models

**RSAR**
|      Backbone      |        Model        |  mAP  |  AP50 | AP75 | Angle  |  lr schd  |  BS  | Config | Download |
| :----------: | :------------: | :---: | :----: | :----: | :----: |:-------: | :--: | :-----: | :---------------: |
| *Random* |  |  |  | | |  | |  |
| ResNet50<br> (800,800) |  GSDet_base     | 35.58 | 68.00 | 33.50 | `le90` | `1x` |  4=2gpu*<br>2img/gpu   | [GSDet_r50_b900_h2h4<br>_h2r1_r2r1_1x_rsar.py](./configs/GSDet_r50_b900_h2h4_h2r1_r2r1_1x_rsar.py) | [last ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_1x_rsar/epoch_12.pth) \| <br> [all ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| <br> [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_1x_rsar/20250712_230231.log) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_1x_rsar/20250714_164452.log) |

**NOTE: For the RSAR dataset, the reported mAP, AP50, and AP75 are evaluated on the test set, not the validation set !!!**


**FAIR1M 1.0**

|      Backbone      |     mAP  | Scale | Angle  |  lr schd  |  BS  | Config | Download |
| :----------: |  :---: | :----: |  :----: |:-------: | :--: | :-----: | :---------------: |
| ResNet50<br> (1024,1024) |  38.21 | single<br>scale | `le90` | `2x` |  4=2gpu*<br>2img/gpu   | [GSDet_r50_b900_h2h4<br>_h2r1_r2r1_2x_fair1m1.0.py](./configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_fair1m1.0.py) | [last ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_fair1m1.0/epoch_24.pth) \| <br> [all ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| <br> [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_fair1m1.0/20250806_015925.log) \| [result](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_fair1m1.0/Task1.zip) |

The accuracy of each category of the FAIR1M1.0 dataset:

| mAP | Boeing737 | Boeing747 | Boeing777 | Boeing787 | C919 | A220 | A321 | A330 | A350 | ARJ21 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 38.2148 | 39.5562 | 80.0102 | 14.8915 | 46.6622 | 24.008 | 45.3125 | 61.6583 | 67.3039 | 66.7712 | 30.8361 |
|  | **Passenger Ship** | **Motorboat** | **Fishing Boat** | **Tugboat** | **Engineering Ship** | **Liquid Cargo Ship** | **Dry Cargo Ship** | **Warship** |     |       |
|  | 14.8353 | 53.8367 | 9.6549 | 34.7008 | 14.4206 | 22.7274 | 40.6588 | 34.6612 |     |      | 
|  | **Small Car** | **Bus** | **Cargo Truck** | **Dump Truck** | **Van** | **Trailer** | **Tractor** | **Excavator** | **Truck Tractor** | |
|  | 66.1281   | 14.2996 | 38.19 | 44.8073 | 64.0166 | 8.8786  | 2.563 | 9.6816 | 0.1855 | |
|  | **Basketball Court** | **Tennis Court** | **Football Field** | **Baseball Field** | |  |  |  |  |  |
|  | 38.3167          | 80.818       | 58.5274        | 88.2552        | |  |  |  |  |  |
|  | **Intersection** | **Roundabout** | **Bridge** |     |       |       |       |       |       |      |
|  | 46.2169      | 7.6276     | 28.2851  |     |       |       |       |       |       |      |

**DOTA1.0**


|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 47.69 | 73.88  |   52.17    |   `le90`   |      `2x`      |  -  | 4=2gpu*<br>2img/gpu      | [GSDet_r50_b900_h2h4<br>_h2r1_r2r1_2x_dotav1.0.py](./configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.0.py) | [all clpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.0/20250703_225830/20250703_225830.log) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.0/Task1.zip) \| [last ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.0/epoch_24.pth) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7387758407976989  
ap of each class:   
plane:0.8850394908129608, baseball-diamond:0.7901977747706781, bridge:0.5057504797851325, ground-track-field:0.7170557083792797, small-vehicle:0.7852998720627247, large-vehicle:0.8320931918879327, ship:0.8824600057919189, tennis-court:0.9084433118119101, basketball-court:0.8425974293353717, storage-tank:0.8032986565311625, soccer-ball-field:0.5587179317542539, roundabout:0.5704119054786441, harbor:0.7469883448696094, swimming-pool:0.6787017814257459, helicopter:0.5745817272681583  
COCO style result:  
AP50: 0.7387758407976989  
AP75: 0.521672244119724  
mAP: 0.47687469541788785


**DOTA1.5**


|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 41.60 | 67.76  |   42.76    |   `le90`   |      `2x`      |  -  | 4=2gpu*<br>2img/gpu      | [GSDet_r50_b900_h2h4<br>_h2r1_r2r1_2x_dotav1.5.py](./configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.5.py) | [last ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.5/epoch_24.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.5/20250801_195757.log) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.5/Task1.zip) \| [all ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.6775932538835394  
ap of each class: plane:0.8014766903362498, baseball-diamond:0.7366690430078225, bridge:0.4942253156296784, ground-track-field:0.6994971123660537, small-vehicle:0.5835555375217742, large-vehicle:0.7801513009007439, ship:0.8747843174596853, tennis-court:0.9086324338485723, basketball-court:0.7794249146701522, storage-tank:0.7514437748549189, soccer-ball-field:0.5862729165944337, roundabout:0.6336968107733043, harbor:0.6983827136287671, swimming-pool:0.7157469877864249, helicopter:0.6210445202896426, container-crane:0.17648767246840794  
COCO style result:  
AP50: 0.6775932538835394  
AP75: 0.42759762043819644  
mAP: 0.4159809401584041

**DIOR-R**


|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,800,200) | 46.34 | 69.80  |   48.16    |   `le90`   |      `2x`      |  -  | 4=2gpu*<br>2img/gpu      | [GSDet_r50_b900_h2h4<br>_h2r1_r2r1_2x_dior.py](./configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior/20241208_124846/20241208_124846.log) \| [results](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior/20250704_182754/20250704_182754.log) |


---------------iou_thr: 0.5---------------

| class                   | gts   | dets   | recall  | ap      |
|-------------------------|-------|--------|---------|---------|
| airplane                | 8212  | 146038 | 0.88894 | 0.78397 |
| airport                 | 666   | 93561  | 0.93393 | 0.60829 |
| baseballfield           | 3434  | 115193 | 0.88701 | 0.76466 |
| basketballcourt         | 2146  | 257083 | 0.94688 | 0.87331 |
| bridge                  | 2589  | 507543 | 0.76477 | 0.48336 |
| chimney                 | 1031  | 128298 | 0.87294 | 0.77658 |
| expressway-service-area | 1085  | 86583  | 0.95668 | 0.86513 |
| expressway-toll-station | 688   | 133752 | 0.86192 | 0.73194 |
| dam                     | 538   | 122793 | 0.93866 | 0.45327 |
| golffield               | 575   | 118450 | 0.94435 | 0.77543 |
| groundtrackfield        | 1885  | 115998 | 0.97082 | 0.79884 |
| harbor                  | 3105  | 211913 | 0.69147 | 0.40581 |
| overpass                | 1782  | 210588 | 0.81481 | 0.60808 |
| ship                    | 35186 | 179220 | 0.90243 | 0.83317 |
| stadium                 | 672   | 58144  | 0.95982 | 0.76221 |
| storagetank             | 23361 | 235388 | 0.80382 | 0.72764 |
| tenniscourt             | 7343  | 139576 | 0.91924 | 0.86866 |
| trainstation            | 509   | 74991  | 0.91945 | 0.57726 |
| vehicle                 | 26640 | 872233 | 0.70653 | 0.55803 |
| windmill                | 2998  | 118197 | 0.88559 | 0.70465 |
| mAP                     |       |        |         | 0.69801 |


DIOR-R/mAP: 0.4634  DIOR-R/AP50: 0.6980  DIOR-R/AP55: 0.6686  DIOR-R/AP60: 0.6336  DIOR-R/AP65: 0.5934  DIOR-R/AP70: 0.5386  DIOR-R/AP75: 0.4816  DIOR-R/AP80: 0.4088  DIOR-R/AP85: 0.3207  DIOR-R/AP90: 0.2123  DIOR-R/AP95: 0.0781

## Citation

```
@inproceedings{ijcai2025p101,
  title     = {GSDet: Gaussian Splatting for Oriented Object Detection},
  author    = {Ding, Zeyu and Zhao, Jiaqi and Zhou, Yong and Du, Wen-liang and Zhu, Hancheng and Yao, Rui},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {900--908},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/101},
  url       = {https://doi.org/10.24963/ijcai.2025/101},
}
```
