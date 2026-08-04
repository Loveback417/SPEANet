# Rotated FCOS

> [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143882011-45b234bc-d04b-4bbe-a822-94bec057ac86.png"/>
</div>

We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction
fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3,
and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well
as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation
related to anchor boxes such as calculating overlapping during training. More importantly, we also avoid all
hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. With the
only post-processing non-maximum suppression (NMS), FCOS with ResNeXt-64x4d-101 achieves 44.7% in AP with single-model
and single-scale testing, surpassing previous one-stage detectors with the advantage of being much simpler. For the
first time, we demonstrate a much simpler and flexible detection framework achieving improved detection accuracy. We
hope that the proposed FCOS framework can serve as a simple and strong alternative for many other instance-level tasks.

## Results and Models

DIOR

|      Backbone      |        Model        |  mAP  |  AP50 | AP75 | Angle  |  lr schd  |  BS  | Config | Download |
| :----------: | :------------: | :---: | :----: | :----: | :----: |:-------: | :--: | :-----: | :---------------: |
| ResNet50<br> (800,800) |    Rotated-<br>FCOS    | 36.90 | 60.40 | 37.00 |`le90` |   `1x`    |  2  | [rotated-fcos-le90_r50_fpn_1x_dior.py](./rotated-fcos-le90_r50_fpn_1x_dior.py) | [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RotatedFCOS/rotated-fcos-le90_r50_fpn_1x_dior/epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RotatedFCOS/rotated-fcos-le90_r50_fpn_1x_dior/20250720_011435/20250720_011435.log) \| <br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) |

RSAR

**NOTE: the mAP, AP50, and AP75 are reported on test set, not val set !!!**

|      Backbone      |        Model        |  mAP  |  AP50 | AP75 | Angle  |  lr schd  |  BS  | Config | Download |
| :----------: | :------------: | :---: | :----: | :----: | :----: |:-------: | :--: | :-----: | :---------------: |
| ResNet50<br> (800,800) |    Rotated-<br>FCOS   | 34.22 | 66.66 | 31.45 |`le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [rotated-fcos-le90_r50_fpn_1x_rsar.py](./rotated-fcos-le90_r50_fpn_1x_rsar.py) | [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/rotated-fcos-le90_r50_fpn_1x_rsar/rotated-fcos-le90_r50_fpn_1x_rsar_epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/rotated-fcos-le90_r50_fpn_1x_rsar/rotated-fcos-le90_r50_fpn_1x_rsar.json) |

DOTA1.0

|         Backbone         |  AP50  | Angle | Separate Angle | Tricks | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                     Configs                                                     |                                                                                                                                                                                   Download                                                                                                                                                                                   |
| :----------------------: | :---: | :---: | :------------: | :----: | :-----: | :------: | :------------: | :-: | :--------: | :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 70.70 | le90  |       Y        |   Y    |   1x    |   4.18   |      26.4      |  -  |     2      |              [rotated-fcos-hbox-le90_r50_fpn_1x_dota](./rotated-fcos-hbox-le90_r50_fpn_1x_dota.py)              |       [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90_20220409_023250.log.json)       |
| ResNet50 (1024,1024,200) | 71.28 | le90  |       N        |   Y    |   1x    |   4.18   |      25.9      |  -  |     2      |                   [rotated-fcos-le90_r50_fpn_1x_dota](./rotated-fcos-le90_r50_fpn_1x_dota.py)                   |                           [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90/rotated_fcos_r50_fpn_1x_dota_le90-d87568ed.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90/rotated_fcos_r50_fpn_1x_dota_le90_20220413_163526.log.json)                           |
| ResNet50 (1024,1024,200) | 71.76 | le90  |       Y        |   Y    |   1x    |   4.23   |      25.7      |  -  |     2      | [rotated-fcos-hbox-le90_r50_fpn_csl-gaussian_1x_dota](./rotated-fcos-hbox-le90_r50_fpn_csl-gaussian_1x_dota.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90-4e044ad2.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90_20220409_080616.log.json) |
| ResNet50 (1024,1024,200) | 71.89 | le90  |       N        |   Y    |   1x    |   4.18   |      26.2      |  -  |     2      |               [rotated-fcos-le90_r50_fpn_kld_1x_dota](./rotated-fcos-le90_r50_fpn_kld_1x_dota.py)               |                   [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_kld_r50_fpn_1x_dota_le90/rotated_fcos_kld_r50_fpn_1x_dota_le90-ecafdb2b.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_kld_r50_fpn_1x_dota_le90/rotated_fcos_kld_r50_fpn_1x_dota_le90_20220409_202939.log.json)                   |

**Notes:**

- `MS` means multiple scale image split.
- `RR` means random rotation.
- `Rotated IoU Loss` need mmcv version 1.5.0 or above.
- `Separate Angle` means angle loss is calculated separately.
  At this time bbox loss uses horizontal bbox loss such as `IoULoss`, `GIoULoss`.
- Tricks means setting `norm_on_bbox`, `centerness_on_reg`, `center_sampling` as `True`.
- Inf time was tested on a single RTX3090.

## Citation

```
@inproceedings{tian2019fcos,
  title={Fcos: Fully convolutional one-stage object detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={9627--9636},
  year={2019}
}
```
