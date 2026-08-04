# Preparing DroneVehicle Dataset (TCSVT'2022)

>[Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning](https://arxiv.org/abs/2003.02437)

>[Official Github Repo](https://github.com/VisDrone/DroneVehicle)


## Download DroneVehicle dataset

The DroneVehicle dataset can be downloaded from [modelscope](https://modelscope.cn/datasets/wokaikaixinxin/DroneVehicle).



The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── DroneVehicle
│   │   ├── train
│   │   │   │   ├── trainimg    # 17990 jpg
│   │   │   │   ├── trainimgr   # 17990 jpg
│   │   │   │   ├── trainlabel  # 17990 xml
│   │   │   │   ├── trainlabelr # 17990 xml
│   │   ├── val
│   │   │   │   ├── valimg      # 1469 jpg
│   │   │   │   ├── valimgr     # 1469 jpg
│   │   │   │   ├── vallabel    # 1469 xml
│   │   │   │   ├── vallabelr   # 1469 xml
│   │   ├── test
│   │   │   │   ├── testimg     # 8980 jpg
│   │   │   │   ├── testimgr    # 8980 jpg
│   │   │   │   ├── testlabel   # 8980 xml
│   │   │   │   ├── testlabelr  # 8980 xml
```


## Classes of DroneVehicle

The 5 classes.

```
'classes':
(
    'car', 'freight car', 'truck', 'bus', 'van'
)
```


## Visualize dataset
```
python tools/analysis_tools/browse_dataset.py your_config --out-dir your_path
# for example
# PYTHONPATH=. python tools/analysis_tools/browse_dataset.py projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_rgb.py --output-dir /root/visual_dronevehicle
```

## Description

The DroneVehicle dataset consists of a total of 56,878 images collected by the drone, half of which are RGB images, and the resting are infrared images. We have made rich annotations with oriented bounding boxes for the five categories. Among them, car has 389,779 annotations in RGB images, and 428,086 annotations in infrared images, truck has 22,123 annotations in RGB images, and 25,960 annotations in infrared images, bus has 15,333 annotations in RGB images, and 16,590 annotations in infrared images, van has 11,935 annotations in RGB images, and 12,708 annotations in infrared images, and freight car has 13,400 annotations in RGB images, and 17,173 annotations in infrared image. This dataset is available on the download page.

In DroneVehicle, to annotate the objects at the image boundaries, we set a white border with a width of 100 pixels on the top, bottom, left and right of each image, so that the downloaded image scale is 840 x 712. When training our detection network, we can perform pre-processing to remove the surrounding white border and change the image scale to 640 x 512.


## Some results

| Method  | Modal | Backbone        | AP50  | mAP  | ep. |  bs  | Config  | Download |
| :-----: | :-----:  | :------:        | :--:  | :--: |  :--: | :--: | :-----: | :------: |
|O2-RTDETR|  RGB     | R18vd (640,512) | 70.95 | 44.86|  36   |  16  | [o2_rtdetr_r18vd_2xb8_36e_dv_rgb](../../../projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_rgb.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_rgb/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_rgb/20260323_144051/20260323_144051.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_rgb) |
|O2-RTDETR|  RGB     | R34vd (640,512) | 72.65 | 45.97 |  36   |  16  | [o2_rtdetr_r34vd_2xb8_36e_dv_rgb](../../../projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_rgb.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_rgb/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_rgb/20260323_213225/20260323_213225.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_rgb) |
|O2-RTDETR|  RGB     | R50vd (640,512) | 73.69 | 47.37 |  36   |  16  | [o2_rtdetr_r50vd_2xb8_36e_dv_rgb](../../../projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_rgb.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_rgb/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_rgb/20260323_213937/20260323_213937.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_rgb) |
|O2-RTDETR|  IR     | R18vd (640,512) | 72.73 | 48.29 |  36   |  16  | [o2_rtdetr_r18vd_2xb8_36e_dv_ir](./configs/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_ir.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_ir/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_ir/20260325_004850/20260325_004850.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_ir) |
|O2-RTDETR|  IR     | R34vd (640,512) | 72.82 | 48.48 |  36   |  16  | [o2_rtdetr_r34vd_2xb8_36e_dv_ir](./configs/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_ir.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_ir/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_ir/20260325_114240/20260325_114240.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_ir) |
|O2-RTDETR|  IR     | R50vd (640,512) | 74.58 | 50.34 |  36   |  16  | [o2_rtdetr_r50vd_2xb8_36e_dv_ir](./configs/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_ir.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_ir/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_ir/20260324_150848/20260324_150848.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_ir) |

Note: These models (dronevehicle) are trained on the training set and evaluated on the test set, without using a validation set.

```bibtex
@article{sun2022drone,
  title={Drone-based RGB-infrared cross-modality vehicle detection via uncertainty-aware learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={32},
  number={10},
  pages={6700--6713},
  year={2022},
  publisher={IEEE}
}
``` 

<div align=center>
<img src="https://github.com/VisDrone/DroneVehicle/blob/master/labelsamples.png" />
</div>

<div align=center>
<img src="https://github.com/VisDrone/DroneVehicle/raw/master/dataset_sample.png" />
</div>


