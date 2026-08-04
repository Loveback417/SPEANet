# Preparing CODrone Dataset

>[CODrone: A Comprehensive Oriented Object Detection benchmark for UAV](https://arxiv.org/abs/2504.20032)

>[CODrone github](https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV)




## Download CODrone dataset

The CODrone dataset can be downloaded from [official google](https://drive.google.com/file/d/1FQ6mUaOr_kATDaH7N2bObD5SRRkV7qJy/view) or [modelscope(魔塔)](https://modelscope.cn/datasets/wokaikaixinxin/CODrone/files).

**How to use modelscope(魔塔) to download CODrone**

1) Install `modelscope`

```shell
pip install modelscope
```

2) Download CODrone

```shell
modelscope download --dataset 'wokaikaixinxin/CODrone' --local_dir 'your_local_path'
```

The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── CODrone
│   │   ├── train
│   │   │   │   ├── annfile # (5002 txt) DOTA format annotation
│   │   │   │   ├── labels  # (5002 xml) VOC format annotation
│   │   │   │   ├── images  # (5002 png)
│   │   ├── val
│   │   │   │   ├── annfile # (2000 txt) DOTA format annotation
│   │   │   │   ├── labels  # (2000 xml) VOC format annotation
│   │   │   │   ├── images  # (2000 png)
│   │   ├── test
│   │   │   │   ├── annfile # (3002 txt) DOTA format annotation
│   │   │   │   ├── labels  # (3002 xml) VOC format annotation
│   │   │   │   ├── images  # (3002 png)
```


## Classes of CODrone

The 12 classes.

```
'classes':
(
    'car', 'people', 'motor', 'truck', 'traffic_sign', 'boat', 'traffic_light', 'ship', 'bicycle', 'tricycle', 'bridge', 'bus'
)
```



## Description

We present a comparison between CODrone and other commonly used UAV-based object detection datasets. CODrone significantly expands several key dimensions, including image resolution, object category diversity, and variation in flight altitude and camera angle. For resolution, CODrone employs a 3840 × 2160 high-resolution onboard camera, aligning with the capabilities of modern UAV hardware. In terms of object classes, unlike most existing UAV OOD datasets that focus primarily on vehicles, CODrone includes a more diverse range of categories, thereby increasing the difficulty and realism of the detection task. Furthermore, we explicitly annotate both altitude and camera angle for each image, enabling research into UAV pose-aware perception and related tasks.

[Paper link](https://arxiv.org/abs/2504.20032)

<div align=center>
<img src="https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV/blob/main/img/first.jpg" />
</div>


```bibtex
@article{ye2025more,
  title={More Clear, More Flexible, More Precise: A Comprehensive Oriented Object Detection benchmark for UAV},
  author={Ye, Kai and Tang, Haidi and Liu, Bowen and Dai, Pingyang and Cao, Liujuan and Ji, Rongrong},
  journal={arXiv preprint arXiv:2504.20032},
  year={2025}
}
``` 