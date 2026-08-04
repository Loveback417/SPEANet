# Preparing ReCon1M Dataset

>[ReCon1M: A Large-Scale Benchmark Dataset for Relation Comprehension in Remote Sensing Imagery](https://arxiv.org/pdf/2406.06028)

>[ReCon1M project page](https://recon1m-dataset.github.io/)

>[ReCon1M IEEE TGRS](https://ieeexplore.ieee.org/document/11082556)


## Download ReCon1M dataset

The STAR dataset can be downloaded from  [official baidu netdist](https://pan.baidu.com/s/1B-DLJDN3YertZyp8E7UtuQ) Access code: mcq6 or [modelscope(魔塔)](https://www.modelscope.cn/datasets/wokaikaixinxin/ReCon1M/files).

**How to use modelscope(魔塔) to download ReCon1M**

1) Install `modelscope`

```shell
pip install modelscope
```

2) Download ReCon1M

```shell
modelscope download --dataset 'wokaikaixinxin/ReCon1M' --local_dir 'your_local_path'
```

The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── ReCon1M
│   │   ├── dataset_split.json (including train 11131, val 3710, test 7421)
│   │   ├── images (22262 png)
│   │   ├── labelTxt (22262 txt)
```


## split ReCon1M dataset

Please crop the original images into 800x800 patches with an overlap of 400 by run

train set

```shell
python tools/data/recon1m/split/img_split.py --base-json tools/data/recon1m/split/split_configs/ss_train.json
```

val set

```shell
 python tools/data/recon1m/split/img_split.py --base-json tools/data/recon1m/split/split_configs/ss_val.json
```

test set

```shell
python tools/data/recon1m/split/img_split.py --base-json tools/data/recon1m/split/split_configs/ss_test.json
```

trainval set
```shell
python tools/data/recon1m/split/img_split.py --base-json tools/data/recon1m/split/split_configs/ss_trainval.json
```

trainvaltest set
```shell
python tools/data/recon1m/split/img_split.py --base-json tools/data/recon1m/split/split_configs/ss_trainvaltest.json
```

Please update the `img_dirs` and `ann_dirs` in json.

The new data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── split_ss_recon1m
│   │   ├── trainval
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train
│   │   │   ├── images (36879 png)
│   │   │   ├── annfiles (36879 txt)
│   │   ├── val
│   │   │   ├── images (13120 png)
│   │   │   ├── annfiles (13120 txt)
│   │   ├── test
│   │   │   ├── images (25025 png)
│   │   │   ├── annfiles (25025 txt)
```

Please change `data_root` in `configs/_base_/datasets/recon1m.py` to `data/split_ss_recon1m/`.

## Classes of ReCon1M

The 60 classes.

```
'classes':
(
    'airplane', 'airport', 'baseball-field', 'basketball-court', 'block',  
    'boarding_bridge', 'bridge', 'building', 'bus', 'cargo',  
    'cargo-truck', 'chimney', 'construction-site', 'container', 'control-tower',  
    'crane', 'dam', 'dry-cargo-ship', 'dump-truck', 'engineering-ship',  
    'excavator', 'exhaust-fan', 'expressway-service-area', 'factory', 'farmland',  
    'fishing-boat', 'football-field', 'gas-station', 'greenbelt', 'harbor',  
    'helicopter-apron', 'intersection', 'liquid-cargo-ship', 'locomotive', 'motorboat',  
    'other-ship', 'other-vehicle', 'parking-lot', 'passenger-ship', 'pool',  
    'railway', 'road', 'roundabout', 'runway', 'small-car',  
    'smoke', 'solar-panel', 'stadium', 'storage-tank', 'storage-tank-group',  
    'tennis-court', 'terminal', 'tractor', 'trailer', 'train-carriage',  
    'truck-tractor', 'tugboat', 'van', 'warship', 'water'
)
```



## Description

ReCon1M is the first million-level relation annotation dataset specifically designed for Scene Graph Generation (SGG) in remote sensing imagery. our dataset is built upon FAIR1M and comprises 22,262 images. It includes annotations for 873,761 object bounding boxes across 60 categories, and 1,052,244 relation triplets across 59 categories based on these bounding boxes.

[Paper link](https://arxiv.org/pdf/2406.06028)

<div align=center>
<img src="https://simg.baai.ac.cn/papers/converted_page_79aa4d769f389bde979f986e27ac5260-03.jpg?x-oss-process=image/format,jpg/interlace,1" />
</div>


```bibtex
@ARTICLE{11082556,
  author={Yan, Qiwei and Deng, Chubo and Liu, Chenglong and Hou, Zhongyan and Liu, Xiaorui and Jiang, Yi and Lu, Wanxuan and Yao, Fanglong and Liu, Xiaoyu and Hao, Lingxiang and Yu, Hongfeng and Sun, Xian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ReCon1M: A Large-Scale Benchmark Dataset for Relation Comprehension in Remote Sensing Imagery}, 
  year={2025},
  volume={63},
  number={},
  pages={1-22},
  keywords={Remote sensing;Visualization;Annotations;Cognition;Semantics;Benchmark testing;Training;Artificial intelligence;Data mining;Computational modeling;Benchmark dataset;relation comprehension;remote sensing;scene graph generation (SGG)},
  doi={10.1109/TGRS.2025.3589986}}
```