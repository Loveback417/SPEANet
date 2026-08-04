# Preparing STAR Dataset

>[STAR: A First-Ever Dataset and A Large-Scale Benchmark for Scene Graph Generation in Large-Size Satellite Imagery](https://arxiv.org/abs/2406.09410)

>[STAR project page](https://linlin-dev.github.io/project/STAR)

>[STAR-MMRotate](https://github.com/VisionXLab/STAR-MMRotate)

>[SGG-ToolKit](https://github.com/Zhuzi24/SGG-ToolKit)


## Download STAR dataset

The STAR dataset can be downloaded from [official hugging face](https://huggingface.co/datasets/Zhuzi24/STAR) or [official baidu netdist](https://pan.baidu.com/s/143LVm6zt-ryGEngltALZtw?pwd=STAR) or [modelscope(魔塔)](https://modelscope.cn/datasets/wokaikaixinxin/STAR/files).

**How to use modelscope(魔塔) to download STAR**

1) Install `modelscope`

```shell
pip install modelscope
```

2) Download STAR

```shell
modelscope download --dataset 'wokaikaixinxin/STAR' --local_dir 'your_local_path'
```

The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── STAR
│   │   ├── train
│   │   │   │   ├── img (771 png, 64GB)
│   │   │   │   ├── object-TXT (771 txt, ~7MB)
│   │   ├── val (The original paper does not use a validation set, so you don't need to download it.)
│   │   │   │   ├── img (238 png, 32GB)
│   │   │   │   ├── object-TXT (238 txt, ~2.5MB)
│   │   ├── test
│   │   │   │   ├── img264 (264 png, 23.1GB)
│   │   │   │   ├── object-TXT (264 txt, ~2.56MB)
```

## split STAR dataset

train set

```shell
python tools/data/star/split/img_split.py --base-json tools/data/star/split/split_configs/ss_train.json
```

test set

```shell
python tools/data/star/split/img_split.py --base-json tools/data/star/split/split_configs/ss_test.json
```

Please update the `img_dirs` and `ann_dirs` in json.

The new data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── split_ss_star
│   │   ├── train
│   │   │   ├── images (70383 png, only 19756 images have gt)
│   │   │   ├── annfiles (70383 txt, only 19756 images have gt)
│   │   ├── test
│   │   │   ├── images (23702 png)
│   │   │   ├── annfiles (23702 txt)
```

Please change `data_root` in `configs/_base_/datasets/star.py` to `data/split_ss_star/`.

## Classes of STAR

The 48 classes.

```
'classes':
(
    'ship', 'boat', 'crane', 'goods_yard', 'tank', 'storehouse', 'breakwater', 'dock',
    'airplane', 'boarding_bridge', 'runway', 'taxiway', 'terminal', 'apron', 'gas_station',
    'truck', 'car', 'truck_parking', 'car_parking', 'bridge', 'cooling_tower', 'chimney',
    'vapor', 'smoke', 'genset', 'coal_yard', 'lattice_tower', 'substation', 'wind_mill',
    'cement_concrete_pavement', 'toll_gate', 'flood_dam', 'gravity_dam', 'ship_lock',
    'ground_track_field', 'basketball_court', 'engineering_vehicle', 'foundation_pit',
    'intersection', 'soccer_ball_field', 'tennis_court', 'tower_crane', 'unfinished_building',
    'arch_dam', 'roundabout', 'baseball_diamond', 'stadium', 'containment_vessel'
)
```

## Description

Scene graph generation (SGG) in satellite imagery (SAI) benefits promoting understanding of geospatial scenarios from perception to cognition. In SAI, objects exhibit great variations in scales and aspect ratios, and there exist rich relationships between objects (even between spatially disjoint objects), which makes it attractive to holistically conduct SGG in large-size very-high-resolution (VHR) SAI. However, there lack such SGG datasets. Due to the complexity of large-size SAI, mining triplets <subject, relationship, object> heavily relies on long-range contextual reasoning. Consequently, SGG models designed for small-size natural imagery are not directly applicable to large-size SAI. This paper constructs a large-scale dataset for SGG in large-size VHR SAI with image sizes ranging from 512 x 768 to 27,860 x 31,096 pixels, named STAR (Scene graph generaTion in lArge-size satellite imageRy), encompassing over 210K objects and over 400K triplets. To realize SGG in large-size SAI, we propose a context-aware cascade cognition (CAC) framework to understand SAI regarding object detection (OBD), pair pruning and relationship prediction for SGG. We also release a SAI-oriented SGG toolkit with about 30 OBD and 10 SGG methods which need further adaptation by our devised modules on our challenging STAR dataset.

[Paper link](https://arxiv.org/abs/2406.09410)

<div align=center>
<img src="https://github.com/VisionXLab/STAR-MMRotate/raw/main/demo/star.jpg" />
</div>

## Oriented Object Detection

|  Detector  | mAP | Configs | Download | Note |
| :--------: |:---:|:-------:|:--------:|:----:|
| Deformable DETR | 17.1 | deformable_detr_r50_1x_star | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/deformable_detr_r50_1x_star.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/deformable_detr_r50_1x_star-fe862bb3.pth?download=true) |
| ARS-DETR | 28.1 | dn_arw_arm_arcsl_rdetr_r50_1x_star | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/dn_arw_arm_arcsl_rdetr_r50_1x_star.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/dn_arw_arm_arcsl_rdetr_r50_1x_star-cbb34897.pth?download=true) |
| RetinaNet | 21.8 | rotated_retinanet_hbb_r50_fpn_1x_star_oc | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_retinanet_hbb_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_retinanet_hbb_r50_fpn_1x_star_oc-3ec35d77.pth?download=true) |
| ATSS | 20.4 | rotated_atss_hbb_r50_fpn_1x_star_oc | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_atss_hbb_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_atss_hbb_r50_fpn_1x_star_oc-f65f07c2.pth?download=true) | 
|  KLD  |  25.0  | rotated_retinanet_hbb_kld_r50_fpn_1x_star_oc  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_retinanet_hbb_kld_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_retinanet_hbb_kld_r50_fpn_1x_star_oc-343a0b83.pth?download=true) |
|  GWD  |  25.3  | rotated_retinanet_hbb_gwd_r50_fpn_1x_star_oc  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_star_oc-566d2398.pth?download=true) |
| KFIoU |  25.5  | rotated_retinanet_hbb_kfiou_r50_fpn_1x_star_oc  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_star_oc-198081a6.pth?download=true) |
| DCFL | 29.0 | dcfl_r50_fpn_1x_star_le135 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/dcfl_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/dcfl_r50_fpn_1x_star_le135-a5945790.pth?download=true) |
| R<sup>3</sup>Det | 23.7 | r3det_r50_fpn_1x_star_oc | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/r3det_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/r3det_r50_fpn_1x_star_oc-c8c4a5e5.pth?download=true) |
| S2A-Net | 27.3 | s2anet_r50_fpn_1x_star_le135 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/s2anet_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/s2anet_r50_fpn_1x_star_le135-42887a81.pth?download=true) |
| FCOS  |  28.1  | [rotated-fcos-le90_r50_fpn_1x_star](./../../../configs/rotated_fcos/rotated-fcos-le90_r50_fpn_1x_star.py)  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_fcos_r50_fpn_1x_star_le90.log) \| [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/STAR/FCOS/rotated_fcos_r50_fpn_1x_star_le90-a579fbf7.pth) | 
| CSL | 27.4 | rotated_fcos_csl_gaussian_r50_fpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_fcos_csl_gaussian_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_fcos_csl_gaussian_r50_fpn_1x_star_le90-6ab9a42a.pth?download=true) | 
| PSC | 30.5 | rotated_fcos_psc_r50_fpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_fcos_psc_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_fcos_psc_r50_fpn_1x_star_le90-7acce1be.pth?download=true) |
| H2RBox-v2 | 27.3 | h2rbox_v2p_r50_fpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/h2rbox_v2p_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/h2rbox_v2p_r50_fpn_1x_star_le90-25409050.pth?download=true) |
| RepPoints  | 19.7 | rotated_reppoints_r50_fpn_1x_star_oc | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_reppoints_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_reppoints_r50_fpn_1x_star_oc-7a6c59b9.pth?download=true) |
| CFA | 25.1 | cfa_r50_fpn_1x_star_le135 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/cfa_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/cfa_r50_fpn_1x_star_le135-287f6b84.pth?download=true) |
| Oriented RepPoints  |  27.0  | oriented_reppoints_r50_fpn_1x_star_le135  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/oriented_reppoints_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/oriented_reppoints_r50_fpn_1x_star_le135-06389ea6.pth?download=true) | |
| G-Rep | 26.9 | g_reppoints_r50_fpn_1x_star_le135 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/g_reppoints_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/g_reppoints_r50_fpn_1x_star_le135-ec243141.pth?download=true) |
| SASM  |  28.2  | sasm_reppoints_r50_fpn_1x_star_oc  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/sasm_reppoints_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/sasm_reppoints_r50_fpn_1x_star_oc-4f1ca558.pth?download=true) | [p_bs=2](https://github.com/yangxue0827/STAR-MMRotate/blob/05c0064cbcd5c44437321b50e1d2d4ee9b4445db/mmrotate/models/detectors/single_stage_crop.py#L310) |
| Faster RCNN | 32.6 | [rotated-faster-rcnn-le90_r50_fpn_1x_star](./../../../configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_star.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_faster_rcnn_r50_fpn_1x_star_le90.log) \| [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/STAR/Faster_RCNN/rotated_faster_rcnn_r50_fpn_1x_star_le90-9a832bc2.pth) |
| Gliding Vertex | 30.7 | gliding_vertex_r50_fpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/gliding_vertex_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/gliding_vertex_r50_fpn_1x_star_le90-5c0bc879.pth?download=true) |
| Oriented RCNN | 33.2 | [oriented-rcnn-le90_r50_fpn_1x_star](./../../../configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_star.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/oriented_rcnn_r50_fpn_1x_star_le90.log) \| [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/STAR/Oriented_RCNN/oriented_rcnn_r50_fpn_1x_star_le90-0b66f6a4.pth) |
| RoI Transformer | 35.7 | roi_trans_r50_fpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/roi_trans_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/roi_trans_r50_fpn_1x_star_le90-e42f64d6.pth?download=true) |
| LSKNet-T | 34.7 | lsk_t_fpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/lsk_t_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/lsk_t_fpn_1x_star_le90-19635614.pth?download=true) |
| LSKNet-S | 37.8 | lsk_s_fpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/lsk_s_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/lsk_s_fpn_1x_star_le90-b77cdbc2.pth?download=true) |
| PKINet-S | 32.8 | pkinet_s_fpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/pkinet_s_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/pkinet_s_fpn_1x_star_le90-e1459201.pth?download=true) |
| ReDet | 39.1 | redet_re50_refpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/redet_re50_refpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/redet_re50_refpn_1x_star_le90-d163f450.pth?download=true) | [ReResNet50](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/re_resnet50_c8_batch256-25b16846.pth?download=true) |
| Oriented RCNN | 40.7 | oriented_rcnn_swin-l_fpn_1x_star_le90 | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/oriented_rcnn_swin-l_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/oriented_rcnn_swin-l_fpn_1x_star_le90-fe6f9e2d.pth?download=true) | [Swin-L](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth?download=true) |


### Train:

single node, for example:
```
python tools/train.py configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_star.py
```

multiple node, for example:
```
bash tools/dist_train.sh configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_star.py 2
```

### Test:

single node, for example:
```
python tools/test.py configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_star.py checkpoint_path
```

multiple node, for example:
```
bash tools/dist_test.sh configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_star.py checkpoint_path 2
```

1) test online

set `test_evaluator` in [`configs/_base_/datasets/star.py`](./../../../configs/_base_/datasets/star.py)
```
test_evaluator = dict(
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/star/Task1')
```

upload `Task1.zip` to STAR evaluation web [Codebench](https://www.codabench.org/competitions/3475/?secret_key=f1aff9f3-696d-4e10-9a63-f113153ed686)

2) test local

The official authors have released the gt for the test set, allowing for local evaluation.

set `test_evaluator` in [`configs/_base_/datasets/star.py`](./../../../configs/_base_/datasets/star.py)
```
test_evaluator = val_evaluator
```

```bibtex
@article{li2025star,
  title={STAR: A first-ever dataset and a large-scale benchmark for scene graph generation in large-size satellite imagery},
  author={Li, Yansheng and Wang, Linlin and Wang, Tingzhu and Yang, Xue and Luo, Junwei and Wang, Qi and Deng, Youming and Wang, Wenbin and Sun, Xian and Li, Haifeng and others},
  journal={IEEE Trans. Pattern Anal. Mach. Intell},
  volume={47},
  number={3},
  pages={1832--1849},
  year={2025}
}
```