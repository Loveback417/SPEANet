# LWGANet: A Lightweight Group Attention Backbone for Remote Sensing Visual Tasks

> [LWGANet: A Lightweight Group Attention Backbone for Remote Sensing Visual Tasks](https://arxiv.org/abs/2501.10040)

> Official Code [Link](https://github.com/lwCVer/LWGANet)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/lwCVer/LWGANet/raw/main/figures/LWGANet.png"/>
</div>

Remote sensing (RS) visual tasks have gained significant academic and practical importance. However, they encounter numerous challenges that hinder effective feature extraction, including the detection and recognition of multiple objects exhibiting substantial variations in scale within a single image. While prior dual-branch or multi-branch architectural strategies have been effective in managing these object variances, they have concurrently resulted in considerable increases in computational demands and parameter counts. Consequently, these architectures are rendered less viable for deployment on resource-constrained devices. Contemporary lightweight backbone networks, designed primarily for natural images, frequently encounter difficulties in effectively extracting features from multiscale objects, which compromises their efficacy in RS visual tasks. This article introduces LWGANet, a specialized lightweight backbone network tailored for RS visual tasks, incorporating a novel lightweight group attention (LWGA) module designed to address these specific challenges. The LWGA module, tailored for RS imagery, adeptly harnesses redundant features to extract a wide range of spatial information, from local to global scales, without introducing additional complexity or computational overhead. This facilitates precise feature extraction across multiple scales within an efficient framework. LWGANet was rigorously
evaluated across twelve datasets, which span four crucial RS visual tasks: scene classification, oriented object detection, semantic segmentation, and change detection. The results confirm LWGANet’s widespread applicability and its ability to maintain an optimal balance between high performance and low complexity, achieving state-of-the-art results across diverse datasets. LWGANet emerged as a novel solution for resource-limited scenarios requiring robust RS image processing capabilities

## Results and models

Imagenet 300-epoch pre-trained LWGANet-L0 backbone: [Download](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/LWGANet/lwganet_l0_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L1 backbone: [Download](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/LWGANet/lwganet_l1_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L2 backbone: [Download](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/LWGANet/lwganet_l2_e296.pth)

DOTA1.0

|         Backbone         |  AP50  | Angle | lr schd | Mem (GB) | fps | Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LWGANet_L2 <br> (1024,1024,200) | 79.02 | le90  |   30e    |  -   |      -      |  single scale rr  | 8=4gpu*<br>2img/gpu | [ORCNN_LWGANet_L2_fpn_<br>le90_dota10_ss_e30.py](./configs/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/LWGANet/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.pth) |



NOTE: The results and checkpoints come from [official github](https://github.com/lwCVer/LWGANet).

Train

Single-node single-GPU

```shell
python tools/train.py projects/LWGANet/configs/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.py
```

Single-node multi-GPU, for example 2 gpus:

```shell 
bash tools/dist_train.sh projects/LWGANet/configs/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.py 2
```

Test

Single-node single-GPU

```shell
python tools/test.py projects/LWGANet/configs/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.py your_checkpoint_path
```


Single-node multi-GPU, for example 2 gpus:

```shell
bash tools/dist_test.sh projects/LWGANet/configs/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.py your_checkpoint_path 2
```

DOTA1.5

|         Backbone         |  AP50  | Angle | lr schd | Mem (GB) | fps | Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LWGANet_L2 <br> (1024,1024,200) | 72.91 | le90  |   30e    |  -  |     -   |  single scale rr |   8=4gpu*<br>2img/gpu   |                 [ORCNN_LWGANet_L2_fpn_<br>le90_dota15_ss_e30.py](./configs/ORCNN_LWGANet_L2_fpn_le90_dota15_ss_e30.py)                  |                   [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/LWGANet/ORCNN_LWGANet_L2_fpn_le90_dota15_ss_e30.pth) |  

NOTE: The results and checkpoints come from [official github](https://github.com/lwCVer/LWGANet).



## Citation

```
@article{lu2025lwganet,
  title={Lwganet: A lightweight group attention backbone for remote sensing visual tasks},
  author={Lu, Wei and Chen, Si-Bao and Ding, Chris HQ and Tang, Jin and Luo, Bin},
  journal={arXiv preprint arXiv:2501.10040},
  year={2025}
}
```
