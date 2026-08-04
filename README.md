This is the official Pytorch/Pytorch implementation of the papers: <br>
# SPEANet: Structural Prior Enhanced Attention Network for Parameter-Efficient Remote Sensing Object Detection

> Wei Lu, Junjie Li, Feifei Sang, Si-Bao Chen* <br>
> Submitted to AAAI 2027 <br>
> [[*paper*](https://openreview.net/attachment?id=r9bTsBGpHw&name=pdf)] <br>

<!-- [ALGORITHM] -->

## Abstract

Remote sensing object detection (RSOD) requires parameter efficient backbones that preserve weak geometric cues under scale variation, blur, shadows, and cluttered backgrounds.
Fixed image operators, including LoG, Sobel, Laplacian,
and Gaussian filters, provide complementary contour and
frequency responses without introducing learnable parameters. However, their direct integration into hierarchical feature learning is challenging because operator responses are
content-agnostic, sensitive to background textures, and not
equally suitable for all feature stages. To address this, we
propose the Structural Prior Enhanced Attention Network
(SPEANet), a parameter-efficient RSOD backbone. Instead
of relying purely on learned convolutions, SPEANet explicitly integrates fixed structural operators into hierarchical feature learning through stage-specific modeling and context conditioned gating. Experiments on five benchmarks demonstrate a favorable accuracy-parameter trade-off. With Oriented
R-CNN, SPEANet achieves 78.55% mAP on DOTA-v1.0,
72.24% on DOTA-v1.5, and 67.30% on DIOR-R using 23.0M
total parameters, of which 5.97M belong to the backbone.

## Pretrained Weights of Backbone
Imagenet 300-epoch pre-trained SPEANet backbone: [Download](https://github.com/Loveback417/SPEANet/releases/download/weight/speanet_s_img_meet_179_71_00_clean.pth)

## Results

DOTA-v1.0

|           Backbone           |  Detector  |  Parameter  | mAP  | Angle | lr schd |      Aug      |                    Batch Size                    | Configs                                                                               |                    Weights                    |
|:----------------------------:|:-----:|:-----:|:-----:|:-----:| :-----: |:-------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------|:-------------:|
| SPEANet <br> (1024,1024,200) | 	Oriented R-CNN | 23.0M | 78.55 | le90  |   3x    | single scale  | 8<br/>(2&nbsp;gpus&nbsp;*&nbsp;4&nbsp;imgs/gpu)  | [orcnn_speanet_dota10_ss_le90_e36.py](./configs/orcnn_speanet_dota10_ss_le90_e36.py)  | [model](https://github.com/Loveback417/SPEANet/releases/download/weight/orcnn_speanet_dota10_ss_le90.pth)|
| SPEANet <br> (1024,1024,200) |S2ANet | 14.0M | 78.16 | le135 |   3x    | single scale  | 16<br/>(2&nbsp;gpus&nbsp;*&nbsp;8&nbsp;imgs/gpu) | [s2anet_speanet_dota10_ss_le135_e36.py](./configs/s2anet_speanet_dota10_ss_le135_e36.py)  | [model](https://github.com/Loveback417/SPEANet/releases/download/weight/s2anet_speanet_dota10_ss_le135.pth)|

DOTA-v1.5

|           Backbone           | Detector  |  Parameter  |  mAP  | Angle | lr schd |     Aug      |      Batch Size       | Configs                                                                              |  Weight                    |
|:----------------------------:|:-----:| :---: |:-----:| :---: | :-----: |:------------:|:---------------------:|:-------------------------------------------------------------------------------------|:---: |
| SPEANet <br> (1024,1024,200) | Oriented R-CNN | 23.0M | 72.24 | le90  |   3x    | single scale | 8<br/>(2&nbsp;gpus&nbsp;*&nbsp;4&nbsp;imgs/gpu) | [orcnn_speanet_dota15_ss_le90_e36.py](./configs/orcnn_speanet_dota15_ss_le90_e36.py) | [model](https://github.com/Loveback417/SPEANet/releases/download/weight/orcnn_speanet_dota15_ss_le90.pth) | 

DIOR-R

|        Backbone         | Detector  |  Parameter  | AP50  | Angle | lr schd |     Aug      |                    Batch Size                    |                                             Configs                                              |  Weight                    |
|:-----------------------:|:-----:| :---: |:-----:| :---: | :-----: |:------------:|:------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:---: |
| SPEANet <br> (800, 800) | Oriented R-CNN | 23.0M | 67.30 | le90  |   3x    | single scale | 16<br/>(2&nbsp;gpus&nbsp;*&nbsp;8&nbsp;imgs/gpu) |       [orcnn_speanet_dior_r_ss_le90_e36.py](./configs/orcnn_speanet_dior_r_ss_le90_e36.py)   | [model](https://github.com/Loveback417/SPEANet/releases/download/weight/orcnn_speanet_diorr_ss_le90.pth) | 


Train

Single-node single-GPU

```shell
python tools/train.py projects/SPEANet/configs/orcnn_speanet_dota10_ss_le90_e36.py
```

Single-node multi-GPU, for example 2 gpus:

```shell 
bash tools/dist_train.sh projects/SPEANet/configs/orcnn_speanet_dota10_ss_le90_e36.py 2
```

Test

Single-node single-GPU

```shell
python tools/test.py projects/SPEANet/configs/orcnn_speanet_dota10_ss_le90_e36.py your_checkpoint_path
```


Single-node multi-GPU, for example 2 gpus:

```shell
bash tools/dist_test.sh projects/SPEANet/configs/orcnn_speanet_dota10_ss_le90_e36.py your_checkpoint_path 2
```

## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n SPEANet python=3.10 -y
conda activate SPEANet
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130

# Due to compatibility issues between different versions, MMCV must be downloaded and built from source locally.
python -m pip install "setuptools<81" wheel ninja psutil packaging pytest-runner
python -m pip install "mmengine>=0.11.0rc0,<1.0.0"
#You may choose a different directory to download the code.
cd ../
git clone -b v2.1.0 https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install --no-build-isolation -v . 2>&1 | tee mmcv_build.log

cd ../SPEANet-main
pip install -r requirements.txt
pip install -v -e .
```


## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models), [ai4rs](https://github.com/wokaikaixinxin/ai4rs), and [mmrotate](https://github.com/open-mmlab/mmrotate) repositories.
MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

If you have any questions about this work, you can contact me. 

Email: 1740547476@qq.com or e125221143@stu.ahu.edu.cn.

Your star is the power that keeps us updating github.


## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. 
Any commercial use should get formal permission first.
