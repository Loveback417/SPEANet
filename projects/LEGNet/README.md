# LEGNet: Lightweight Edge-Gaussian Driven Network for Low-Quality Remote Sensing Image Object Detection

> [LEGNet: Lightweight Edge-Gaussian Driven Network for Low-Quality Remote Sensing Image Object Detection](https://arxiv.org/abs/2503.14012)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/lwCVer/LEGNet/blob/main/docs/legnet.png" width="800"/>
</div>

Remote sensing object detection (RSOD) often suffers from degradations such as low spatial resolution, sensor noise, motion blur, and adverse illumination. These factors diminish feature distinctiveness, leading to ambiguous object representations and inadequate foreground-background separation. Existing RSOD methods exhibit limitations in robust detection of low-quality objects. To address these pressing challenges, we introduce LEGNet, a lightweight backbone network featuring a novel Edge-Gaussian Aggregation (EGA) module specifically engineered to enhance feature representation derived from low-quality remote sensing images. EGA module integrates: (a) orientation-aware Scharr filters to sharpen crucial edge details often lost in low-contrast or blurred objects, and (b) Gaussian-prior-based feature refinement to suppress noise and regularize ambiguous feature responses, enhancing foreground saliency under challenging conditions. EGA module alleviates prevalent problems in reduced contrast, structural discontinuities, and ambiguous feature responses prevalent in degraded images, effectively improving model robustness while maintaining computational efficiency. Comprehensive evaluations across five benchmarks (DOTA-v1.0, v1.5, DIOR-R, FAIR1M-v1.0, and VisDrone2019) demonstrate that LEGNet achieves state-of-the-art performance, particularly in detecting low-quality objects.

## Results and models

DOTA1.0

|         Backbone         |  AP50  | Angle | lr schd | Mem (GB) | fps | Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LEGNet-Tiny <br> (1024,1024,200) | 79.37 | le90  |   3x    |  8.5G~   |      -      |  single scale rr | 8=4gpu*<br>2img/gpu | [orcnn_legnet_tiny_dota10<br>_test_ss_e36.py](./configs/orcnn_legnet_tiny_dota10_test_ss_e36.py) | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/LEGNet/lwegnet_tiny_orcnn_dota19_ss.pth) |
| LEGNet-Small <br> (1024,1024,200) | 80.03 | le90  |   3x   |   8.5G~   |      -      |  single scale rr |  8=4gpu*<br>2img/gpu  |             [orcnn_legnet_small_dota10<br>_test_ss_e36.py](./configs/orcnn_legnet_small_dota10_test_ss_e36.py)              |   [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/LEGNet/lwegnet_small_orcnn_dota10_ss.pth)     |


NOTE: The results and checkpoints come from [official github](https://github.com/lwCVer/LEGNet).

Train

Single-node single-GPU

```shell
python tools/train.py projects/LEGNet/configs/orcnn_legnet_tiny_dota10_test_ss_e36.py
```

Single-node multi-GPU, for example 2 gpus:

```shell 
bash tools/dist_train.sh projects/LEGNet/configs/orcnn_legnet_tiny_dota10_test_ss_e36.py 2
```

Test

Single-node single-GPU

```shell
python tools/test.py projects/LEGNet/configs/orcnn_legnet_tiny_dota10_test_ss_e36.py your_checkpoint_path
```


Single-node multi-GPU, for example 2 gpus:

```shell
python tools/dist_test.sh projects/LEGNet/configs/orcnn_legnet_tiny_dota10_test_ss_e36.py your_checkpoint_path 2
```

DOTA1.5

|         Backbone         |  AP50  | Angle | lr schd | Mem (GB) | fps | Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LEGNet-Small <br> (1024,1024,200) | 72.89 | le90  |   3x    |   8.5G~   |     -      |  single scale  |   8=4gpu*<br>2img/gpu   |                 [orcnn_legnet_small_dota15<br>_test_ss_e36.py](./configs/orcnn_legnet_small_dota15_test_ss_e36.py)                  |                   [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/LEGNet/lwegnet_small_orcnn_dota15_ss.pth) |  

NOTE: The results and checkpoints come from [official github](https://github.com/lwCVer/LEGNet).



## Citation

```
@article{lu2025legnet,
  title={LEGNet: Lightweight Edge-Gaussian Driven Network for Low-Quality Remote Sensing Image Object Detection},
  author={Lu, Wei and Chen, Si-Bao and Li, Hui-Dong and Shu, Qing-Ling and Ding, Chris HQ and Tang, Jin and Luo, Bin},
  journal={arXiv preprint arXiv:2503.14012},
  year={2025}
}
```
