# Changer (TGRS 2023)

> TGRS Link [Changer: Feature Interaction is What You Need for Change Detection](https://ieeexplore.ieee.org/document/10129139)

> ArXiv Link [Changer: Feature Interaction is What You Need for Change Detection](https://arxiv.org/abs/2209.082906)

> Official Code [Link](https://github.com/likyoo/open-cd)

## Introduction

## Abstract
Change detection is an important tool for long-term earth observation missions. It takes bi-temporal images as input and predicts “where” the change has occurred. Different from other dense prediction tasks, a meaningful consideration for change detection is the interaction between bi-temporal features. With this motivation, in this paper we propose a novel general change detection architecture, MetaChanger, which includes a series of alternative interaction layers in the feature extractor. To verify the effectiveness of MetaChanger, we propose two derived models, ChangerAD and ChangerEx with simple interaction strategies: Aggregation-Distribution (AD) and “exchange”. AD is abstracted from some complex interaction methods, and “exchange” is a completely parameter&computation-free operation by exchanging bi-temporal features. In addition, for better alignment of bi-temporal features, we propose a Flow Dual-Alignment Fusion (FDAF) module which allows interactive alignment and feature fusion. Crucially, we observe Changer series models achieve competitive performance on different scale change detection datasets. Further, our proposed ChangerAD and ChangerEx could serve as a starting baseline for future MetaChanger design.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/44317497/192922229-9a9480c2-cb12-42e5-84e6-92ee1df1f775.png" width="90%"/>
</div>

```bibtex
@ARTICLE{10129139,
  author={Fang, Sheng and Li, Kaiyu and Li, Zhe},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Changer: Feature Interaction is What You Need for Change Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-11},
  keywords={Feature extraction;Task analysis;Transformers;Image segmentation;Decoding;Semantics;Indexes;Change detection;deep neural network;feature interaction;high-resolution remote sensing (RS) image},
  doi={10.1109/TGRS.2023.3277496}}
```

## Results and models

### S2Looking

|  Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            |                           download                           |
| :-------: | :------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ChangerEx |   r18    |  512x512  |  80000  |    -     |   75.04   | 59.35  |  66.28   | 49.57 | [config] | [model]  |
| ChangerEx |   s50    |  512x512  |  80000  |    -     |   74.63   | 61.08  |  67.18   | 50.58 | [config] | [model]  |
| ChangerEx |   s101   |  512x512  |  80000  |    -     |   74.40   | 61.95  |  67.61   | 51.07 | [config] | [model]  |
| ChangerEx |  MIT-B0  |  512x512  |  80000  |    -     |   73.01   | 62.04  |  67.08   | 50.47 | [config] | [model]  |



### LEVIR-CD

|  Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |                            config                            |                           download                           |
| :-------: | :------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ChangerEx |   r18    |  512x512  |  40000  |    -     |   92.86   | 90.78  |  91.81   | 84.86 | [config](./configs/changer_ex_r18_512x512_40k_levircd.py) | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/changer/changer_ex_r18_512x512_40k_levircd/ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth)  |
| ChangerEx |   s50    |  512x512  |  40000  |    -     |   93.47   | 90.95  |  92.19   | 85.51 | [config](./configs/changer_ex_s50_512x512_40k_levircd.py) | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/changer/changer_ex_s50_512x512_40k_levircd/ChangerEx_s50-512x512_40k_levircd_20220702-145628.pth)  |
| ChangerEx |   s101   |  512x512  |  40000  |    -     |   93.38   | 91.31  |  92.33   | 85.76 | [config](./configs/changer_ex_s101_512x512_40k_levircd.py) | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/changer/changer_ex_s101_512x512_40k_levircd/ChangerEx_s101-512x512_40k_levircd_20220710-082722.pth)  |
| ChangerEx |  MIT-B0  |  512x512  |  40000  |    -     |   93.61   | 90.56  |  92.06   | 85.29 | [config] | [model]  |


- All metrics are based on the category "change".
- All scores are computed on the test set.
- The performance of `MIT-B0` is unstable. `ChangerEx` with `MIT-B0` may fluctuate about 0.5 F1-Score.