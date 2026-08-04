# Preparing DIOR Dataset

<!-- [DATASET] -->

```bibtex
@article{LI2020296,
    title = {Object detection in optical remote sensing images: A survey and a new benchmark},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {159},
    pages = {296-307},
    year = {2020},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2019.11.023},
    url = {https://www.sciencedirect.com/science/article/pii/S0924271619302825},
    author = {Ke Li and Gang Wan and Gong Cheng and Liqiu Meng and Junwei Han}
```

```bibtex
@ARTICLE{9795321,
  author={Cheng, Gong and Wang, Jiabao and Li, Ke and Xie, Xingxing and Lang, Chunbo and Yao, Yanqing and Han, Junwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Anchor-Free Oriented Proposal Generator for Object Detection}, 
  year={2022},
  volume={60},
  number={},
  pages={1-11},
  keywords={Proposals;Detectors;Object detection;Generators;Feature extraction;Remote sensing;Shape;Anchor-free oriented proposal generator (AOPG);oriented object detection;oriented proposal generation},
  doi={10.1109/TGRS.2022.3183022}}

```

## Download DIOR dataset

The DIOR dataset can be downloaded from [here](https://gcheng-nwpu.github.io/#Datasets) or [modelscope(魔塔)](https://www.modelscope.cn/datasets/wokaikaixinxin/DIOR/).

The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── DIOR
│   │   ├── JPEGImages-trainval
│   │   ├── JPEGImages-test
│   │   ├── Annotations
│   │   │   ├─ Oriented Bounding Boxes
│   │   │   ├─ Horizontal Bounding Boxes
│   │   ├── ImageSets
│   │   │   ├─ Main
│   │   │   │  ├─ train.txt
│   │   │   │  ├─ val.txt
│   │   │   │  ├─ test.txt
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/dior.py` to `data/dior/`.
