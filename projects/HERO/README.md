# HERO-Det: Hilbert Curve-Encoded Rotation-Equivariant Oriented Object Detector

[Official github](https://github.com/Qian-CV/HERO-Det)

[AAAI 2026 Link](https://ojs.aaai.org/index.php/AAAI/article/view/37753)

[PDF Link](https://ojs.aaai.org/index.php/AAAI/article/view/37753/41715)

🤷‍♂️🤷‍♀️[Qi Ming*](https://github.com/ming71), [Liuqian Wang*](https://github.com/Qian-CV), Juan Fang†, Xudong Zhao†, Yucheng Xu, Ziyi Teng, Yue Zhou, Xiaoxi Hu, Xiaohan Zhang, Yufei Guo

![](https://github.com/Qian-CV/HERO-Det/blob/main/doc/fig1.png)

## Introduction 
This repository is the official implementation of “Hilbert Curve-Encoded Rotation-Equivariant Oriented Object Detector with Locality-Preserving Spatial Mapping”, built upon the MMRotate/MMDetection ecosystem.

## What's New 
- Add HERO codebase and config for DOTA multi-scale training :alarm_clock: **2025-11-11**
- Provide Hilbert-based sequence transforms and cyclic-shift fusion module :alarm_clock: **2025-11-11**

## Results and Models

| Model |  mAP  | Angle | lr schd | Batch Size | AUG |                          Configs                          |          Download          |
| :---: |:-----:| :---: | :-----: |:----------:| :---: |:--------------------------------------------------------:|:--------------------------:|
| HERO <br>(ResNet-50, FPN) | -     | le90 | 3x |     4   | none | [HERO-3x](./configs/hero_le90_r50_fpn_3x_dota.py) | come soon |
| HERO <br>(ResNet-50, FPN) | 79.56 | le90 | 3x |     4   | MS   | [HERO-3x](./configs/hero_le90_r50_fpn_3x_dota-ms.py) | come soon |

Notes:
- We will release checkpoints soon. You can place your files in `tools/model_weight/` and update the links above.

## Overall Architecture
1) FPN features are flattened to 1D sequences via Hilbert mapping (H×W → L)  
2) Optional cross-scale fusion using cross-attention over the sequences  
3) Hilbert-Conv1D processes sequences before RPN classification/regression  
4) RoI extractor outputs rotated (8×8) and horizontal (hor_size×hor_size) features  
5) RoI head: rotated features for classification; horizontal features for regression; optional ORN enhancement

## Code Structure
- RPN head (Hilbert sequence modeling)
  - `projects/HERO/hero/hilbert_rpn_head.py`
- RoI extraction (RRoI + horizontal RoI)
  - `projects/HERO/hero/rotate_single_level_Hroi_and_Rroi_extractor.py`
- RoI head (classification + regression)
  - `projects/HERO/hero/hilbert_convfc_rbbox_head.py`
- HERO utilities
  - `projects/HERO/hero/HPFormer.py` — Hilbert/Row-Major/Snake/Morton/Peano transforms (flatten/unflatten)
  - `projects/HERO/hero/hilbert_cross_attention.py` — cross-scale attention for sequences
  - `projects/HERO/hero/hilbert_cyclic_shift.py`, `projects/HERO/hero/cyclic_shift_direct.py` — rotation simulation
  - `projects/HERO/hero/cyclic_shift_methods.py` — rotation-aware fusion modules
  - `projects/HERO/hero/ResidualORN.py` — residual ORN module

## Installation
if you donnot install `hilbertcurve`, please install it first.
```bash
pip install hilbertcurve -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Data Preparation
Follow MMRotate’s DOTA preparation. See `tools/data/dota/` for scripts and instructions.

## Training
```bash
# single-GPU
python tools/train.py projects/HERO/configs/hero_le90_r50_fpn_3x_dota.py

# multi-GPU
 bash tools/dist_train.sh projects/HERO/configs/hero_le90_r50_fpn_3x_dota.py 2
 ```

## Inference
```bash
python tools/test.py projects/HERO/configs/hero_le90_r50_fpn_3x_dota.py path/your_checkpoint.pth
```

## Citation
If you find this work useful, please cite: TBD

## Acknowledgements
[Official github](https://github.com/Qian-CV/HERO-Det)
