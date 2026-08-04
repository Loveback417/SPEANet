# Fourier Angle Alignment for Oriented Object Detection in Remote Sensing (CVPR 2026)

[Arxiv](https://arxiv.org/abs/2602.23790)

[Official repo](https://github.com/gcy0423/Fourier-Angle-Alignment)

## 📰 Abstract
In remote sensing rotated object detection, mainstream methods suffer from two bottlenecks, directional incoherence at detector neck and task conflict at detecting head. Ulitising fourier rotation equivariance, we introduce **Fourier Angle Alignment**, which analyses angle information through frequency spectrum and aligns the main direction to a certain orientation. Then we propose two plug and play modules : **FAAFusion** and **FAA Head**. FAAFusion works at the detector neck, aligning the main direction of higher-level features to the lower-level features and then fusing them. FAA Head serves as a new detection head, which pre-aligns RoI features to a canonical angle and adds them to the original features before classification and regression. Experiments on DOTA-v1.0, DOTA-v1.5 and HRSC2016 show that our method can greatly improve previous work. Particularly, our method achieves new state-of-the-art results of 78.72% mAP on DOTA-v1.0 and 72.28% mAP on DOTA-v1.5 datasets with single scale training and testing, validating the efficacy of our approach in remote sensing object detection.

<div align="center">
  <img src="https://github.com/gcy0423/Fourier-Angle-Alignment/blob/main/Method-FAA.png"  width="800"/>
</div>


## 🚀 Results and Configs


### DOTA-v1.0
| Model |      mAP      | Angle | lr schd | Config |
| :--- |:-------------:| :---: | :---: | :--- |
| LSKNet-S + FAA | 78.49 (+1.00) | le90 | 1x | [config](./configs/lsk_s_fpn_1x_dota_le90_faa.py) |


```
python tools/train.py projects/FAA/configs/lsk_s_fpn_1x_dota_le90_faa.py
```

### FAA Head Only (DOTA-v1.0)


| Model |      mAP      | Angle | lr schd | Config | Download |
| :--- |:-------------:| :---: | :---: | :--- | :---: |
| LSKNet-S + FAA Head | 78.27 (+0.78) | le90 | 1x | [config](./configs/lsk_s_fpn_1x_dota_le90_faahead.py) | [baidu cloud](https://pan.baidu.com/s/1VuMQcn33I8SMY9oMDfZ4hg?pwd=vmk3) \| [modelscope](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/FAA/lsk_faahead_epoch_12.pth) |

NOTE: We do not retrain the model from scratch. The checkpoint is from official repo.

Train and Test commands are as follows:

```
python tools/train.py projects/FAA/configs/lsk_s_fpn_1x_dota_le90_faahead.py
python tools/test.py  projects/FAA/configs/lsk_s_fpn_1x_dota_le90_faahead.py your_path/lsk_faahead_epoch_12.pth
```


## 🧩 Plug and Play

Our proposed modules are designed to be lightweight and highly extensible. You can easily integrate them into your own custom detectors by simply copying the corresponding files.

### 1. FAAFusion
**File:** [`projects/FAA/faa/faafusion.py`](./faa/faafusion.py)

`FAAFusion` serves as a plug-and-play module for the feature pyramid network. It dynamically aligns the orientation of the high-level feature map to match the low-level feature map before fusion.

```python
from faafusion import FAAFusion

# Initialize the module
fusion_module = FAAFusion(m=7, c_mid=16)

# Inputs:
# x_high: Tensor of shape [B, C, H_h, W_h] (High-level feature)
# x_low:  Tensor of shape [B, C, H_l, W_l] (Low-level feature)

# Output:
# fused:  Tensor of shape [B, C, H_l, W_l] (Aligned and fused feature)
fused_feature = fusion_module(x_high, x_low)
```
>**Note**: We also provide FAAFusionFPN in the same file, which is a ready-to-use FPN variant integrating FAAFusion.

### 2. FAA Head
**File:** [`projects/FAA/faa/faa_head.py`](./faa/faa_head.py)

`FAAHead` can directly replace standard RoI heads. It explicitly aligns the 7x7 RoI features to a canonical direction using frequency spectrum analysis, effectively alleviating the task conflict between classification and regression.

```python
from faahead import FAAHead

# Initialize the head (arguments are inherited from RotatedShared2FCBBoxHead)
faa_head = FAAHead(
    num_classes=15, 
    in_channels=256, 
    fc_out_channels=1024, 
    roi_feat_area=7*7
)

# Input:
# x: RoI features of shape [N, C, 7, 7] (where C is typically 256)

# Outputs:
# cls_score: Classification logits
# bbox_pred: Bounding box regression offsets
cls_score, bbox_pred = faa_head(x)
```


## 📖 Citation

```
@InProceedings{gcy2026faa,
  title={Fourier Angle Alignment for Oriented Object Detection in Remote Sensing},
  author={Gu, Changyu and Chen, Linwei and Gu, Lin and Fu, Ying},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
