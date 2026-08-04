# ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior based Orientation Prediction for Detecting Aerial Image Objects (CVPR 2025)

> CVPR 2025 [Link](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_ABBSPO_Adaptive_Bounding_Box_Scaling_and_Symmetric_Prior_based_Orientation_CVPR_2025_paper.html)

> Official Code [Link](https://github.com/KAIST-VICLab/ABBSPO)

> Project Page [Link](https://kaist-viclab.github.io/ABBSPO_site/)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://kaist-viclab.github.io/ABBSPO_site/image/overall%20pipeline.png"/>
</div>

Weakly supervised Oriented Object Detection (WS-OOD) has gained attention as a cost-effective alternative to fully supervised methods, providing efficiency and high accuracy. Among weakly supervised approaches, horizontal bounding box (HBox) supervised OOD stands out for its ability to directly leverage existing HBox annotations while achieving the highest accuracy under weak supervision settings. This paper introduces adaptive bounding box scaling and symmetry-prior-based orientation prediction, called ABBSPO that is a framework for WS-OOD. Our ABBSPO addresses the limitations of previous HBox-supervised OOD methods, which compare ground truth (GT) HBoxes directly with predicted RBoxes' minimum circumscribed rectangles, often leading to inaccuracies. To overcome this, we propose: (i) Adaptive Bounding Box Scaling (ABBS) that appropriately scales the GT HBoxes to optimize for the size of each predicted RBox, ensuring more accurate prediction for RBoxes' scales; and (ii) a Symmetric Prior Angle (SPA) loss that uses the inherent symmetry of aerial objects for self-supervised learning, addressing the issue in previous methods where learning fails if they consistently make incorrect predictions for all three augmented views (original, rotated, and flipped). Extensive experimental results demonstrate that our ABBSPO achieves state-of-the-art results, outperforming existing methods.

## Results and models

**Note**:
The DOTA-v1.0 checkpoint is trained **using only the training split** (without validation data).

| Dataset | Model | Training Log | Config |
|--------|-------|--------------|------|
| DIOR | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ABBSPO/abbspo_dior_epoch12.pth) | [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ABBSPO/abbspo_dior_epoch12.log) | [config](./configs/abbspo-le90_r50_fpn-1x_dior.py) |
| DOTA-v1.0 | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ABBSPO/abbspo_dota_epoch12.pth) | [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ABBSPO/abbspo_dota_epoch12.log) | [config](./configs/abbspo-le90_r50_fpn-1x_dota.py) |
| SIMD | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ABBSPO/abbspo_simd_epoch12.pth) | [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ABBSPO/abbspo_simd_epoch12.log) | [config] |




## Citation

```
@InProceedings{Lee_2025_CVPR,
    author    = {Lee, Woojin and Chang, Hyugjae and Moon, Jaeho and Lee, Jaehyup and Kim, Munchurl},
    title     = {ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior based Orientation Prediction for Detecting Aerial Image Objects},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {8848-8858}
}
```
