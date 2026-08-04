# Wholly-WOOD: Wholly Leveraging Diversified-quality Labels for Weakly-supervised Oriented Object Detection

> [Wholly-WOOD: Wholly Leveraging Diversified-quality Labels for Weakly-supervised Oriented Object Detection](https://arxiv.org/abs/2502.09471)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/VisionXLab/point2rbox-v2/blob/main/resources/whollywood.png" width="800"/>
</div>

Accurately estimating the orientation of visual objects with compact rotated bounding boxes (RBoxes) has become a prominent demand, which challenges existing object detection paradigms that only use horizontal bounding boxes (HBoxes). To equip the detectors with orientation awareness, supervised regression/classification modules have been introduced at the high cost of rotation annotation. Meanwhile, some existing datasets with oriented objects are already annotated with horizontal boxes or even single points. It becomes attractive yet remains open for effectively utilizing weaker single point and horizontal annotations to train an oriented object detector (OOD). We develop Wholly-WOOD, a weakly-supervised OOD framework, capable of wholly leveraging various labeling forms (Points, HBoxes, RBoxes, and their combination) in a unified fashion. By only using HBox for training, our Wholly-WOOD achieves performance very close to that of the RBox-trained counterpart on remote sensing and other areas, significantly reducing the tedious efforts on labor-intensive annotation for oriented objects.

## Basic patterns

Extract [basic_patterns.zip](https://github.com/open-mmlab/mmrotate/files/13816461/basic_patterns.zip) to data folder. The path can also be modified in config files.

## Results and models

### End-to-end training

wait a minute  ...

## Citation

```
@ARTICLE{10891210,
  author={Yu, Yi and Yang, Xue and Li, Yansheng and Han, Zhenjun and Da, Feipeng and Yan, Junchi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Wholly-WOOD: Wholly Leveraging Diversified-Quality Labels for Weakly-Supervised Oriented Object Detection}, 
  year={2025},
  volume={47},
  number={6},
  pages={4438-4454},
  keywords={Annotations;Detectors;Training;Accuracy;Remote sensing;Object detection;Manuals;Visualization;Electronic mail;Labeling;Oriented object detection;weakly-supervised learning;computer vision},
  doi={10.1109/TPAMI.2025.3542542}}

```
