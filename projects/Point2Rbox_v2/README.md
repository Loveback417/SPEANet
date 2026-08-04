# Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances

> [Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Point2RBox-v2_Rethinking_Point-supervised_Oriented_Object_Detection_with_Spatial_Layout_Among_CVPR_2025_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/VisionXLab/point2rbox-v2/raw/main/resources/point2rbox_v2.png" width="800"/>
</div>

With the rapidly increasing demand for oriented object detection (OOD), recent research involving weakly-supervised detectors for learning OOD from point annotations has gained great attention. In this paper, we rethink this challenging task setting with the layout among instances and present Point2RBox-v2. At the core are three principles: 1) Gaussian overlap loss. It learns an upper bound for each instance by treating objects as 2D Gaussian distributions and minimizing their overlap. 2) Voronoi watershed loss. It learns a lower bound for each instance through watershed on Voronoi tessellation. 3) Consistency loss. It learns the size/rotation variation between two output sets with respect to an input image and its augmented view. Supplemented by a few devised techniques, e.g. edge loss and copy-paste, the detector is further enhanced. To our best knowledge, Point2RBox-v2 is the first approach to explore the spatial layout among instances for learning point-supervised OOD. Our solution is elegant and lightweight, yet it is expected to give a competitive performance especially in densely packed scenes: 62.61%/86.15%/34.71% on DOTA/HRSC/FAIR1M.

## Results and models

### End-to-end training

**DIOR-R**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,800,200) | 18.53 | 34.31  |   17.30    |   le90   |      1x      |  -  | 2=1gpu*<br>2img/gpu      | [point2rbox_v2-1x-dior.py](./configs/point2rbox_v2-1x-dior.py) | [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2-1x-dior/epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2-1x-dior/20250715_090534.log) \| [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) |

Note: This is the unofficial checkpoint. The official code is [here](https://github.com/VisionXLab/point2rbox-v2). The end-to-end training results on DIOR-R is 34.70 AP50 from Table 2 in [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Point2RBox-v2_Rethinking_Point-supervised_Oriented_Object_Detection_with_Spatial_Layout_Among_CVPR_2025_paper.pdf). In our reimplementation, the end-to-end training performance reaches 34.31 AP50 on DIOR-R.

| class   | airplane | airport | baseballfield | basketballcourt | bridge  | 
|---------|----------|---------|---------------|-----------------|---------|
| ap      | 0.54956  | 0.09324 | 0.65348       | 0.78988         | 0.11580 |
| **class**   | **chimney**  | **expressway-service-area** | **expressway-toll-station** | **dam**     | **golffield** |
| ap          | 0.66228      | 0.06702                 | 0.33284                 | 0.04869     | 0.09787       |
| **class**   | **groundtrackfield** | **harbor** | **overpass** | **ship**    | **stadium** |
| ap          | 0.44686           | 0.01818    | 0.25220      | 0.59536     | 0.41879     |
| **class**   | **storagetank** | **tenniscourt** | **trainstation** | **vehicle** | **windmill** |
| ap          | 0.45945     | 0.80095     | 0.08553      | 0.22283     | 0.15133     |
| **mAP**     |         |         |         |         |             |
| **0.34311** |         |         |         |         |             |

**Train**

```
python tools/train.py projects/Point2Rbox_v2/configs/point2rbox_v2-1x-dior.py
```

**Test**
```
python tools/test.py projects/Point2Rbox_v2/configs/point2rbox_v2-1x-dior.py work_dirs/point2rbox_v2-1x-dior/epoch_12.pth
```


**DOTA-v1.0**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | lr | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :---: | :--------: | :---------------------------------------------: | :-------------------------------: |
| ResNet50 <br> (1024,1024,200) | 23.00 | 49.14  |  18.05  |   le90   |  1x  | -  | 5e-5 | 2=1gpu*<br>2img/gpu      | [point2rbox_v2<br>-1x-dota.py](./configs/point2rbox_v2-1x-dota.py) | [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2-1x-dota/epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2-1x-dota/20250717_091611/20250717_091611.log) \| <br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2-1x-dota/Task1.zip)|

Note: This is the **unofficial** checkpoint. The official code is [here](https://github.com/VisionXLab/point2rbox-v2).  
Note: The official result is **51.00 AP50** on DOTA-v1.0, but in this project the result is **49.14 AP50** on DOTA-v1.0.

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.4914394914250513  
ap of each class: plane:0.7895142420733647, baseball-diamond:0.5192427808914901, bridge:0.1258016724118419, ground-track-field:0.39544236467140365, small-vehicle:0.7195815397041687, large-vehicle:0.6251712927402223, ship:0.7485794995411434, tennis-court:0.8844509906311375, basketball-court:0.37082091881498935, storage-tank:0.7344498985015621, soccer-ball-field:0.15261130797479555, roundabout:0.3277056203538273, harbor:0.2876561426278339, swimming-pool:0.4914347280842572, helicopter:0.1991293723537319  
COCO style result:  
AP50: 0.4914394914250513  
AP75: 0.18050823121731294  
mAP: 0.2299087999300767  

**Train**
```
python tools/train.py projects/Point2Rbox_v2/configs/point2rbox_v2-1x-dota.py
```

**Test**
```
python tools/test.py projects/Point2Rbox_v2/configs/point2rbox_v2-1x-dota.py work_dirs/point2rbox_v2-1x-dota/epoch_12.pth
```


### Two-stage training

#### DOTA-v1.0

##### Step 1: Pseudo Generator

Use the above trained model (1st stage, train Point2RBox-v2) as the pseudo generator:

```
# this config file runs inference on trainval set
# DOTA-v1.0
python tools/test.py projects/Point2Rbox_v2/configs/point2rbox_v2-pseudo-generator-dota.py work_dirs/point2rbox_v2-1x-dota/epoch_12.pth
```

Now the pseudo labels for trainval set have been saved at `data/split_ss_dota/point2rbox_v2_pseudo_labels.bbox.json`. The **unofficial** point2rbox_v2 dota1.0 pseudo labels are available for [download here](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2_dotav1.0_rotated-fcos-1x-dota-using-pseudo/point2rbox_v2_pseudo_labels.bbox.json)

##### Step 2: Train FCOS using pseudo labels

The pseudo labels for trainval set have been saved at `data/split_ss_dota/point2rbox_v2_pseudo_labels.bbox.json`, with which we can train/test/visualize the FCOS detector (2nd stage, train FCOS):

**Train**
```
# DOTA-v1.0, two gpus
bash tools/dist_train.sh projects/Point2Rbox_v2/configs/rotated-fcos-1x-dota-using-pseudo.py 2
```

**Test**
```
# DOTA-v1.0, two gpus
bash tools/dist_test.sh projects/Point2Rbox_v2/configs/rotated-fcos-1x-dota-using-pseudo.py work_dirs/rotated-fcos-1x-dota-using-pseudo/epoch_12.pth 2
```

**Resutls**
|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | lr | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :---: | :--------: | :---------------------------------------------: | :-------------------------------: |
| ResNet50 <br> (1024,1024,200) | 30.13 | 59.72  |  25.96  |   le90   |  1x  | -  | 5e-5 | 4=2gpu*<br>2img/gpu      | [rotated-fcos-1x-dota<br>-using-pseudo.py](./configs/rotated-fcos-1x-dota-using-pseudo.py) | [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2_dotav1.0_rotated-fcos-1x-dota-using-pseudo/epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2_dotav1.0_rotated-fcos-1x-dota-using-pseudo/20250824_205758/20250824_205758.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2_dotav1.0_rotated-fcos-1x-dota-using-pseudo/Task1.zip) |

Note: This is the **unofficial** checkpoint. The official code is [here](https://github.com/VisionXLab/point2rbox-v2).  
Note: The official result is **62.61 AP50** on DOTA-v1.0, but in this project the result is **59.72 AP50** on DOTA-v1.0.

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.5971806950191928  
ap of each class: plane:0.8821492602611487, baseball-diamond:0.6976029709533964, bridge:0.23295956370696166, ground-track-field:0.4422073192152871, small-vehicle:0.7981679875980286, large-vehicle:0.7618276501438279, ship:0.8708650319927672, tennis-court:0.8927201081861138, basketball-court:0.44973190926571155, storage-tank:0.8326338727989608, soccer-ball-field:0.1618861439131032, roundabout:0.41003620166821614, harbor:0.44950578316259227, swimming-pool:0.5799995657964826, helicopter:0.49541705662529406  
COCO style result:  
AP50: 0.5971806950191928  
AP75: 0.25957322565256036  
mAP: 0.30134339372567415

## Citation

```
@inproceedings{yu2025point2rbox,
  title={Point2rbox-v2: Rethinking point-supervised oriented object detection with spatial layout among instances},
  author={Yu, Yi and Ren, Botao and Zhang, Peiyuan and Liu, Mingxin and Luo, Junwei and Zhang, Shaofeng and Da, Feipeng and Yan, Junchi and Yang, Xue},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19283--19293},
  year={2025}
}
```
