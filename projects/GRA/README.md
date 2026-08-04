# GRA: Detecting Oriented Objects through Group-wise Rotating and Attention

> [GRA: Detecting Oriented Objects through Group-wise Rotating and Attention](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02600.pdf)

> Official Code [Link](https://github.com/wangjiangshan0725/GRA)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/wangjiangshan0725/GRA/raw/master/figs/module.png" width="800"/>
</div>

Oriented object detection, an emerging task in recent years, aims to identify and locate objects across varied orientations. This requires the detector to accurately capture the orientation information, which varies significantly within and across images. Despite the existing substantial efforts, simultaneously ensuring model effectiveness and parameter efficiency remains challenging in this scenario. In this paper, we propose a lightweight yet effective Group-wise Rotating and Attention (GRA) module to replace the convolution operations in backbone networks for oriented object detection. GRA can adaptively capture finegrained features of objects with diverse orientations, comprising two key components: Group-wise Rotating and Group-wise Attention. Groupwise Rotating first divides the convolution kernel into groups, where each group extracts different  object features by rotating at a specific angle according to the object orientation. Subsequently, Group-wise Attention is employed to adaptively enhance the object-related regions in the feature. The collaborative effort of these components enables GRA to effectively capture the various orientation information while maintaining parameter efficiency. Extensive experimental results demonstrate the superiority of our method. For example, GRA achieves a new state-of-the-art (SOTA) on the DOTA-v2.0 benchmark, while saving the parameters by nearly 50% compared to the previous SOTA methods.


## Results and models

**Pretrain**

You can get the pretrained weight of GRA-ResNet50 from [Modelscope](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GRA/GRA-ResNet50.pth)

**DOTA-v1.0**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | lr | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :---: | :--------: | :---------------------------------------------: | :-------------------------------: |
| GRA-Res50 <br> (1024,1024,200) | 46.67 | 76.50  |  50.43  |   le90   |  1x  | -  | 5e-3 | 2=1gpu*<br>2img/gpu      | [oriented-rcnn-le90_gra_<br>r50_fpn_1x_dota.py](./configs/oriented-rcnn-le90_gra_r50_fpn_1x_dota.py) | [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GRA/oriented-rcnn-le90_gra_r50_fpn_1x_dota/epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GRA/oriented-rcnn-le90_gra_r50_fpn_1x_dota/20250825_195535/20250825_195535.log) \| <br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GRA/oriented-rcnn-le90_gra_r50_fpn_1x_dota/Task1.zip)|

Note: This is the **unofficial** checkpoint. The official code is [here](https://github.com/wangjiangshan0725/GRA).  
Note: The official result is **77.65 AP50, 51.77 AP75, 47.91 mAP** on DOTA-v1.0, but in this project the result is **76.50 AP50, 50.43 AP75, 46.67 mAP** on DOTA-v1.0. The official detailed results are shown in [issues #1](https://github.com/wangjiangshan0725/GRA/issues/1).

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.764966710102518  
ap of each class: plane:0.8915046857771493, baseball-diamond:0.8049205441344987, bridge:0.5339044473076197, ground-track-field:0.7187059629718775, small-vehicle:0.7881238524345027, large-vehicle:0.8453599585442728, ship:0.8801477521215209, tennis-court:0.9079193762344024, basketball-court:0.869623001685805, storage-tank:0.8479346560345278, soccer-ball-field:0.6211291443787291, roundabout:0.6472616008665192, harbor:0.7327152419042207, swimming-pool:0.7152175244561494, helicopter:0.670032902685976  
COCO style result:  
AP50: 0.764966710102518  
AP75: 0.5043323747552313  
mAP: 0.4667206777295608


**Train**

```
python tools/train.py config_path
``` 

For example:

```
python tools/train.py projects/GRA/configs/oriented-rcnn-le90_gra_r50_fpn_1x_dota.py
```


**Test**
```
python tools/test.py config_path checkpoint_path
```  

For example:

```
python tools/test.py projects/GRA/configs/oriented-rcnn-le90_gra_r50_fpn_1x_dota.py work_dirs/oriented-rcnn-le90_gra_r50_fpn_1x_dota/epoch_12.pth
```


## Citation

```
@inproceedings{wang2024gra,
  title={Gra: Detecting oriented objects through group-wise rotating and attention},
  author={Wang, Jiangshan and Pu, Yifan and Han, Yizeng and Guo, Jiayi and Wang, Yiru and Li, Xiu and Huang, Gao},
  booktitle={European Conference on Computer Vision},
  pages={298--315},
  year={2024},
  organization={Springer}
}
```
