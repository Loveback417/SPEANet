# Rethinking Boundary Discontinuity Problem for Oriented Object Detection

> [Rethinking Boundary Discontinuity Problem for Oriented Object Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Rethinking_Boundary_Discontinuity_Problem_for_Oriented_Object_Detection_CVPR_2024_paper.html)

> Official Code [Link](https://github.com/Pandora-CV/cvpr24acm)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/Pandora-CV/cvpr24acm/raw/main/images/rotation_invarience.png" width="800"/>
</div>

Oriented object detection has been developed rapidly in the past few years where rotation equivariance is crucial for detectors to predict rotated boxes. It is expected that the prediction can maintain the corresponding rotation when objects rotate but severe mutation in angular prediction is sometimes observed when objects rotate near the boundary angle which is well-known boundary discontinuity problem. The problem has been long believed to be caused by the sharp loss increase at the angular boundary and widely used joint-optim IoU-like methods deal with this problem by loss-smoothing. However we experimentally find that even state-of-the-art IoU-like methods actually fail to solve the problem. On further analysis we find that the key to solution lies in encoding mode of the smoothing function rather than in joint or independent optimization. In existing IoU-like methods the model essentially attempts to fit the angular relationship between box and object where the break point at angular boundary makes the predictions highly unstable. To deal with this issue we propose a dual-optimization paradigm for angles. We decouple reversibility and joint-optim from single smoothing function into two distinct entities which for the first time achieves the objectives of both correcting angular boundary and blending angle with other parameters. Extensive experiments on multiple datasets show that boundary discontinuity problem is well-addressed. Moreover typical IoU-like methods are improved to the same level without obvious performance gap.

## Results and models

**DIOR-R**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,800,200) | 35.33 | 60.23  |  34.82  |   le90   |      1x      |  -  | 2=1gpu*<br>2img/gpu      | [rotated-fcos-le90_r50_fpn_acm_1x_dior.py](./configs/rotated-fcos-le90_r50_fpn_acm_1x_dior.py) | [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ACM/rotated-fcos-le90_r50_fpn_acm_1x_dior/epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ACM/rotated-fcos-le90_r50_fpn_acm_1x_dior/20250720_163940/20250720_163940.log) \| [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) |

Note: This is the **unofficial** checkpoint. The official code is [here](https://github.com/Pandora-CV/cvpr24acm). 



**Train**

```
python tools/train.py projects/ACM/configs/rotated-fcos-le90_r50_fpn_acm_1x_dior.py
```

**Test**
```
python tools/test.py projects/ACM/configs/rotated-fcos-le90_r50_fpn_acm_1x_dior.py work_dirs/rotated-fcos-le90_r50_fpn_acm_1x_dior/epoch_12.pth
```


**DOTA-v1.0**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :---: | :--------: | :---------------------------------------------: | :-------------------------------: |
| ResNet50 <br> (1024,1024,200) | 41.26 | 70.66 |  42.64  |   le90   |  1x  | -  |  2=1gpu*<br>2img/gpu      | [rotated-fcos-le90_r50_<br>fpn_acm_1x_dota.py](./configs/rotated-fcos-le90_r50_fpn_acm_1x_dota.py) | [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ACM/rotated-fcos-le90_r50_fpn_acm_1x_dota/epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ACM/rotated-fcos-le90_r50_fpn_acm_1x_dota/20250720_233414/20250720_233414.log) \| <br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ACM/rotated-fcos-le90_r50_fpn_acm_1x_dota/Task1.zip)|

Note: This is the **unofficial** checkpoint. The official code is [here](https://github.com/Pandora-CV/cvpr24acm).  

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7065559995751901  
ap of each class: plane:0.8806294055023749, baseball-diamond:0.7399491614299935, bridge:0.4907825390970488, ground-track-field:0.6154156427181114, small-vehicle:0.7972950196384038, large-vehicle:0.7723134420058748, ship:0.8707195250417594, tennis-court:0.9088853874083718, basketball-court:0.8217234880576788, storage-tank:0.8403756495997674, soccer-ball-field:0.569704951607838, roundabout:0.6121159959421891, harbor:0.639052894061831, swimming-pool:0.67781914686964, helicopter:0.36155774464696927  
COCO style result:  
AP50: 0.7065559995751901  
AP75: 0.4264321623730036  
mAP: 0.4126253690505995

**Train**
```
python tools/train.py projects/ACM/configs/rotated-fcos-le90_r50_fpn_acm_1x_dota.py
```

**Test**
```
python tools/test.py projects/ACM/configs/rotated-fcos-le90_r50_fpn_acm_1x_dota.py work_dirs/rotated-fcos-le90_r50_fpn_acm_1x_dota/epoch_12.pth
```


## Citation

```
@InProceedings{Xu_2024_CVPR,
    author    = {Xu, Hang and Liu, Xinyuan and Xu, Haonan and Ma, Yike and Zhu, Zunjie and Yan, Chenggang and Dai, Feng},
    title     = {Rethinking Boundary Discontinuity Problem for Oriented Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17406-17415}
}
```
