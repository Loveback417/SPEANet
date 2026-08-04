# Mask R-CNN

> [Mask R-CNN](https://arxiv.org/abs/1703.06870)

<!-- [ALGORITHM] -->

## Abstract

We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143967081-c2552bed-9af2-46c4-ae44-5b3b74e5679f.png"/>
</div>

## Results and Models

**NWPU VHR-10**

| Backbone | Lr schd | box AP | box AP50 | box AP75 | mask AP | mask AP50 | mask AP75 | Config | Download |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| R-50-FPN | 2x | 67.1 | 92.0 | 78.6 | 65.8 | 91.6 | 72.1 | [config](./configs/mask_rcnn_r50_fpn_2x_nwpu.py) | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/mask_rcnn/mask_rcnn_r50_fpn_2x_nwpu/epoch_24.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/mask_rcnn/mask_rcnn_r50_fpn_2x_nwpu/20260403_141226/20260403_141226.log) \| [all ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/mask_rcnn/mask_rcnn_r50_fpn_2x_nwpu) |


## Citation

```latex
@article{He_2017,
   title={Mask R-CNN},
   journal={2017 IEEE International Conference on Computer Vision (ICCV)},
   publisher={IEEE},
   author={He, Kaiming and Gkioxari, Georgia and Dollar, Piotr and Girshick, Ross},
   year={2017},
   month={Oct}
}
```