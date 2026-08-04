# Learning to Aggregate Multi-Scale Context for Instance Segmentation in Remote Sensing Images (TNNLS 2024)

> [arxiv](https://arxiv.org/abs/2111.11057)

> [TNNLS](https://ieeexplore.ieee.org/abstract/document/10412679)

> [Official github](https://github.com/yeliudev/CATNet)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/yeliudev/CATNet/main/.github/model.svg" width="800"/>
</div>

The task of instance segmentation in remote sensing images, aiming at performing per-pixel labeling of objects at the instance level, is of great importance for various civil applications. Despite previous successes, most existing instance segmentation methods designed for natural images encounter sharp performance degradations when they are directly applied to top-view remote sensing images. Through careful analysis, we observe that the challenges mainly come from the lack of discriminative object features due to severe scale variations, low contrasts, and clustered distributions. In order to address these problems, a novel context aggregation network (CATNet) is proposed to improve the feature extraction process. The proposed model exploits three lightweight plug-and-play modules, namely, dense feature pyramid network (DenseFPN), spatial context pyramid (SCP), and hierarchical region of interest extractor (HRoIE), to aggregate global visual context at feature, spatial, and instance domains, respectively. DenseFPN is a multi-scale feature propagation module that establishes more flexible information flows by adopting interlevel residual connections, cross-level dense connections, and feature reweighting strategy. Leveraging the attention mechanism, SCP further augments the features by aggregating global spatial context into local regions. For each instance, HRoIE adaptively generates RoI features for different downstream tasks.

## Results and models

|  Dataset |  Model  | Backbone | Schd | Aug | BBox AP | Mask AP | Download |                                                                                                                                                                                                   
| :-------: | :---: | :---: | :-----: | :------: | :------------: | :-------: | :--------: | 
| iSAID (800*800) | [CAT Mask R-CNN](./configs/cat_mask_rcnn_r50_3x_isaid.py) | ResNet-50  |   3x  |  X   |  45.1  |  37.2 | [ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/CATNet/cat_mask_rcnn_r50_3x_isaid/cat_mask_rcnn_r50_3x_isaid-384df911.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/CATNet/cat_mask_rcnn_r50_3x_isaid/checkpoints_cat_mask_rcnn_r50_3x_isaid.txt)    |


NOTE: The results and checkpoints come from [official github](https://github.com/yeliudev/CATNet).

NOTE: We does not retrain the model. The results, log, and checkpoints are reported in the original repo.

NOTE: Official repo said: "All the models are trained using 4 NVIDIA A100 GPUs and are evaluated using the default metrics of the datasets".

Train

Single-node multi-GPU, for example 4 gpus:

```shell 
bash tools/dist_train.sh projects/CATNet/configs/cat_mask_rcnn_r50_3x_isaid.py 4
```

> If an `out-of-memory` error occurs on iSAID dataset, please uncomment [L15-L17](./catnet/isaid.py#L15:L17) in the dataset code and try again. This will filter out a few images with more than 1,000 objects, largely reducing the memory cost.


Test

Single-node multi-GPU, for example 4 gpus:

```shell
 bash tools/dist_test.sh projects/CATNet/configs/cat_mask_rcnn_r50_3x_isaid.py your_path/cat_mask_rcnn_r50_3x_isaid-384df911.pth 4
```

Result Visualization
```shell
python demo/image_demo.py demo/demo.jpg projects/CATNet/configs/cat_mask_rcnn_r50_3x_isaid.py your_path/cat_mask_rcnn_r50_3x_isaid-384df911.pth --out-file your_path/demo_result.jpg --palette random
```


## Citation

```
@article{liu2024learning,
  title={Learning to aggregate multi-scale context for instance segmentation in remote sensing images},
  author={Liu, Ye and Li, Huifang and Hu, Chao and Luo, Shuang and Luo, Yan and Chen, Chang Wen},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={36},
  number={1},
  pages={595--609},
  year={2024},
  publisher={IEEE}
}
```
