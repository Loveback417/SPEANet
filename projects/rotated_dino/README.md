# Rotated DINO (ICLR 2023)

> [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)

<!-- [ALGORITHM] -->

## Abstract

We present DINO (DETR with Improved deNoising anchOr boxes), a state-of-the-art end-to-end object detector. DINO improves over previous DETR-like models in performance and efficiency by using a contrastive way for denoising training, a mixed query selection method for anchor initialization, and a look forward twice scheme for box prediction. DINO achieves 49.4AP in 12 epochs and 51.3AP in 24 epochs on COCO with a ResNet-50 backbone and multi-scale features, yielding a significant improvement of +6.0AP and +2.7AP, respectively, compared to DN-DETR, the previous best DETR-like model. DINO scales well in both model size and data size. Without bells and whistles, after pre-training on the Objects365 dataset with a SwinL backbone, DINO obtains the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP). Compared to other models on the leaderboard, DINO significantly reduces its model size and pre-training data size while achieving better results.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644233/207820666-099e6a85-59c4-45d6-a687-91b5781d11cd.png"/>
</div>

## Results and Models

**DOTA-v1.0 (Single-Scale Training and Testing)**

| Method | Backbone | AP50 | epoch | Config | Download |
| :----: | :------: | :--: | :-----: | :-----: | :------: |
|rotated dino| R50 (1024,1024,200) | 72.80 | 12ep |   [rotated_dino_4scale_r50_2xb2_12e_dota](./configs/rotated_dino_4scale_r50_2xb2_12e_dota.py)      |  [12th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_dino/rotated_dino_4scale_r50_2xb2_12e_dota/epoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_dino/rotated_dino_4scale_r50_2xb2_12e_dota/20250919_223722/20250919_223722.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_dino/rotated_dino_4scale_r50_2xb2_12e_dota) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_dino/rotated_dino_4scale_r50_2xb2_12e_dota/Task1.zip) |


**DIOR-R (Single-Scale Training and Testing)**

| Method | Backbone | AP50 | epoch | Config | Download |
| :----: | :------: | :--: | :-----: | :-----: | :------: |
|rotated dino| R50 (800,800) | 65.78 | 12ep | [rotated_dino_4scale_r50_2xb4_12e_dior](./configs/rotated_dino_4scale_r50_2xb4_12e_dior.py)      |  [12th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_dino/rotated_dino_4scale_r50_2xb4_12e_dior/epoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_dino/rotated_dino_4scale_r50_2xb4_12e_dior/20260316_235449/20260316_235449.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_dino/rotated_dino_4scale_r50_2xb4_12e_dior) |



## Training

To train the model(s) in the paper, run this command:

```bash
# example
bash tools/dist_train.sh projects/rotated_dino/configs/rotated_dino_4scale_r50_2xb4_12e_dior.py 2
```

## Evaluation


```bash
# example
bash tools/dist_test.sh projects/rotated_dino/configs/rotated_dino_4scale_r50_2xb4_12e_dior.py your_checkpoint.pth 2
```

## Citation

```latex
@inproceedings{zhangdino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang and Zhu, Jun and Ni, Lionel and Shum, Heung-Yeung},
  booktitle={The Eleventh International Conference on Learning Representations}
}
```