
# Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks (Arxiv 2024)

[Official Repo](https://github.com/IDEA-Research/Grounded-Segment-Anything)

## Installation

```shell
pip install transformers==4.46.0 nltk -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage

**NOTE: the current path is `ai4rs`**

1. Download checkpoint

```shell
# download weights
mkdir ./work_dirs/grounded_sam
wget -P ./work_dirs/grounded_sam https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth
wget -P ./work_dirs/grounded_sam https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

2. Inference Grounded-SAM with a single image and obtain visualization result.

```
# demo
python ./projects/grounded_sam/grounded_sam_demo.py \
    ./demo/det.jpg \
    'mmdet::grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py' \
    ./work_dirs/grounded_sam/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
    --texts 'bench . car .' \
    --sam-type 'vit_b' --sam-weight ./work_dirs/grounded_sam/sam_vit_b_01ec64.pth \
    --nltk_root ./nltk_data/ --out-path output.png
```


<div align=center>
<img src="https://github.com/user-attachments/assets/6a5b2e6c-01fe-4bbc-820c-693a4fef3ffe" width=600/>
</div>


## BibTeX

```shell
@article{ren2024grounded,
  title={Grounded sam: Assembling open-world models for diverse visual tasks},
  author={Ren, Tianhe and Liu, Shilong and Zeng, Ailing and Lin, Jing and Li, Kunchang and Cao, He and Chen, Jiayu and Huang, Xinyu and Chen, Yukang and Yan, Feng and others},
  journal={arXiv preprint arXiv:2401.14159},
  year={2024}
}
```