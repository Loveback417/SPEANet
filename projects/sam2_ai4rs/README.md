
# SAM2 AI4RS

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

The project folder holds codes related to ai4rs and SAM2.

Script Descriptions:

1. `eval_zero-shot-oriented-detection_dota.py` implement Zero-shot Oriented Object Detection with SAM2. It prompts SAM2 with predicted boxes from a horizontal object detector.
2. `demo_zero-shot-oriented-detection.py` inference single image for Zero-shot Oriented Object Detection with SAM2.
3. `data_builder` holds configuration information and process of dataset, dataloader.

The project is refer to [playground](https://github.com/open-mmlab/playground).

## Install

```shell
pip install hydra-core --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage

**NOTE: the current path is `ai4rs`**

1. Inference SAM2-AI4RS with a single image and obtain visualization result.

```shell
# download weights
mkdir ./work_dirs/sam2_ai4rs
# wget -P ./work_dirs/sam2_ai4rs https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
# wget -P ./work_dirs/sam2_ai4rs https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
# wget -P ./work_dirs/sam2_ai4rs https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
wget -P ./work_dirs/sam2_ai4rs https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
wget -P ./work_dirs/sam2_ai4rs https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dota/epoch_72.pth
```

```shell
# demo
python ./projects/sam2_ai4rs/demo_zero-shot-oriented-detection.py  ./demo/dota_demo.jpg \
    ./projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py \
    ./work_dirs/sam2_ai4rs/epoch_72.pth \
    --model-cfg configs/sam2/sam2_hiera_l.yaml --sam-weight ./work_dirs/sam2_ai4rs/sam2_hiera_large.pt --out-path output.png
```

If you want to save output masks from SAM,
```shell
python ./projects/sam2_ai4rs/demo_zero-shot-oriented-detection.py  ./demo/dota_demo.jpg \
    ./projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py \
    ./work_dirs/sam2_ai4rs/epoch_72.pth \
    --model-cfg configs/sam2/sam2_hiera_l.yaml --sam-weight ./work_dirs/sam2_ai4rs/sam2_hiera_large.pt --out-path output.png \
    --save-masks True --masks-path ./sam2_ai4rs_output_masks/
```

<div align=center>
<img src="https://github.com/user-attachments/assets/209d52b1-80e7-45e6-b112-c96164a4fd47"  width="600"/>
</div>

2. Evaluate the quantitative evaluation metric on DOTA data set.

```shell
python ./projects/sam2_ai4rs/eval_zero-shot-oriented-detection_dota.py \
    ./projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py \
    ./work_dirs/sam2_ai4rs/epoch_72.pth \
    --model-cfg configs/sam2/sam2_hiera_l.yaml --sam-weight ./work_dirs/sam2_ai4rs/sam2_hiera_large.pt
```
