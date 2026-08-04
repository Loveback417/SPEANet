# https://github.com/FishAndWasabi/YOLO-MS/blob/main/mmyolo/configs/yoloms_previous/yoloms-s_syncbn_fast_8xb32-300e_coco_previous.py

_base_ =  './yoloms_syncbn_fast_2xb4-36e_dota_previous.py'

pretrain = ('https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/'
            'yoloms_previous_version_coco/yoloms-s_syncbn_fast_8xb32-300e_coco_previous.pth')

deepen_factor = 1 / 3  # Depth scaling factor
widen_factor = 0.54  # Width scaling factor

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=pretrain),
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(
        in_channels=int(_base_.out_channels*widen_factor),
        feat_channels=int(_base_.out_channels*widen_factor),
    )
)