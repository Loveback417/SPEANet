_base_ = 'yoloms-xs_syncbn_fast_2xb4-36e_dota_previous.py'

pretrain = ('https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/'
            'rotated_yoloms/yoloms_coco/yoloms-xs_syncbn_fast_8xb32-300e_coco.pth')

out_channels = 240
widen_factor = 0.42

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=pretrain),
    backbone=dict(
        arch="C3-K3579-A234",
        widen_factor=widen_factor,
        msblock_layer_type="MSBlock_kxk_1x1_Layer",
        msblock_attention_cfg=dict(
            type="GQL",
            length=3,
            size=4
        ),
        msblock_start_branch_id=2,
        msblock_mid_expand_ratio=2
    ),
    neck=dict(
        out_channels=out_channels,
        widen_factor=widen_factor,
        kernel_sizes=dict(
            top_down=[(1, 3, 3), (1, 5, 5)],
            bottom_up=[(1, 5, 5), (1, 7, 7)]
        ),
        msblock_layer_type="MSBlock_kxk_1x1_Layer",
        msblock_start_branch_id=2,
        msblock_mid_expand_ratio=2
    ),
    bbox_head=dict(
        type='RotatedYOLOMSSepBNHead',
        stacked_convs=2,
        share_conv=False,
        in_channels=int(out_channels*widen_factor),
        feat_channels=int(out_channels*widen_factor),
        groups=[4, 4, 4],
        reg_kernel_sizes=[[3, 3], [5, 3], [7, 3]],
        cls_kernel_sizes=[[3, 3], [5, 3], [7, 3]])
)