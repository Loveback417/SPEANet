# https://github.com/FishAndWasabi/YOLO-MS/blob/main/mmyolo/configs/yoloms_previous/yoloms_syncbn_fast_8xb32-300e_coco_previous.py

_base_ =  '../../../configs/rotated_rtmdet/rotated_rtmdet_l-3x-dota.py'

custom_imports = dict(
    imports = ['projects.rotated_yoloms.rotated_yoloms'], allow_failed_imports=False)

pretrain = ('https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/'
            'yoloms_previous_version_coco/yoloms_syncbn_fast_8xb32-300e_coco_previou.pth')

deepen_factor = 2 / 3  # Depth scaling factor
widen_factor = 0.8  # Width scaling factor

# PAFPN Channels
in_channels = [320, 640, 1280]  # Input channels
mid_channels = [160, 320, 640]  # Middle channels
out_channels = 240  # Output channels

# MS-Block Configurations
msblock_layer_type = "MSBlockBottleNeckLayer"
backbone_msblock_down_ratio = 1  # Downsample ratio in Backbone
neck_msblock_down_ratio = 0.5  # Downsample ratio in PAFPN
msblock_mid_expand_ratio = 2  # Channel expand ratio for each branch
msblock_layers_num = 3  # Number of layers in MS-Block
msblock_channel_split_ratios = [1, 1, 1]  # Channel split ratios

# Normalization and Activation Configurations
norm_cfg = dict(type='BN')  # Normalization config
act_cfg = dict(type='SiLU', inplace=True)  # Activation config

# Kernel Sizes for MS-Block in PAFPN
kernel_sizes = dict(
    bottom_up=[[1, (3, 3), (3, 3)], [1, (3, 3), (3, 3)]],
    top_down=[[1, (3, 3), (3, 3)], [1, (3, 3), (3, 3)]]
)

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=pretrain),
    backbone=dict(
        _delete_=True,
        type='YOLOMS',
        arch='C3-K3579',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        norm_eval=False,
        act_cfg=act_cfg,
        msblock_layer_type=msblock_layer_type,
        msblock_down_ratio=backbone_msblock_down_ratio,
        msblock_mid_expand_ratio=msblock_mid_expand_ratio,
        msblock_layers_num=msblock_layers_num,
        msblock_norm_cfg=norm_cfg,
        msblock_act_cfg=act_cfg
    ),
    neck=dict(
        _delete_=True,
        type='YOLOMSPAFPN',
        in_channels=in_channels,
        mid_channels=mid_channels,
        out_channels=out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        kernel_sizes=kernel_sizes,
        msblock_layer_type=msblock_layer_type,
        msblock_down_ratio=neck_msblock_down_ratio,
        msblock_mid_expand_ratio=msblock_mid_expand_ratio,
        msblock_layers_num=msblock_layers_num,
        msblock_channel_split_ratios=msblock_channel_split_ratios,
        msblock_act_cfg=act_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg
    ),
    bbox_head=dict(
        in_channels=int(out_channels*widen_factor),
        feat_channels=int(out_channels*widen_factor),
        act_cfg=dict(inplace=True, type='LeakyReLU')
    )
)

# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = dict(batch_size=4, num_workers=4)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=999))

train_cfg = dict(val_interval=1)


auto_scale_lr = dict(base_batch_size=8, enable=False)