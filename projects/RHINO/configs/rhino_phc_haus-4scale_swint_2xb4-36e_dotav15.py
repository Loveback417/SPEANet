from mmengine.config import read_base
with read_base():
    from projects.rotated_dino.configs.rotated_dino_4scale_swint_2xb4_12e_dotav15 import *


from projects.RHINO.rhino import (RHINOPositiveHungarianClassificationHead, DNGroupHungarianAssigner,
                                  HausdorffCost, GDCost)



costs = [
    dict(type=FocalLossCost, weight=2.0),
    dict(type=HausdorffCost, weight=5.0, box_format='xywha'),
    dict(
        type=GDCost,
        loss_type='kld',
        fun='log1p',
        tau=1,
        sqrt=False,
        weight=5.0)
]

model.update(
    bbox_head=dict(
        type=RHINOPositiveHungarianClassificationHead,
        loss_iou=dict(
            type=GDLoss,
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=5.0)),
    dn_cfg=dict(
        noise_mode='only_xywh', # 'only_xyxy', 'only_angle', 'only_xywh', 'all_xyxya'
        group_cfg=dict(max_num_groups=30)),
    train_cfg=dict(
        assigner=dict(match_costs=costs),
        dn_assigner=dict(type=DNGroupHungarianAssigner, match_costs=costs),
    ))


# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 36
train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=4)
param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (2 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=8, enable=False)