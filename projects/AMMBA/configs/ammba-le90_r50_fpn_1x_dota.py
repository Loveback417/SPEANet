from mmengine.config import read_base
from mmdet.models import RetinaNet, ResNet, FPN
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.losses import FocalLoss, CrossEntropyLoss
from mmdet.models.task_modules import PseudoSampler
from mmrotate.models import RotatedATSSHead
from mmrotate.models.task_modules import FakeRotatedAnchorGenerator, DeltaXYWHTRBBoxCoder, RBboxOverlaps2D
from mmrotate.models.losses import RotatedIoULoss
from projects.AMMBA.ammba import BFPA, MeanMaxAssigner

with read_base():
    from configs._base_.datasets.dota import *
    from configs._base_.schedules.schedule_1x import *
    from configs._base_.default_runtime import *

angle_version = 'le90'

model = dict(
    type=RetinaNet,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=[
        dict(
            type=FPN,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5),
        dict(
            type=BFPA,
            in_channels=256,
            out_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')],
    bbox_head=dict(
        type=RotatedATSSHead,
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type=FakeRotatedAnchorGenerator,
            angle_version=angle_version,
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type=DeltaXYWHTRBBoxCoder,
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type=RotatedIoULoss, mode='linear', loss_weight=2.0),
        loss_centerness=dict(
            type=CrossEntropyLoss, use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type=MeanMaxAssigner,
            topk=9,
            iou_calculator=dict(type=RBboxOverlaps2D)),
        sampler=dict(
            type=PseudoSampler),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

train_cfg.update(val_interval=4)
find_unused_parameters=True