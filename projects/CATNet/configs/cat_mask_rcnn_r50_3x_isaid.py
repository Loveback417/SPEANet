_base_ = [
    'isaid.py',
    'schedule_3x.py',
    '../../../configs/_base_/default_runtime.py',
    'cat_mask_rcnn_r50_fpn.py',
]

custom_imports = dict(imports=['projects.CATNet.catnet'], allow_failed_imports=False)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')