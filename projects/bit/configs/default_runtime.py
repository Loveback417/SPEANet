default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='CDLocalVisBackend')] # mmrotate/visualization/vis_backend_cd.py
visualizer = dict(
    type='CDLocalVisualizer',   # mmrotate/visualization/local_visualizer_cd.py
    vis_backends=vis_backends, name='visualizer', alpha=1.0)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='mmseg.SegTTAModel')