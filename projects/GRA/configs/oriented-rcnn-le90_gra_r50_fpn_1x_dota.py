_base_ = [
    '../../../configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'
]

custom_imports = dict(
    imports=['projects.GRA.gra'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        type='GRAResNet',
        replace = [
            ['x'],
            ['0', '1', '2', '3'],
            ['0', '1', '2', '3', '4', '5'],
            ['0', '1', '2']
        ],
        kernel_number = 1,
        num_groups = 32,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://www.modelscope.cn/models/wokaikaixinxin/'
                       'ai4rs/resolve/master/GRA/GRA-ResNet50.pth')),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.4)})
)

train_cfg = dict(val_interval=12)

# base_batch_size = (1 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=2, enable=False)