from mmengine.config import read_base

with read_base():
    from .messdet_appr_4xb2_36e_dota import *

checkpoint = ('https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/'
              'resolve/master/messdet/recspnext_str_reca-1e07cda0.pth')  # noqa
is_strict = True

model.update(
    backbone=dict(
        is_strict=is_strict,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=checkpoint)),
    neck=dict(is_strict=is_strict)
)