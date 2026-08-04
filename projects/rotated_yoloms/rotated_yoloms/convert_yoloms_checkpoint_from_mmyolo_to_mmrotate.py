import torch
from collections import OrderedDict

# https://github.com/FishAndWasabi/YOLO-MS/blob/main/docs/model_zoos.md
# -----------------------------------------------------------------------------------------------------------------
# yoloms_previous_version
# -----------------------------------------------------------------------------------------------------------------
# |yoloms_syncbn_fast_8xb32-300e_coco_previous.py   |yoloms_syncbn_fast_8xb32-300e_coco-f8e265a4-6fc78bdc-d6517016.pth
# |yoloms-s_syncbn_fast_8xb32-300e_coco_previous.py |yoloms-s_syncbn_fast_8xb32-300e_coco-3c287073-ff478592-3403211e.pth
# |yoloms-xs_syncbn_fast_8xb32-300e_coco_previous.py|yoloms-xs_syncbn_fast_8xb32-300e_coco-ba5ca70c-409b3fd6-b3990f9a.pth
# -----------------------------------------------------------------------------------------------------------------
# yoloms
# -----------------------------------------------------------------------------------------------------------------
# yoloms_syncbn_fast_8xb32-300e_coco.py    | yoloms_syncbn_fast_8xb32-300e_coco-0d9c8664-878989a7.pth
# yoloms-s_syncbn_fast_8xb32-300e_coco.py  | yoloms-s_syncbn_fast_8xb32-300e_coco-32f45010-8109ab67.pth
# yoloms-xs_syncbn_fast_8xb32-300e_coco.py | yoloms-xs_syncbn_fast_8xb32-300e_coco-9643af25-ba69c092.pth
# -----------------------------------------------------------------------------------------------------------------


pth_path = 'yoloms_syncbn_fast_8xb32-300e_coco-f8e265a4-6fc78bdc-d6517016.pth'
new_path = 'yoloms_syncbn_fast_8xb32-300e_coco_previou.pth'

# pth_path = 'yoloms-s_syncbn_fast_8xb32-300e_coco-3c287073-ff478592-3403211e.pth'
# new_path = 'yoloms-s_syncbn_fast_8xb32-300e_coco_previous.pth'

# pth_path = 'yoloms-xs_syncbn_fast_8xb32-300e_coco-ba5ca70c-409b3fd6-b3990f9a.pth'
# new_path = 'yoloms-xs_syncbn_fast_8xb32-300e_coco_previous.pth'

# pth_path = 'yoloms_syncbn_fast_8xb32-300e_coco-0d9c8664-878989a7.pth'
# new_path = 'yoloms_syncbn_fast_8xb32-300e_coco.pth'

# pth_path = 'yoloms-s_syncbn_fast_8xb32-300e_coco-32f45010-8109ab67.pth'
# new_path = 'yoloms-s_syncbn_fast_8xb32-300e_coco.pth'

# pth_path = 'yoloms-xs_syncbn_fast_8xb32-300e_coco-9643af25-ba69c092.pth'
# new_path = 'yoloms-xs_syncbn_fast_8xb32-300e_coco.pth'

src_model = torch.load(pth_path)
print(type(src_model['state_dict']))
print(len(src_model['state_dict']))
dst_state_dict = OrderedDict()
for k, v in src_model['state_dict'].items():    # type=<class 'collections.OrderedDict'>
    if 'head_module.' in k:
        print(k)
        new_name = k.replace('head_module.', '')
        print(new_name)
        dst_state_dict[new_name] = v
    else:
        dst_state_dict[k] = v

new_model = dict(meta=src_model['meta'],
                 state_dict=dst_state_dict,
                 param_schedulers=src_model['param_schedulers']
                 )
torch.save(new_model, new_path)