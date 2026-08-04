from mmengine.config import read_base
from mmrotate.datasets.yolov5_dota15 import YOLOv5DOTA15Dataset

with read_base():
    from .messdet_appr_4xb2_36e_dota import *


data_root = 'data/split_ss_dota1.5/'
num_classes = 16  # Number of classes for classification
dataset_type = YOLOv5DOTA15Dataset

model.update(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)),
    train_cfg = dict(
        assigner=dict(num_classes=num_classes)))
train_dataloader.update(dataset=dict(type=dataset_type, data_root=data_root))
val_dataloader.update(dataset=dict(type=dataset_type, data_root=data_root))
test_dataloader.update(dataset=dict(type=dataset_type, data_root=data_root))

