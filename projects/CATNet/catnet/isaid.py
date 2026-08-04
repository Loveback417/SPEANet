from mmdet.datasets import Objects365V1Dataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class CATNet_ISAIDDataset(Objects365V1Dataset):

    METAINFO = {
        'classes':
        ('ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
        'basketball_court', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle',
        'Small_Vehicle', 'Helicopter', 'Swimming_pool', 'Roundabout',
        'Soccer_ball_field', 'plane', 'Harbor')}

    def filter_data(self):
        data_info = super().filter_data()
        return [d for d in data_info if len(d['instances']) <= 1000]