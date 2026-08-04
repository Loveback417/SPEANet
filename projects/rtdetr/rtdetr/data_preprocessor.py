import random
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmdet.structures import DetDataSample


class BatchSyncRandomResize(nn.Module):
    """Batch random resize which synchronizes the random size across ranks.

    Args:
        random_size_range (tuple, optional): The multi-scale random range
            during multi-scale training. Defaults to None.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
        interpolations (Union[str, Sequence[str]]): Algorithm used for
            torch.nn.functional.interpolation. Default: ``'bilinear'``.
        random_sizes (Sequence[int], optional): The multi-scale random size
            during multi-scale training. Defaults to None.
    """

    def __init__(self,
                 random_size_range: Optional[Tuple[int, int]] = None,
                 interval: int = 10,
                 size_divisor: int = 32,
                 interpolations: Union[str, Sequence[str]] = 'bilinear',
                 random_sizes: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.rank, self.world_size = get_dist_info()
        self._input_size = None
        self._interval = interval
        self._size_divisor = size_divisor

        if random_size_range is not None:
            assert random_sizes is None, (
                '`random_size_range` and `random_sizes` '
                'cannot be set at the same time')
            _random_sizes = list(
                range(
                    round(random_size_range[0] / size_divisor),
                    round(random_size_range[1] / size_divisor) + 1))
        else:
            assert random_size_range is None, (
                '`random_size_range` and `random_sizes` '
                'cannot be set at the same time')
            _random_sizes = []
            for size in random_sizes:
                assert size % size_divisor == 0
                _random_sizes.append(size // size_divisor)
        self._random_sizes = _random_sizes

        if isinstance(interpolations, str):
            interpolations = [interpolations]
        supported_interpolations = {'nearest', 'bilinear', 'bicubic', 'area'}
        for interp in interpolations:
            assert interp in supported_interpolations, (
                f'unsupported interpolation method: {interp}')
        self._interp = interpolations[0]
        self._interpolations = interpolations

    def forward(
        self, inputs: Tensor, data_samples: List[DetDataSample]
    ) -> Tuple[Tensor, List[DetDataSample]]:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(
                inputs, size=self._input_size, mode=self._interp)
            for data_sample in data_samples:
                img_shape = (int(data_sample.img_shape[0] * scale_y),
                             int(data_sample.img_shape[1] * scale_x))
                pad_shape = (int(data_sample.pad_shape[0] * scale_y),
                             int(data_sample.pad_shape[1] * scale_x))
                data_sample.set_metainfo({
                    'img_shape': img_shape,
                    'pad_shape': pad_shape,
                    'batch_input_shape': self._input_size
                })
                data_sample.gt_instances.bboxes[
                    ...,
                    0::2] = data_sample.gt_instances.bboxes[...,
                                                            0::2] * scale_x
                data_sample.gt_instances.bboxes[
                    ...,
                    1::2] = data_sample.gt_instances.bboxes[...,
                                                            1::2] * scale_y
                if 'masks' in data_sample.gt_instances:
                    masks = data_sample.gt_instances.masks
                    if isinstance(masks, Tensor):
                        data_sample.gt_instances.masks = F.interpolate(
                            masks.unsqueeze(0),
                            size=img_shape,
                            mode='nearest').squeeze(0)
                    else:
                        data_sample.gt_instances.masks = masks.resize(
                            img_shape)
                if 'ignored_instances' in data_sample:
                    data_sample.ignored_instances.bboxes[
                        ..., 0::2] = data_sample.ignored_instances.bboxes[
                            ..., 0::2] * scale_x
                    data_sample.ignored_instances.bboxes[
                        ..., 1::2] = data_sample.ignored_instances.bboxes[
                            ..., 1::2] * scale_y
        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size, self._interp = self._get_random_size_and_interp(
                aspect_ratio=float(w / h), device=inputs.device)
        return inputs, data_samples

    def _get_random_size_and_interp(self, aspect_ratio: float,
                                    device: torch.device) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and broadcast to
        all ranks."""
        tensor = torch.LongTensor(3).to(device)
        if self.rank == 0:
            size = random.choice(self._random_sizes)
            size = (self._size_divisor * size,
                    self._size_divisor * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]
            tensor[2] = random.randint(0, len(self._interpolations) - 1)
        # barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        interp = self._interpolations[tensor[2].item()]
        return input_size, interp