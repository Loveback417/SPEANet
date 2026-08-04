from typing import Sequence, Union, Tuple
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from numpy import random
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type

Number = Union[int, float]

class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Required Keys:

    - img (np.uint8)

    Modified Keys:

    - img (np.float32)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[Number] = (0.5, 1.5),
                 saturation_range: Sequence[Number] = (0.5, 1.5),
                 hue_delta: int = 18,
                 swap_channel: bool = True,
                 clip_val: int = -1,
                 force_float32: bool = True) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.swap_channel = swap_channel
        self.clip_val = clip_val
        self.force_float32 = force_float32

    @cache_randomness
    def _random_flags(self) -> Sequence[Number]:
        mode = random.randint(2)
        brightness_flag = random.randint(2)
        contrast_flag = random.randint(2)
        saturation_flag = random.randint(2)
        hue_flag = random.randint(2)
        swap_flag = random.randint(2) if self.swap_channel else 0
        delta_value = random.uniform(-self.brightness_delta,
                                     self.brightness_delta)
        alpha_value = random.uniform(self.contrast_lower, self.contrast_upper)
        saturation_value = random.uniform(self.saturation_lower,
                                          self.saturation_upper)
        hue_value = random.uniform(-self.hue_delta, self.hue_delta)
        swap_value = random.permutation(3)

        return (mode, brightness_flag, contrast_flag, saturation_flag,
                hue_flag, swap_flag, delta_value, alpha_value,
                saturation_value, hue_value, swap_value)

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        assert 'img' in results, '`img` is not found in results'
        img = results['img']
        ori_dtype = img.dtype
        img = img.astype(np.float32)

        (mode, brightness_flag, contrast_flag, saturation_flag, hue_flag,
         swap_flag, delta_value, alpha_value, saturation_value, hue_value,
         swap_value) = self._random_flags()

        # random brightness
        if brightness_flag:
            img += delta_value
            if self.clip_val > 0:
                img = img.clip(0, self.clip_val)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if mode == 1:
            if contrast_flag:
                img *= alpha_value
                if self.clip_val > 0:
                    img = img.clip(0, self.clip_val)

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if saturation_flag:
            img[..., 1] *= saturation_value
            # For image(type=float32), after convert bgr to hsv by opencv,
            # valid saturation value range is [0, 1]
            if saturation_value > 1:
                img[..., 1] = img[..., 1].clip(0, 1)

        # random hue
        if hue_flag:
            img[..., 0] += hue_value
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if contrast_flag:
                img *= alpha_value
                if self.clip_val > 0:
                    img = img.clip(0, self.clip_val)

        if not self.force_float32:
            img = img.astype(ori_dtype, copy=False)

        # randomly swap channels
        if swap_flag:
            img = img[..., swap_value]

        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(brightness_delta={self.brightness_delta}, '
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)}, '
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)}, '
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str

class MinIoURandomCrop(BaseTransform):
    """Random crop the image & bboxes & masks & segmentation map, the cropped
    patches have minimum IoU requirement with original image & bboxes & masks.

    & segmentation map, the IoU threshold is randomly selected from min_ious.


    Required Keys:

    - img
    - img_shape
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_bboxes_labels
    - gt_masks
    - gt_ignore_flags
    - gt_seg_map


    Args:
        min_ious (Sequence[float]): minimum IoU threshold for all intersections
            with bounding boxes.
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
            where a >= min_crop_size).
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        cover_all_box (bool, optional): ensure all bboxes are covered in
            the final crop.
        trials (int, optional): Number of trials to find a crop for a given
            value of minimal IoU (Jaccard) overlap. Default, 40.
    """

    def __init__(self,
                 min_ious: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size: float = 0.3,
                 bbox_clip_border: bool = True,
                 cover_all_box: bool = True,
                 trials: int = 50) -> None:
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border
        self.cover_all_box = cover_all_box
        self.trials = trials

    @cache_randomness
    def _random_mode(self) -> Number:
        return random.choice(self.sample_mode)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to crop images and bounding boxes with minimum
        IoU constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """
        assert 'img' in results, '`img` is not found in results'
        assert 'gt_bboxes' in results, '`gt_bboxes` is not found in results'
        img = results['img']
        boxes = results['gt_bboxes']
        h, w, c = img.shape
        while True:
            mode = self._random_mode()
            self.mode = mode
            if mode == 1:
                return results

            min_iou = self.mode
            for i in range(self.trials):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = boxes.overlaps(
                    HorizontalBoxes(patch.reshape(-1, 4).astype(np.float32)),
                    boxes).numpy().reshape(-1)

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    if overlaps.max() < min_iou:
                        continue

                    if self.cover_all_box and overlaps.min() < min_iou:
                        continue

                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        centers = boxes.centers.numpy()
                        mask = ((centers[:, 0] > patch[0]) *
                                (centers[:, 1] > patch[1]) *
                                (centers[:, 0] < patch[2]) *
                                (centers[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    if results.get('gt_bboxes', None) is not None:
                        boxes = results['gt_bboxes']
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        boxes.translate_([-patch[0], -patch[1]])
                        if self.bbox_clip_border:
                            boxes.clip_(
                                [patch[3] - patch[1], patch[2] - patch[0]])
                        results['gt_bboxes'] = boxes

                        # ignore_flags
                        if results.get('gt_ignore_flags', None) is not None:
                            results['gt_ignore_flags'] = \
                                results['gt_ignore_flags'][mask]

                        # labels
                        if results.get('gt_bboxes_labels', None) is not None:
                            results['gt_bboxes_labels'] = results[
                                'gt_bboxes_labels'][mask]

                        # mask fields
                        if results.get('gt_masks', None) is not None:
                            results['gt_masks'] = results['gt_masks'][
                                mask.nonzero()[0]].crop(patch)
                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape[:2]

                # seg fields
                if results.get('gt_seg_map', None) is not None:
                    results['gt_seg_map'] = results['gt_seg_map'][
                        patch[1]:patch[3], patch[0]:patch[2]]
                return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_ious}, '
        repr_str += f'min_crop_size={self.min_crop_size}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


class Expand(BaseTransform):
    """Random expand the image & bboxes & masks & segmentation map.

    Randomly place the original image on a canvas of ``ratio`` x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_masks
    - gt_seg_map


    Args:
        mean (sequence): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (sequence)): range of expand ratio.
        seg_ignore_label (int): label of ignore segmentation map.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean: Sequence[Number] = (0, 0, 0),
                 to_rgb: bool = True,
                 ratio_range: Sequence[Number] = (1, 4),
                 seg_ignore_label: int = None,
                 prob: float = 0.5) -> None:
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    @cache_randomness
    def _random_ratio(self) -> float:
        return random.uniform(self.min_ratio, self.max_ratio)

    @cache_randomness
    def _random_left_top(self, ratio: float, h: int,
                         w: int) -> Tuple[int, int]:
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        return left, top

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to expand images, bounding boxes, masks,
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes, masks, segmentation
                map expanded.
        """
        if self._random_prob() > self.prob:
            return results
        assert 'img' in results, '`img` is not found in results'
        img = results['img']
        h, w, c = img.shape
        ratio = self._random_ratio()
        # speedup expand when meets large image
        if len(set(self.mean)) == 1:
            expand_img = np.empty((int(h * ratio), int(w * ratio), c),
                                  img.dtype)
            expand_img.fill(self.mean[0])
        else:
            expand_img = np.full((int(h * ratio), int(w * ratio), c),
                                 self.mean,
                                 dtype=img.dtype)
        left, top = self._random_left_top(ratio, h, w)
        expand_img[top:top + h, left:left + w] = img
        results['img'] = expand_img
        results['img_shape'] = expand_img.shape[:2]

        # expand bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].translate_([left, top])

        # expand masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].expand(
                int(h * ratio), int(w * ratio), top, left)

        # expand segmentation map
        if results.get('gt_seg_map', None) is not None:
            gt_seg = results['gt_seg_map']
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label,
                                    dtype=gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results['gt_seg_map'] = expand_gt_seg

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, to_rgb={self.to_rgb}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob})'
        return repr_str