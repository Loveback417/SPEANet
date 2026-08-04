from typing import Tuple, List, Optional, Sequence
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from mmengine.model import bias_init_with_prob
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmcv.ops.nms import batched_nms
from mmdet.utils import (ConfigType, OptInstanceList, reduce_mean)
from mmdet.models.dense_heads.yolox_head import YOLOXHead
from mmdet.models.utils import multi_apply
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmrotate.structures.bbox.transforms import norm_angle
from mmrotate.registry import MODELS, TASK_UTILS

@MODELS.register_module()
class RotatedYOLOXHead(YOLOXHead):
    """
    Args:
        angle_version (str):
        separate_angle (bool): If true, angle prediction is separated from
            bbox regression loss. Default: False.
        with_angle_l1 (bool): If true, compute L1 loss with angle.
            Default: True.
        angle_norm_factor (float): Regularization factor of angle. Only
            used when with_angle_l1 is True
        angle_coder (dict): Config of angle coder.
        loss_angle (dict): Config of angle loss, only used when
            separate_angle is True.
    """
    def __init__(self,
                 angle_version: str = 'le90',
                 separate_angle: bool = False,
                 with_angle_l1: bool = True,
                 angle_norm_factor: float = np.pi,
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 loss_angle: ConfigType = dict(type='SmoothFocalLoss'),
                 **kwargs):
        self.angle_coder = TASK_UTILS.build(angle_coder)
        super().__init__(**kwargs)
        assert angle_version in ['oc', 'le90', 'le135']
        self.angle_version = angle_version
        self.separate_angle = separate_angle
        self.with_angle_l1 = with_angle_l1
        self.angle_norm_factor = angle_norm_factor
        if separate_angle:
            self.loss_angle = MODELS.build(loss_angle)

    def _init_layers(self) -> None:
        """Initialize heads for all level feature maps."""
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        self.multi_level_conv_ang = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj, conv_ang = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)
            self.multi_level_conv_ang.append(conv_ang)

    def _build_predictor(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Initialize predictor layers of a single level head."""
        conv_cls, conv_reg, conv_obj = super()._build_predictor()
        conv_ang = nn.Conv2d(self.feat_channels, self.angle_coder.encode_size, 1)
        return conv_cls, conv_reg, conv_obj, conv_ang

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_ang in self.multi_level_conv_ang:
            conv_ang.bias.data.fill_(bias_init)

    def forward_single(self, x: Tensor, cls_convs: nn.Module,
                       reg_convs: nn.Module, conv_cls: nn.Module,
                       conv_reg: nn.Module, conv_obj: nn.Module,
                       conv_ang: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        angle_pred = conv_ang(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, angle_pred, objectness

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """

        return multi_apply(self.forward_single, x, self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj,
                           self.multi_level_conv_ang)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angles for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 5),
              the last dimension 5 arrange as (cx, cy, w, h, angle).
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)
        flatten_decoded_angle = self.angle_coder.decode(flatten_angle_preds).unsqueeze(-1)
        flatten_rbboxes = self._bbox_decode_cxcywha(flatten_priors, flatten_bbox_preds,
                                                    flatten_decoded_angle)

        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            max_scores, labels = torch.max(flatten_cls_scores[img_id], 1)
            valid_mask = flatten_objectness[
                img_id] * max_scores >= cfg.score_thr
            results = InstanceData(
                bboxes=flatten_rbboxes[img_id][valid_mask],
                scores=max_scores[valid_mask] *
                flatten_objectness[img_id][valid_mask],
                labels=labels[valid_mask])

            result_list.append(
                self._bbox_post_process(
                    results=results,
                    cfg=cfg,
                    rescale=rescale,
                    with_nms=with_nms,
                    img_meta=img_meta))

        return result_list

    def _bbox_decode_cxcywha(self, priors: Tensor, bbox_preds: Tensor, decoded_angle: Tensor) -> Tensor:
        """
        Args:
            priors (Tensor): Center proiors of an image, has shape
                (num_instances, 2).
            bbox_preds (Tensor): Box energies / deltas for all instances,
                has shape (batch_size, num_instances, 4).
            decoded_angle (Tensor): ().

        Returns:
            Tensor: Decoded bboxes in (cx, cy, w, h, angle) format. Has
            shape (batch_size, num_instances, 5).
        """
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
        angle_regular = norm_angle(decoded_angle, self.angle_version)
        decoded_rbbox = torch.cat([xys, whs, angle_regular], dim=-1)

        return decoded_rbbox

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 5),
              the last dimension 5 arrange as (cx, cy, w, h, angle).
        """

        if rescale:
            assert img_meta.get('scale_factor') is not None
            results.bboxes[..., :4] /= results.bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
        return results

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            angle_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            angle_preds (Sequence[Tensor]): Box angles for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors, H, W).
            objectnesses (Sequence[Tensor]): Score factor for all scale level,
                each is a 4D-tensor, has shape (batch_size, 1, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_decoded_angle = self.angle_coder.decode(flatten_angle_preds).unsqueeze(-1)
        flatten_rbboxes = self._bbox_decode_cxcywha(flatten_priors, flatten_bbox_preds,
                                                    flatten_decoded_angle)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_targets_single,
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_cls_preds.detach(), flatten_rbboxes.detach(),
             flatten_objectness.detach(), batch_gt_instances,
             batch_img_metas, batch_gt_instances_ignore)

        # The experimental results show that 'reduce_mean' can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        if num_pos > 0:
            loss_cls = self.loss_cls(
                flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
                cls_targets) / num_total_samples

            if self.separate_angle:
                flatten_hbboxes = bbox_cxcywh_to_xyxy(flatten_rbboxes[..., :4])
                hbbox_xyxy_targets = bbox_cxcywh_to_xyxy(bbox_targets[..., :4])
                angle_targets = bbox_targets[..., 4:5]
                angle_targets = self.angle_coder.encode(angle_targets)
                loss_bbox = self.loss_bbox(
                    flatten_hbboxes.view(-1, 4)[pos_masks],
                    hbbox_xyxy_targets) / num_total_samples
                loss_angle = self.loss_angle(
                    flatten_angle_preds.view(-1, self.angle_coder.encode_size)[pos_masks],
                    angle_targets) / num_total_samples
            else:
                loss_bbox = self.loss_bbox(
                    flatten_rbboxes.view(-1, 5)[pos_masks],
                    bbox_targets) / num_total_samples

        else:
            # Avoid cls and reg branch not participating in the gradient
            # propagation when there is no ground-truth in the images.
            # For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/issues/7298
            loss_cls = flatten_cls_preds.sum() * 0
            loss_bbox = flatten_rbboxes.sum() * 0
            if self.separate_angle:
                loss_angle = flatten_angle_preds.sum() * 0

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)
        if self.separate_angle:
            loss_dict.update(loss_angle = loss_angle)

        if self.use_l1:
            if num_pos > 0:
                if self.with_angle_l1:
                    flatten_rbbox_preds = torch.cat([
                        flatten_bbox_preds,
                        flatten_decoded_angle / self.angle_norm_factor
                    ], dim=-1)
                    loss_l1 = self.loss_l1(
                        flatten_rbbox_preds.view(-1, 5)[pos_masks],
                        l1_targets) / num_total_samples
                else:
                    loss_l1 = self.loss_l1(
                        flatten_bbox_preds.view(-1, 4)[pos_masks],
                        l1_targets) / num_total_samples
            else:
                # Avoid cls and reg branch not participating in the gradient
                # propagation when there is no ground-truth in the images.
                # For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/issues/7298
                loss_l1 = flatten_bbox_preds.sum() * 0
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    @torch.no_grad()
    def _get_targets_single(
            self,
            priors: Tensor,
            cls_preds: Tensor,
            decoded_bboxes: Tensor,
            objectness: Tensor,
            gt_instances: InstanceData,
            img_meta: dict,
            gt_instances_ignore: Optional[InstanceData] = None) -> tuple:
        """Compute classification, regression, and objectness targets for
        priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 5] in [cx, cy, w, h, a] format.
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            tuple:
                foreground_mask (list[Tensor]): Binary mask of foreground
                targets.
                cls_target (list[Tensor]): Classification targets of an image.
                obj_target (list[Tensor]): Objectness targets of an image.
                bbox_target (list[Tensor]): BBox targets of an image.
                l1_target (int): BBox L1 targets of an image.
                num_pos_per_img (int): Number of positive samples in an image.
        """

        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 5))
            if self.with_angle_l1:
                l1_target = cls_preds.new_zeros((0, 5))
            else:
                l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        return super()._get_targets_single(priors, cls_preds, decoded_bboxes,
                                           objectness, gt_instances, img_meta,
                                           gt_instances_ignore)

    def _get_l1_target(self,
                       l1_target: Tensor,
                       gt_bboxes: Tensor,
                       priors: Tensor,
                       eps: float = 1e-8) -> Tensor:
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = gt_bboxes[..., :4]
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        if self.with_angle_l1:
            angle_target = gt_bboxes[..., 4:5] / self.angle_norm_factor
            return torch.cat([l1_target, angle_target], dim=-1)
        else:
            return l1_target
