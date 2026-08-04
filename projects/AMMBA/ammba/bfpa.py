import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmengine.model import  BaseModule, constant_init, normal_init
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import NonLocal2d
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmdet.utils import OptConfigType


class DyReLU(nn.Module):
    """Dynamic ReLU."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.expand_ratio = expand_ratio
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // expand_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // expand_ratio,
                      out_channels * self.expand_ratio),
            nn.Hardsigmoid(inplace=True))

    def forward(self, x) -> Tensor:
        x_out = x
        b, c, h, w = x.size()
        x = self.avg_pool(x).view(b, c)
        x = self.fc(x).view(b, -1, 1, 1)

        a1, b1, a2, b2 = torch.split(x, self.out_channels, dim=1)
        a1 = (a1 - 0.5) * 2 + 1.0
        a2 = (a2 - 0.5) * 2
        b1 = b1 - 0.5
        b2 = b2 - 0.5
        out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        return out

class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.

    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset.contiguous(), mask)
        if self.with_norm:
            x = self.norm(x)
        return x

class BFPA(BaseModule):
    """BFPA (Balanced Feature Pyramids with Attention)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    the paper `Libra R-CNN: Towards Balanced Learning for Object Detection
    <https://arxiv.org/abs/1904.02701>`_ for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_levels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 zero_init_offset=True,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super(BFPA, self).__init__(init_cfg)
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = DyReLU(in_channels, out_channels)
        self._init_weights()

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                self.in_channels,
                reduction=1,  # Channel reduction ratio. default:2
                use_scale=False, # Whether to scale pairwise_weight by
                                 # `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
                                 # Default: True.
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]

        for i in range(self.num_levels):

            offset_and_mask = self.spatial_conv_offset(inputs[i])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(inputs[i], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if i > 0:
                low_feat = self.spatial_conv_low(inputs[i - 1], offset, mask)
                sum_feat = sum_feat + \
                    low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if i < len(inputs) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.spatial_conv_high(inputs[i + 1], offset, mask),
                    size=inputs[i].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat = sum_feat + high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    sum_feat / summed_levels, output_size=gather_size)
            else:
                gathered = F.interpolate(
                    sum_feat / summed_levels, size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='bilinear', align_corners=True)
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])

        return tuple(outs)