import math
from typing import Sequence, Union, List

import torch
from torch.nn import functional as F
import torch.nn as nn

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmrotate.models.necks.base_yolo_neck import BaseYOLONeck
from mmrotate.registry import MODELS

from .msblock import MSBlock

@MODELS.register_module()
class YOLOMSPAFPN(BaseYOLONeck):
    """Path Aggregation Network with MS-Blocks.

    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        mid_channels (Sequence[int]): Number of middle channels per scale. Defaults to [].

        deepen_factor (float): Depth multiplier, scales the number of blocks in the MS-Block layer. Defaults to 1.0.
        widen_factor (float): Width multiplier, scales the number of channels in each layer. Defaults to 1.0.
        freeze_all (bool): Whether to freeze the model. Defaults to False.
        use_depthwise (bool): Whether to use depthwise separable convolution in blocks. Defaults to False.
        upsample_cfg (dict): Configuration dictionary for the interpolation layer. Defaults to `dict(scale_factor=2, mode='nearest')`.
        kernel_sizes (Sequence[Union[int, Sequence[int]]]): Kernel sizes for the MS-Block layers. Defaults to [1, 3, 3].

        msblock_layer_type (str): The type of layer used in MS-Block. Defaults to "MSBlockBottleNeckLayer".
        msblock_down_ratio (float): Channel down ratio for the downsample convolution layer in MS-Block. Defaults to 1.
        msblock_mid_expand_ratio (float): Channel expansion ratio for each branch in MS-Block. Defaults to 2.
        msblock_layers_num (int): Number of layers in the MS-Block. Defaults to 3.
        msblock_start_branch_id (int): The index of the branch where operations start in MS-Block. Defaults to 1.
        msblock_channel_split_ratios (list[float]): Ratios for splitting channels among branches in MS-Block. Defaults to [1, 1, 1].
        msblock_attention_cfg (:obj:`ConfigDict` or dict, optional): Configuration dictionary for attention in MS-Block. Defaults to None.
        msblock_act_cfg (:obj:`ConfigDict` or dict): Configuration dictionary for activation layers in MS-Block. Defaults to `dict(type='SiLU', inplace=True)`.

        alpha (float): Weight for combining the input feature map with the output feature map. Defaults to 1.
        conv_group (str): Grouping method for convolutions. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Configuration dictionary for convolution layers. Defaults to None.
        drop_layer (:obj:`ConfigDict` or dict, optional): Configuration dictionary for dropout layers. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Configuration dictionary for normalization layers. Defaults to `dict(type='BN')`.
        act_cfg (:obj:`ConfigDict` or dict): Configuration dictionary for activation layers. Defaults to `dict(type='SiLU', inplace=True)`.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or list[:obj:`ConfigDict`]): Initialization configuration dictionary. Defaults to:
            `dict(type='Kaiming', layer='Conv2d', a=math.sqrt(5), distribution='uniform', mode='fan_in', nonlinearity='leaky_relu')`.
    """

    def __init__(
            self,
            in_channels: Sequence[int],
            out_channels: int,
            mid_channels: Sequence[int] = [],

            deepen_factor: float = 1.0,
            widen_factor: float = 1.0,
            freeze_all: bool = False,
            use_depthwise: bool = False,
            upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
            kernel_sizes: Sequence[Union[int, Sequence[int]]] = [1, 3, 3],

            msblock_layer_type="MSBlockBottleNeckLayer",
            msblock_down_ratio: float = 1,
            msblock_mid_expand_ratio: float = 2,
            msblock_layers_num: int = 3,
            msblock_start_branch_id: int = 1,
            msblock_channel_split_ratios=[1, 1, 1],
            msblock_attention_cfg: OptConfigType = None,
            msblock_act_cfg: ConfigType = dict(type='SiLU', inplace=True),

            alpha=1,
            conv_group="auto",
            conv_cfg: bool = None,
            drop_layer=None,
            norm_cfg: ConfigType = dict(type='BN'),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = dict(
                type='Kaiming',
                layer='Conv2d',
                a=math.sqrt(5),
                distribution='uniform',
                mode='fan_in',
                nonlinearity='leaky_relu')
    ) -> None:

        self.conv = DepthwiseSeparableConvModule \
            if use_depthwise else ConvModule
        self.upsample_cfg = upsample_cfg

        if isinstance(kernel_sizes, list):
            kernel_sizes = {"bottom_up": [kernel_sizes for _ in range(2)],
                            "top_down": [kernel_sizes for _ in range(2)]}
        self.kernel_sizes = kernel_sizes

        self.layer_config = dict(
            down_ratio=msblock_down_ratio,
            mid_expand_ratio=msblock_mid_expand_ratio,
            attention_cfg=msblock_attention_cfg,
            channel_split_ratios=msblock_channel_split_ratios,
            start_branch_id=msblock_start_branch_id,
            layers_num=round(msblock_layers_num * deepen_factor),
            layer_type=msblock_layer_type,
            drop_layer=drop_layer,
            conv_group=conv_group,
            norm_cfg=norm_cfg,
            act_cfg=msblock_act_cfg,
            conv_cfg=conv_cfg
        )

        self.conv_cfg = conv_cfg
        self.mid_channels = [int(mid_channel * widen_factor) for mid_channel in mid_channels]

        self.layer = MSBlock

        self.alpha = alpha

        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=int(out_channels * widen_factor),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if self.in_channels[idx] != self.mid_channels[idx]:
            proj = self.conv(self.in_channels[idx],
                             self.mid_channels[idx],
                             1,
                             norm_cfg=self.norm_cfg,
                             act_cfg=self.act_cfg)
        else:
            proj = nn.Identity()
        if idx == len(self.in_channels) - 1:
            layer = self.conv(
                self.mid_channels[idx],
                self.mid_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return nn.Sequential(proj, layer)

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(**self.upsample_cfg)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """

        self.layer_config["kernel_sizes"] = self.kernel_sizes["top_down"][idx - 1]
        if idx == 1:
            return self.layer(self.mid_channels[idx - 1] * 2,
                              self.mid_channels[idx - 1],
                              **self.layer_config)
        else:
            return nn.Sequential(
                self.layer(self.mid_channels[idx - 1] * 2,
                           self.mid_channels[idx - 1],
                           **self.layer_config),
                self.conv(self.mid_channels[idx - 1],
                          self.mid_channels[idx - 2],
                          kernel_size=1,
                          norm_cfg=self.norm_cfg,
                          act_cfg=self.act_cfg))

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return self.conv(self.mid_channels[idx],
                         self.mid_channels[idx],
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         norm_cfg=self.norm_cfg,
                         act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        self.layer_config["kernel_sizes"] = self.kernel_sizes["bottom_up"][idx]
        return self.layer(self.mid_channels[idx] * 2,
                          self.mid_channels[idx + 1],
                          **self.layer_config)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return self.conv(
            self.mid_channels[idx],
            self.out_channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        if len(inputs) == len(self.in_channels) + 1:
            input_1st = inputs[0]
            inputs = inputs[1:]
        else:
            input_1st = None
            assert len(inputs) == len(self.in_channels)

        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            result = self.out_layers[idx](outs[idx])
            if input_1st is not None:
                result = self.alpha * result + (1 - self.alpha) * F.adaptive_avg_pool2d(input_1st,
                                                                                        result.shape[-2:]).mean(1,
                                                                                                                keepdim=True)
            results.append(result)

        return tuple(results)
