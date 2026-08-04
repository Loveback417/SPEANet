from typing import Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import ConvModule

from mmdet.utils import OptConfigType
from mmrotate.registry import MODELS

from .msblock_layers import (
    MSBlock_1x1_kxk_Layer,
    MSBlock_kxk_1x1_Layer,
    MSBlockBottleNeckLayer,
    MSBlock_kxk_Layer,
)


class MSBlock(nn.Module):
    """MSBlock

    Args:
        in_channel (int): The number of input channels of this Module.
        out_channel (int): The number of output channels of this Module.
        kernel_sizes (Sequence[Union[int, Sequence[int]]]): A sequence of kernel sizes for each branch in the MSBlock.
        down_ratio (float): The channel reduction ratio for the downsample convolution layer in the MSBlock. Defaults to 1.0.
        mid_expand_ratio (float): The channel expansion ratio for each branch in the MSBlock. Defaults to 2.0.
        layers_num (int): The number of layers in each branch of the MSBlock. Defaults to 3.
        start_branch_id (int): The index of the branch where operations start. Defaults to 1.
        channel_split_ratios (list[float]): Ratios for splitting channels among branches. Defaults to [1/3, 1/3, 1/3].
        layer_type (str): The type of layer used for each branch. Defaults to "MSBlockBottleNeckLayer".
        drop_layer (:obj:`ConfigDict` or dict, optional): Configuration dictionary for the dropout layer. Defaults to None.
        attention_cfg (:obj:`ConfigDict` or dict, optional): Configuration dictionary for the attention module in the MSBlock. Defaults to None.
        out_attention_cfg (:obj:`ConfigDict` or dict, optional): Configuration dictionary for the output attention module. Defaults to None.
        conv_group (str): The grouping method for convolutions. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Configuration dictionary for convolution layers. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Configuration dictionary for normalization layers. Defaults to dict(type="BN").
        act_cfg (:obj:`ConfigDict` or dict): Configuration dictionary for activation layers. Defaults to dict(type="SiLU", inplace=True).
    """

    layer_dict = {
        "MSBlock_1x1_kxk_Layer": MSBlock_1x1_kxk_Layer,
        "MSBlock_kxk_1x1_Layer": MSBlock_kxk_1x1_Layer,
        "MSBlockBottleNeckLayer": MSBlockBottleNeckLayer,
        "MSBlock_kxk_Layer": MSBlock_kxk_Layer,
    }

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_sizes: Sequence[Union[int, Sequence[int]]],
        down_ratio: float = 1.0,
        mid_expand_ratio: float = 2.0,
        layers_num: int = 3,
        start_branch_id: int = 1,
        channel_split_ratios=[1 / 3, 1 / 3, 1 / 3],
        layer_type="MSBlockBottleNeckLayer",
        drop_layer: OptConfigType = None,
        attention_cfg: OptConfigType = None,
        out_attention_cfg: OptConfigType = None,
        conv_group="auto",
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = dict(type="BN"),
        act_cfg: OptConfigType = dict(type="SiLU", inplace=True),
    ) -> None:
        super().__init__()
        assert len(kernel_sizes) == len(channel_split_ratios)

        # Normalize channel split ratios
        self.channel_split_ratios = [
            ratio / sum(channel_split_ratios) for ratio in channel_split_ratios
        ]
        self.layers_num = layers_num
        self.start_branch_id = start_branch_id

        # Input feature processing
        self.in_channel = int((in_channel * len(kernel_sizes)) * down_ratio)
        self.in_conv = ConvModule(
            in_channel,
            self.in_channel,
            1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )

        # Middle feature processing
        self.mid_channels = [
            int(ratio * self.in_channel) for ratio in self.channel_split_ratios
        ]
        self.mid_expand_ratio = mid_expand_ratio
        groups = [
            int(mid_channel * self.mid_expand_ratio) for mid_channel in self.mid_channels
        ]

        layer = self.layer_dict[layer_type]
        self.mid_convs = []
        for kernel_size, group, mid_channel in zip(kernel_sizes, groups, self.mid_channels):
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [
                layer(
                    mid_channel,
                    group,
                    kernel_size=kernel_size,
                    conv_group=conv_group,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                )
                for _ in range(int(self.layers_num))
            ]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)

        # Indices for channel slicing
        mid_channels = [0] + self.mid_channels
        self.indices = [
            (sum(mid_channels[: i + 1]), sum(mid_channels[: i + 2]))
            for i in range(len(self.mid_convs))
        ]

        # Attention layers
        self.attention = None
        if attention_cfg is not None:
            attention_cfg["dim"] = self.in_channel
            self.attention = MODELS.build(attention_cfg)

        self.out_conv = ConvModule(
            self.in_channel,
            out_channel,
            1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )

        self.out_attention = None
        if out_attention_cfg is not None:
            out_attention_cfg["dim"] = out_channel
            self.out_attention = MODELS.build(out_attention_cfg)

        # Dropout layer
        self.drop_layer = nn.Identity()
        if drop_layer is not None:
            self.drop_layer = build_dropout(drop_layer)

    def forward(self, x: Tensor, query=None) -> Tensor:
        """Forward process

        Args:
            x (Tensor): The input tensor.
        """
        out = self.in_conv(x)
        channels = []

        n = len(self.mid_convs)
        for i in range(n):
            start, end = self.indices[i]
            channel = out[:, start:end, ...]
            if i >= self.start_branch_id:
                channel = channel + channels[i - 1]
            channel = self.mid_convs[i](channel)
            channels.append(channel)

        out = torch.cat(channels, dim=1)

        if self.attention is not None:
            out = self.attention(out, query) if query is not None else self.attention(out)

        out = self.drop_layer(out)
        out = self.out_conv(out)

        if self.out_attention is not None:
            out = self.out_attention(out)

        return out
