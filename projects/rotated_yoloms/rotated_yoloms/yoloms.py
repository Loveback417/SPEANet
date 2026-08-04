import math
from typing import List, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmrotate.models.backbones.yolo_base_backbone import YoloBaseBackbone
from mmrotate.registry import MODELS
from .msblock import MSBlock


@MODELS.register_module()
class YOLOMS(YoloBaseBackbone):
    """YOLOMS (Multi-Scale YOLO) Backbone Network.

    This class implements the backbone network for YOLOMS, a multi-scale object detection
    architecture based on YOLO. The backbone processes input images through multiple stages
    with increasing receptive fields, using MSBlock (Multi-Scale Block) as its core building component.

    Args:
        arch (str): Architecture name, selecting predefined block configurations.
        input_channels (int): Number of input channels for the backbone.
        deepen_factor (float): Depth multiplier, scales the number of blocks in each stage.
        widen_factor (float): Width multiplier, scales the number of channels in each layer.
        out_indices (Sequence[int]): Indices of output stages to return feature maps from.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
        plugins (Union[dict, List[dict]], optional): Additional plugins for the backbone.

        conv_cfg (dict, optional): Config for convolution layers.
        norm_cfg (dict): Config for normalization layers.
        norm_eval (bool): Whether to set norm layers to eval mode, freezing running stats.
        act_cfg (dict): Config for activation layers.

        msblock_layer_type (str): Type of MSBlock layer to use.
        msblock_down_ratio (float): Channel reduction ratio for downsample conv layer before MSBlock.
        msblock_mid_expand_ratio (float): Channel expansion ratio for intermediate branches in MSBlock.
        msblock_layers_num (int): Number of layers in MSBlock.
        msblock_conv_group (str): Grouping strategy for convolutions in MSBlock.
        msblock_conv_cfg (dict, optional): Config for MSBlock convolution layers.
        msblock_norm_cfg (dict): Config for MSBlock normalization layers.
        msblock_act_cfg (dict): Config for MSBlock activation layers.
        msblock_start_branch_id (int): Starting branch ID in MSBlock.
        msblock_attention_cfg (dict, optional): Config for branch attention in MSBlock.
        msblock_out_attention_cfg (dict, optional): Config for output attention in MSBlock.

        spp_config (dict): Config for Spatial Pyramid Pooling block.
        drop_layer (dict, optional): Config for dropout layers.
        init_cfg (dict): Initialization config.

    Returns:
        tuple[Tensor]: Multi-level feature maps from specified output stages.
    """

    arch_settings = {
        'C3-K3579': [
            [MSBlock, 80, 160, [1, (3, 3), (3, 3)], [1, 1, 1], False, None],
            [MSBlock, 160, 320, [1, (5, 5), (5, 5)], [1, 1, 1], False, None],
            [MSBlock, 320, 640, [1, (7, 7), (7, 7)], [1, 1, 1], False, None],
            [MSBlock, 640, 1280, [1, (9, 9), (9, 9)], [1, 1, 1], True, None],
        ],
        'C3-K3579-A234': [
            [MSBlock, 80, 160, [1, (3, 3), (3, 3)], [1, 1, 1], False, None],
            [MSBlock, 160, 320, [1, (5, 5), (5, 5)], [1, 1, 1], False, True],
            [MSBlock, 320, 640, [1, (7, 7), (7, 7)], [1, 1, 1], False, True],
            [MSBlock, 640, 1280, [1, (9, 9), (9, 9)], [1, 1, 1], True, True],
        ],
    }

    def __init__(
            self,
            arch: str = 'C3-K3579',
            input_channels: int = 3,
            deepen_factor: float = 1.0,
            widen_factor: float = 1.0,
            out_indices: Sequence[int] = (2, 3, 4),
            frozen_stages: int = -1,
            plugins: Union[dict, List[dict]] = None,

            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN'),
            norm_eval: bool = False,
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),

            msblock_layer_type: str = "MSBlockBottleNeckLayer",
            msblock_down_ratio: float = 1.0,
            msblock_mid_expand_ratio: float = 2.0,
            msblock_layers_num: int = 3,
            msblock_conv_group: str = "auto",
            msblock_conv_cfg: OptConfigType = None,
            msblock_norm_cfg: ConfigType = dict(type='BN'),
            msblock_act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            msblock_start_branch_id: int = 1,
            msblock_attention_cfg: OptConfigType = None,
            msblock_out_attention_cfg: OptConfigType = None,

            spp_config: ConfigType = dict(type="SPPFBottleneck", kernel_sizes=5),
            drop_layer: dict = None,
            init_cfg: OptMultiConfig = dict(
                type='Kaiming',
                layer='Conv2d',
                a=math.sqrt(5),
                distribution='uniform',
                mode='fan_in',
                nonlinearity='leaky_relu'
            )
    ) -> None:
        arch_setting = self.arch_settings[arch]
        self.conv = ConvModule
        self.conv_cfg = conv_cfg

        self.msblock_down_ratio = msblock_down_ratio
        self.msblock_mid_expand_ratio = msblock_mid_expand_ratio
        self.msblock_layers_num = msblock_layers_num
        self.msblock_conv_group = msblock_conv_group
        self.msblock_conv_cfg = msblock_conv_cfg
        self.msblock_norm_cfg = msblock_norm_cfg
        self.msblock_act_cfg = msblock_act_cfg
        self.msblock_layer_type = msblock_layer_type
        self.msblock_start_branch_id = msblock_start_branch_id
        self.msblock_attention_cfg = msblock_attention_cfg
        self.msblock_attention_indexs = [
            True if setting[-1] is not None else False for setting in arch_setting
        ]
        self.msblock_out_attention_cfg = msblock_out_attention_cfg

        self.spp_config = spp_config
        self.drop_layer = drop_layer

        super().__init__(
            arch_setting,
            deepen_factor,
            widen_factor,
            input_channels,
            out_indices,
            frozen_stages=frozen_stages,
            plugins=plugins,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg
        )

        self.query = None
        if sum(self.msblock_attention_indexs) != 0:
            assert msblock_attention_cfg is not None, \
                "msblock_attention_cfg should not be None when using branch attention."
            query_length = msblock_attention_cfg.get("length", 3)
            query_size = msblock_attention_cfg.get("size", 4)
            self.query = nn.Parameter(
                torch.randn((query_length, query_size * query_size)),
                requires_grad=True
            ).to(next(self.parameters()).device)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        stem = nn.Sequential(
            ConvModule(
                3,
                int(self.arch_setting[0][1] * self.widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                int(self.arch_setting[0][1] * self.widen_factor // 2),
                int(self.arch_setting[0][1] * self.widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                int(self.arch_setting[0][1] * self.widen_factor // 2),
                int(self.arch_setting[0][1] * self.widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        )
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        layer, in_channels, out_channels, kernel_sizes, channel_split_ratios, use_spp, use_attention_cfg = setting
        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)
        downsample_channel = int(in_channels * self.msblock_down_ratio)

        stage = []
        conv_layer = self.conv(
            in_channels,
            downsample_channel,
            3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        stage.append(conv_layer)

        if use_spp:
            self.spp_config["in_channels"] = downsample_channel
            self.spp_config["out_channels"] = downsample_channel
            spp = MODELS.build(self.spp_config)
            stage.append(spp)

        attention_cfg = self.msblock_attention_cfg if use_attention_cfg else None

        csp_layer = layer(
            downsample_channel,
            out_channels,
            down_ratio=1,
            mid_expand_ratio=self.msblock_mid_expand_ratio,
            start_branch_id=self.msblock_start_branch_id,
            attention_cfg=attention_cfg,
            out_attention_cfg=self.msblock_out_attention_cfg,
            kernel_sizes=kernel_sizes,
            channel_split_ratios=channel_split_ratios,
            layers_num=self.msblock_layers_num * self.deepen_factor,
            layer_type=self.msblock_layer_type,
            conv_group=self.msblock_conv_group,
            conv_cfg=self.msblock_conv_cfg,
            drop_layer=self.drop_layer,
            act_cfg=self.msblock_act_cfg,
            norm_cfg=self.msblock_norm_cfg
        )
        stage.append(csp_layer)
        return stage

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            layers = getattr(self, layer_name)
            for layer in layers:
                if isinstance(layer, MSBlock):
                    x = layer(x, self.query)
                else:
                    x = layer(x)

            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
