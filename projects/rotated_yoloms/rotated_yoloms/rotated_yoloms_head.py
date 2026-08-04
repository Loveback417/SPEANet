from torch import nn
from mmcv.cnn import ConvModule
from mmrotate.registry import MODELS
from mmrotate.models.dense_heads.rotated_rtmdet_head import RotatedRTMDetSepBNHead

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class DWConv(nn.Module):
    """Depthwise Convolution Module."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 conv_cfg=None,
                 norm_cfg=None,
                 groups=4,
                 act_cfg=None):
        super().__init__()
        if kernel_size == 3:
            groups = 1
        else:
            groups = groups if groups != -1 else out_channels

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=autopad(kernel_size),
            groups=groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x):
        return self.conv(x)

@MODELS.register_module()
class RotatedYOLOMSSepBNHead(RotatedRTMDetSepBNHead):

    def __init__(self,
                 reg_kernel_sizes=[[3], [5], [7]],
                 cls_kernel_sizes=[[3], [5], [7]],
                 groups=[4, 4, 4],
                 **kwargs):
        self.reg_kernel_sizes = reg_kernel_sizes
        self.cls_kernel_sizes = cls_kernel_sizes
        self.groups = groups
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_ang = nn.ModuleList()
        if self.with_objectness:
            self.rtm_obj = nn.ModuleList()
        for n in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    DWConv(
                        chn,
                        self.feat_channels,
                        self.cls_kernel_sizes[n][i],
                        groups=self.groups[n],
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_convs.append(
                    DWConv(
                        chn,
                        self.feat_channels,
                        self.reg_kernel_sizes[n][i],
                        groups=self.groups[n],
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_ang.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.angle_coder.encode_size,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            if self.with_objectness:
                self.rtm_obj.append(
                    nn.Conv2d(
                        self.feat_channels,
                        1,
                        self.pred_kernel_size,
                        padding=self.pred_kernel_size // 2))

        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
