import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmrotate.registry import MODELS

@MODELS.register_module()
class CEFI(BaseModule):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.depth_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                    groups=channel)
        self.point_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
                                    groups=1)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        y1_avg = self.avg_pool(x1).view(b, c)
        y1_max = self.max_pool(x1).view(b, c)
        y1 = y1_avg + y1_max
        y1 = self.fc(y1).view(b, c, 1, 1)

        gsconv_x1 = self.depth_conv(x1)
        gsconv_out1 = self.point_conv(gsconv_x1)
        gsconv_out1 = gsconv_out1 + x1

        gsconv_x2 = self.depth_conv(x2)
        gsconv_out2 = self.point_conv(gsconv_x2)
        gsconv_out2 = gsconv_out2 + x2
        out11 = x1 * y1.expand_as(x1)
        out12 = gsconv_out2 * y1.expand_as(x1)
        out_last1 = out11 + out12 + gsconv_out1

        y2_avg = self.avg_pool(x2).view(b, c)
        y2_max = self.max_pool(x2).view(b, c)
        y2 = y2_avg + y2_max
        y2 = self.fc(y2).view(b, c, 1, 1)
        out21 = x2 * y2.expand_as(x2)
        out22 = gsconv_out1 * y2.expand_as(x2)
        out_last2 = out21 + out22 + gsconv_out2

        return out_last1, out_last2

@MODELS.register_module()
class SEFI(BaseModule):
    def __init__(self,in_channels):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # map尺寸不变，缩减通道
        avgout1 = torch.mean(x1, dim=1, keepdim=True)
        avgout2 = torch.mean(x2, dim=1, keepdim=True)
        maxout1, _ = torch.max(x1, dim=1, keepdim=True)
        maxout2, _ = torch.max(x2, dim=1, keepdim=True)
        out1 = torch.cat([avgout1, maxout2], dim=1)
        out2 = torch.cat([avgout2, maxout1], dim=1)
        out1 = self.sigmoid(self.conv2d(out1))
        out2 = self.sigmoid(self.conv2d(out2))
        out_1 = out1 * x1
        out_2 = out2 * x2

        return out_1, out_2