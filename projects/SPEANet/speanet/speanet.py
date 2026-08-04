# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import os
import copy
from mmcv.cnn import build_norm_layer
from math import log
import numpy
import matplotlib.pyplot as plt
from mmengine.model import BaseModule
# from ai4rs.registry import MODELS
from mmrotate.registry import MODELS
find_unused_parameters=False


# Fixed-kernel utilities


def create_gaussian_kernel(size: int, sigma: float) -> Tensor:
    """Create a normalized 2D Gaussian kernel.

    Args:
        size: Odd kernel size.
        sigma: Standard deviation of the Gaussian.

    Returns:
        A tensor with shape ``[1, 1, size, size]``.
    """
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel.unsqueeze(0).unsqueeze(0)

def create_log_kernel(size: int, sigma: float) -> Tensor:
    """Create a zero-mean Laplacian-of-Gaussian kernel.

    Args:
        size: Odd kernel size.
        sigma: Standard deviation of the Gaussian envelope.

    Returns:
        A tensor with shape ``[1, 1, size, size]``.
    """
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    # Analytical LoG response followed by zero-mean correction.
    log_kernel = (xx**2 + yy**2 - 2 * sigma**2) / (sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    log_kernel = log_kernel - log_kernel.mean()
    return log_kernel.unsqueeze(0).unsqueeze(0)

def create_directional_sobel_kernels() -> Tensor:
    """Create Sobel kernels for 0, 45, 90, and 135 degrees.

    Returns:
        A tensor with shape ``[4, 1, 3, 3]``.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    sobel_45 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32)
    sobel_135 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
    
    kernels = [sobel_x, sobel_45, sobel_y, sobel_135]
    return torch.stack(kernels).unsqueeze(1)  # [4, 1, 3, 3]

def create_laplacian_kernel() -> Tensor:
    """Create an eight-neighborhood Laplacian kernel.

    Returns:
        A tensor with shape ``[1, 1, 3, 3]``.
    """
    return torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


class FixedKernelConv2d(nn.Module):
    """Apply a fixed operator independently to grouped feature channels."""
    def __init__(self, channels: int, kernel: Tensor, stride: int = 1, groups: int = 1):
        super().__init__()
        kernel_out_c, kernel_in_c, kh, kw = kernel.shape
        assert kernel_in_c == 1
        
        padding = (kh - 1) // 2
        self.conv = nn.Conv2d(channels, channels * kernel_out_c, (kh, kw), 
                              stride=stride, padding=padding, groups=groups, bias=False)
        # Repeat the operator bank for each input channel and freeze its weights.
        repeated_kernel = kernel.repeat(channels, 1, 1, 1)
        self.conv.weight.data.copy_(repeated_kernel)
        self.conv.weight.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def show_feature(out: Tensor):
    """Visualize feature channels for debugging."""
    feature_map = out.detach().cpu().squeeze().numpy()
    if feature_map.ndim == 3:
        feature_map = numpy.transpose(feature_map, [1, 2, 0])
    
    num_channels = feature_map.shape[2]
    cols = 6
    rows = math.ceil(num_channels / cols)
    
    plt.figure(figsize=(cols * 2, rows * 2))
    for c in range(num_channels):
        ax = plt.subplot(rows, cols, c + 1)
        plt.axis('off')
        plt.imshow(feature_map[:, :, c], cmap='viridis')
    plt.tight_layout()
    plt.show()


class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, self.outdim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.gaussian = GaussianPrior(self.outdim, 5, 0.5, norm_layer, act_layer)
        self.norm_g = build_norm_layer(norm_layer, self.outdim)[1]
        self.conv_path = nn.Sequential(
            nn.Conv2d(self.outdim, self.outdim, kernel_size=3, stride=2, padding=1, groups=self.outdim),
            act_layer(),
            build_norm_layer(norm_layer, self.outdim)[1])
        self.pool_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            build_norm_layer(norm_layer, self.outdim)[1])
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]

        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        pool_path = self.pool_path(x)  # m = [B, 2C, H/2, W/2]
        conv_path = self.conv_path(x)  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv_path, pool_path], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]
        return x


# Multi-Order Directional Prior with Frequency-Aware Scaling and Encoding
class MDPFASE(nn.Module):
    """Encode fixed first- and second-order directional responses."""
    def __init__(self, dim, mlp_ratio, norm_layer, act_layer=nn.ReLU):
        super().__init__()
        # Fixed first- and second-order operator branches.
        sobel_kernels = create_directional_sobel_kernels()
        self.sobel_conv = FixedKernelConv2d(dim, sobel_kernels, groups=dim)
        self.direction_enhance = nn.Sequential(
            build_norm_layer(norm_layer, dim * 4)[1], 
            act_layer(),
            nn.Conv2d(dim * 4, dim * 2, 1), 
            build_norm_layer(norm_layer, dim * 2)[1], 
            act_layer(),
            nn.Conv2d(dim * 2, dim, 1), build_norm_layer(norm_layer, dim)[1]
        )
        self.direction_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1), act_layer(),
            nn.Conv2d(dim, dim, 1), nn.Sigmoid()
        )
        laplacian_kernel = create_laplacian_kernel()
        self.laplacian_conv = FixedKernelConv2d(dim, laplacian_kernel, groups=dim)
        self.norm_laplacian = build_norm_layer(norm_layer, dim)[1]

        # Keep this attribute name to preserve existing checkpoint keys.
        self.FASE_Refiner = FASE(dim=dim * 2, mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer)
        
        # Restore the original channel dimension after FASE encoding.
        self.final_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            build_norm_layer(norm_layer, dim)[1],
            act_layer()
        )

    def forward(self, x):
        # First-order directional responses.
        sobel_response = self.sobel_conv(x)
        sobel_response = self.direction_enhance(sobel_response)
        sobel_response = sobel_response * self.direction_weights(sobel_response)

        # Isotropic second-order response.
        laplacian_response = self.norm_laplacian(self.laplacian_conv(x))
        
        combined_features = torch.cat([sobel_response, laplacian_response], dim=1)
        
        # Globally condition and project the combined derivative responses.
        refined_features = self.FASE_Refiner(combined_features)
        output_prior = self.final_proj(refined_features)
        
        return output_prior
    

class GaussianPrior(nn.Module):
    """Apply a fixed Gaussian operator followed by normalization and activation."""

    def __init__(self, dim, size, sigma, norm_layer, act_layer):
        super().__init__()
        gaussian_kernel = create_gaussian_kernel(size, sigma)
        self.gaussian = FixedKernelConv2d(dim, gaussian_kernel, groups=dim)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        return gaussian



class WASM(nn.Module):
    """Mix Gaussian-guided approximation and Laplacian-guided detail branches.

    This module follows an analysis-interaction-synthesis pattern inspired by
    wavelets, but it does not perform an explicit discrete wavelet transform.
    """
    def __init__(self, dim, mlp_ratio, drop_path, norm_layer, act_layer):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Learnable approximation and detail transforms.
        self.approx_path = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.Conv2d(dim, dim, 1))
        self.detail_path = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.Conv2d(dim, dim, 1))
        
        # Fixed low-pass and high-pass operator priors.
        self.gaussian_prior = FixedKernelConv2d(dim, create_gaussian_kernel(5, 1.0), groups=dim)
        self.laplacian_prior = FixedKernelConv2d(dim, create_laplacian_kernel(), groups=dim)

        self.norm_approx = build_norm_layer(norm_layer, dim)[1]
        self.norm_detail = build_norm_layer(norm_layer, dim)[1]

        # Cross-branch modulation and synthesis.
        self.pa_proj = nn.Conv2d(dim, dim, 1)
        self.da_proj = nn.Conv2d(dim, dim, 1)
        self.act = act_layer()
        self.fusion = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, bias=False),
                                    build_norm_layer(norm_layer, dim)[1]
                                    )
        
    def forward(self, x):
        # Analysis: augment each learnable branch with its fixed prior.
        approx = self.act(self.norm_approx(self.approx_path(x) + self.gaussian_prior(x)))
        detail = self.act(self.norm_detail(self.detail_path(x) + self.laplacian_prior(x)))
        
        # Bidirectional approximation-detail interaction.
        approx_refined = approx * torch.sigmoid(self.da_proj(detail))
        detail_refined = detail * torch.sigmoid(self.pa_proj(approx))
        
        # Synthesis with an identity connection.
        recomposed = torch.cat([approx_refined, detail_refined], dim=1)
        out = x + self.fusion(recomposed)
        return out


class ContextGuidedFrequencyFiltering(nn.Module):
    """Modulate a prior-enhanced response using learned spatial context."""

    def __init__(self, channel, norm_layer, act_layer):
        super().__init__()
        # A dilated context encoder predicts a single-channel spatial gate.
        self.context_encoder = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=3, padding=2, dilation=2, bias=False),
            build_norm_layer(norm_layer, channel // 2)[1],
            act_layer(),
            nn.Conv2d(channel // 2, channel, kernel_size=1, bias=False)
        )
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Refine the gated prior and learned feature after residual fusion.
        self.enhancer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer(),
            nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        )
        self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, c, att):
        # c and att are the learned feature and prior-enhanced response.
        identity = c
        context = self.context_encoder(c)
        gate = self.gate_generator(context)  # [B, 1, H, W]
        gated_att = att * gate
        fused_feature = c + gated_att
        refined_feature = self.enhancer(fused_feature)
        return self.norm(identity + refined_feature)


class FASE(nn.Module):
    """Scale a pointwise feature transform using globally aggregated context."""

    def __init__(self, dim, mlp_ratio, norm_layer, act_layer):
        super().__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # Summarize locally filtered responses into a global descriptor.
        self.freq_aggregator = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Attribute name retained for checkpoint compatibility. This branch
        # generates global-context-conditioned scaling, not an entropy estimate.
        self.entropy_generator = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            act_layer(),
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4),
            act_layer(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Pointwise feature transformation.
        self.proj = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )
        self.norm = build_norm_layer(norm_layer, dim)[1]

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        
        x_proj = self.proj(x)

        global_context = self.freq_aggregator(x).view(B, C, 1, 1)
        global_context_expanded = global_context.expand(-1, -1, H, W)
        scaling_map = self.entropy_generator(global_context_expanded)
        encoded_x = x_proj * scaling_map
        
        out = identity + self.norm(encoded_x)
        return out
    

class HPRB(nn.Module):
    """Apply stage-specific prior extraction, CGFF, and FASE encoding.

    Stage 0 uses MDP-FASE; Stages 1-3 use WASM. The full HPRB
    transformation is wrapped by stochastic depth and an identity path.
    """
    def __init__(self, dim, stage, mlp_ratio, drop_path, act_layer, norm_layer):
        super().__init__()
        if stage == 0:
            self.StructureExtractor = MDPFASE(dim, mlp_ratio, norm_layer, act_layer)
        else:
            self.StructureExtractor = WASM(dim, mlp_ratio, drop_path, norm_layer, act_layer)
        self.FeatureFusion = ContextGuidedFrequencyFiltering(dim, norm_layer, act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.FASE = FASE(dim, mlp_ratio, norm_layer, act_layer)

    def forward(self, x: Tensor) -> Tensor:
        att = self.StructureExtractor(x)
        x_fused = self.FeatureFusion(x, att)
        x_out = self.FASE(x_fused)
        # Apply stochastic depth to the complete HPRB residual.
        return x + self.drop_path(x_out - x)



class BasicStage(nn.Module):
    def __init__(self, dim, stage, depth, mlp_ratio, drop_path, norm_layer, act_layer):
        super().__init__()

        blocks_list = [
            HPRB(dim=dim, stage=stage,mlp_ratio=mlp_ratio,
                       drop_path=drop_path[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class LoGPriorBranch(nn.Module):
    """Form a residual LoG-enhanced response with a fixed kernel."""

    def __init__(self, out_c, kernel_size, sigma, norm_layer, act_layer):
        super().__init__()
        log_kernel = create_log_kernel(kernel_size, sigma)
        self.LoG = FixedKernelConv2d(out_c, log_kernel, groups=out_c)
        self.act = act_layer()
        self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = build_norm_layer(norm_layer, out_c)[1]
    
    def forward(self, x):
        log_response = self.LoG(x)
        log_edge = self.act(self.norm1(log_response))
        return self.norm2(x + log_edge)
    

class ContextGuidedLoGStem(nn.Module):
    """Preserve LoG contour responses before early downsampling."""

    def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
        super().__init__()
        out_c14 = int(stem_dim / 4)
        out_c12 = int(stem_dim / 2)
        self.conv_init = nn.Conv2d(in_chans, out_c14, 7, padding=3)
        self.FeatureFusion = ContextGuidedFrequencyFiltering(out_c14, norm_layer, act_layer)
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            build_norm_layer(norm_layer, out_c12)[1])
        self.LoG = LoGPriorBranch(out_c14, 7, 1.0, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, out_c12)[1]
        self.drfd = DRFD(out_c12, norm_layer, act_layer)

    def forward(self, x):
        x = self.conv_init(x)
        
        x = self.FeatureFusion(x, self.LoG(x))
        x = self.Conv_D(x)
        x = self.drfd(x)

        return x  # [B, C, H/4, W/4]
    

@MODELS.register_module()
class SPEANet(BaseModule):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 stem_dim=32,
                 depths=(1, 4, 4, 2),
                 norm_layer=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.num_features = int(stem_dim * 2 ** (self.num_stages - 1))

        if stem_dim == 96:
            act_layer = nn.ReLU

        self.Stem = ContextGuidedLoGStem(
            in_chans=in_chans,
            stem_dim=stem_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        
        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.stages = nn.ModuleList()
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(stem_dim * 2 ** i_stage),
                               stage=i_stage,
                               depth=depths[i_stage],
                               mlp_ratio=mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               norm_layer=norm_layer,
                               act_layer=act_layer
                               )
            self.stages.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                self.stages.append(
                    DRFD(dim=int(stem_dim * 2 ** i_stage), norm_layer=norm_layer, act_layer=act_layer)
                )

        self.fork_feat = fork_feat

        self.forward = self.forward_det
        # add a norm layer for each output
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                raise NotImplementedError
            else:
                layer = build_norm_layer(norm_layer, int(stem_dim * 2 ** i_emb))[1]
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.Stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        # return outs
        return tuple(outs)