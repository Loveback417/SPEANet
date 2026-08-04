# Copyright (c) AI4RS. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import ModuleList, Sequential

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import Upsample
from mmrotate.registry import MODELS


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


class TransformerEncoder(nn.Module):
    def __init__(self,
                 in_dims,
                 embed_dims,
                 num_heads,
                 dropout,
                 norm_cfg):
        super(TransformerEncoder, self).__init__()
        self.attn = Attention(
            in_dims,
            embed_dims,
            num_heads,
            dropout)
        self.ff = FeedForward(
            in_dims,
            embed_dims,
            dropout
        )
        self.norm1 = build_norm_layer(norm_cfg, in_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, in_dims)[1]

    def forward(self, x):
        x_ = self.attn(self.norm1(x)) + x
        y = self.ff(self.norm2(x_)) + x_
        return y


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            in_dims,
            embed_dims,
            num_heads,
            drop_rate,
            norm_cfg,
            apply_softmax=True
    ):
        super(TransformerDecoder, self).__init__()
        self.attn = Cross_Attention(
            in_dims,
            num_heads,
            embed_dims,
            drop_rate,
            apply_softmax)
        self.ff = FeedForward(
            in_dims,
            in_dims * 2,
            drop_rate
        )
        self.norm1 = build_norm_layer(norm_cfg, in_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, in_dims)[1]

    def forward(self, x, ref):
        x_ = self.attn(self.norm1(x), self.norm1(ref)) + x
        y = self.ff(self.norm2(x_)) + x_
        return y


@MODELS.register_module()
class BITHead(BaseDecodeHead):
    """BIT Head

    This head is the improved implementation of'Remote Sensing Image
    Change Detection With Transformers<https://github.com/justchenhao/BIT_CD>'

    Args:
        in_channels (int): Number of input feature channels (from backbone). Default:  512
        channels (int): Number of output channels of pre_process. Default:  32.
        embed_dims (int): Number of expanded channels of Attention block. Default:  64.
        enc_depth (int): Depth of block of transformer encoder. Default:  1.
        enc_with_pos (bool): Using position embedding in transformer encoder.
            Default:  True
        dec_depth (int): Depth of block of transformer decoder. Default:  8.
        num_heads (int): Number of Multi-Head Cross-Attention Head of transformer encoder.
            Default:  8.
        use_tokenizer (bool),Using semantic token. Default:  True
        token_len (int): Number of dims of token. Default:  4.
        pre_upsample (int): Scale factor of upsample of pre_process.
            (default upsample to 64x64)
            Default: 2.
    """

    def __init__(self,
                 in_channels=256,
                 channels=32,
                 encoder_head_dim=64,
                 decoder_head_dim=8,
                 enc_depth=1,
                 enc_with_pos=True,
                 dec_depth=8,
                 num_heads=8,
                 drop_rate=0.,
                 pool_size=2,
                 pool_mode='max',
                 use_tokenizer=True,
                 token_len=4,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(in_channels, channels, init_cfg=init_cfg,**kwargs)
        del self.conv_seg
        self.norm_cfg = norm_cfg
        self.encoder_head_dim = encoder_head_dim
        self.decoder_head_dim = decoder_head_dim
        self.use_tokenizer = use_tokenizer
        self.num_heads = num_heads
        if not use_tokenizer:
            # If a tokenzier is not to be used，then downsample the feature maps
            self.pool_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = pool_size * pool_size
        else:
            self.token_len = token_len
            self.conv_a = ConvModule(
                self.channels,
                self.token_len,
                1,
                conv_cfg=self.conv_cfg,
                act_cfg=None,
                bias=False
            )

        self.enc_with_pos = enc_with_pos
        if enc_with_pos:
            self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, self.channels))

        # pre_process to backbone feature
        self.pre_process = Sequential(
            nn.Upsample(scale_factor=2),
            ConvModule(
                self.in_channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                act_cfg=None
            )
        )

        # Transformer Encoder
        self.encoder = ModuleList()
        for _ in range(enc_depth):
            block = TransformerEncoder(
                self.channels,
                self.encoder_head_dim,
                self.num_heads,
                drop_rate,
                norm_cfg=self.norm_cfg,
            )
            self.encoder.append(block)

        # Transformer Decoder
        self.decoder = ModuleList()
        for _ in range(dec_depth):
            block = TransformerDecoder(
                self.channels,
                self.decoder_head_dim,
                self.num_heads,
                drop_rate=drop_rate,
                norm_cfg=self.norm_cfg,
            )
            self.decoder.append(block)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=3 // 2, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size=3,
                      padding=3 // 2, stride=1))

    # Token
    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshaped_tokens(self, x):
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == 'avg':
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        else:
            x = x
        tokens = x.permute((0, 2, 3, 1)).flatten(1, 2)
        return tokens

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)
        x1, x2 = torch.chunk(inputs, 2, dim=1)
        x1 = self.pre_process(x1)
        x2 = self.pre_process(x2)
        # Tokenization
        if self.use_tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshaped_tokens(x1)
            token2 = self._forward_reshaped_tokens(x2)

        # Transformer encoder forward
        token = torch.cat([token1, token2], dim=1)
        if self.enc_with_pos:
            token += self.enc_pos_embedding
        for i, _encoder in enumerate(self.encoder):
            token = _encoder(token)
        token1, token2 = torch.chunk(token, 2, dim=1)

        # Transformer decoder forward
        for _decoder in self.decoder:
            b, c, h, w = x1.shape
            x1 = x1.permute((0, 2, 3, 1)).flatten(1, 2)
            x2 = x2.permute((0, 2, 3, 1)).flatten(1, 2)

            x1 = _decoder(x1, token1)
            x2 = _decoder(x2, token2)

            x1 = x1.transpose(1, 2).reshape((b, c, h, w))
            x2 = x2.transpose(1, 2).reshape((b, c, h, w))

        # Feature differencing
        y = torch.abs(x1 - x2)
        y = self.upsample(y)

        return y

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.classifier(output)
        return output