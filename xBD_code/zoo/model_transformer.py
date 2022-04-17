import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34
import segmentation_models_pytorch as smp
from einops import rearrange

from importlib.machinery import SourceFileLoader
bitmodule = SourceFileLoader('bitmodule', 'zoo/bit_resnet.py').load_module()

import matplotlib.pyplot as plt
import random

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )

class TwoLayerConv2d_NoBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                        #  nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

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
        # vis_tmp2(out)
        return out

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

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = bitmodule.resnet18(pretrained=True, replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = bitmodule.resnet34(pretrained=True, replace_stride_with_dilation=[False,True,True])
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(384, 32, kernel_size=3, padding=1)
        self.conv_pred2 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_2 = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x_2) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        x_8_pool = self.resnet.maxpool(x_8)

        if self.resnet_stages_num > 3:
            x_10 = self.resnet.layer3(x_8_pool) # 1/8, in=128, out=256

        if self.resnet_stages_num > 4:
            raise NotImplementedError

        x = self.upsamplex2(x_10)
        x = torch.concat([x, x_8], axis=1)
        x = self.conv_pred(x)
        x = self.upsamplex2(x)

        x_up2 = torch.concat([x, x_4], axis=1)
        x_up2 = self.conv_pred2(x_up2)

        return x, x_up2

'''
class BASE_UNet_Transformer(ResNet_Encoder):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_UNet_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )

        print("using BiT Transformer !!!!") 

        self.token_len = token_len
        self.conv_a32 = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_a64 = nn.Conv2d(64, self.token_len, kernel_size=1, padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        self.dim = dim
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, dim))
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, self.token_len*2, dim*2))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, dim,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
            self.pos_embedding_decoder_2 =nn.Parameter(torch.randn(1, dim*2,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        self.transformer_2 = Transformer(dim=dim*2, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder_2 = TransformerDecoder(dim=dim*2, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        self.conv_remap256 = nn.Conv2d(256, dim, kernel_size=3, padding=1)
        self.conv_cat256 = nn.Conv2d(256*2, dim, kernel_size=3, padding=1)
        self.conv_remap128 = nn.Conv2d(128, dim, kernel_size=3, padding=1)
        self.conv_cat128 = nn.Conv2d(128*2, dim, kernel_size=3, padding=1)
        self.conv_remap64 = nn.Conv2d(64, dim*2, kernel_size=3, padding=1)
        self.conv_cat64 = nn.Conv2d(64*2, dim*2, kernel_size=3, padding=1)

        self.conv_concat = nn.Conv2d(96, dim, kernel_size=3, padding=1)

        # self.linear = nn.Linear()

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        if c == 32:
            spatial_attention = self.conv_a32(x)
        else:
            spatial_attention = self.conv_a64(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if x.shape[2] == self.dim:
            if self.with_pos:
                x += self.pos_embedding
            x = self.transformer(x)
        else:
            if self.with_pos:
                x += self.pos_embedding_2
            x = self.transformer_2(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if h == self.dim:
            # x = x + self.pos_embedding_decoder
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.transformer_decoder(x, m)
            x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        else:
            # x = x + self.pos_embedding_decoder_2
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.transformer_decoder_2(x, m)
            x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def trans_module(self, x1, x2):
        # diff
        if x1.shape[1] == 256:
            conv_map32 = self.conv_remap256
            conv_cat32 = self.conv_cat256
        elif x1.shape[1] == 128:
            conv_map32 = self.conv_remap128
            conv_cat32 = self.conv_cat128
        elif x1.shape[1] == 64:
            conv_map32 = self.conv_remap64
            conv_cat32 = self.conv_cat64

        # x = conv_cat32(torch.concat([x1, x2], axis=1))
        x1 = conv_map32(x1)
        x2 = conv_map32(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token = self._forward_reshape_tokens(x)

        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            token = self._forward_transformer(self.tokens_)
        # print("after transformer encoding token.shape", x.shape, token.shape)

        # forward transformer decoder
        x = conv_map32(torch.abs(x1 - x2))
        if self.with_decoder:
            x = self._forward_transformer_decoder(x, token)
        else:
            x = self._forward_simple_decoder(x, token)
        return x


    def forward(self, x):
        # forward backbone resnet
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        x1, x1_64, x1_64_2, x1_128, x1_256 = self.forward_single(x1)
        x2, x2_64, x2_64_2, x2_128, x2_256 = self.forward_single(x2)

        # print(x1.shape, x1_64.shape, x1_64_2.shape, x1_128.shape, x1_256.shape)
        x_256 = self.trans_module(x1_256, x2_256)
        x_256 = self.upsamplex2(x_256)

        # x_128 = self.trans_module(x1_128, x2_128)
        x_64 = self.trans_module(x1_64_2, x1_64_2)

        x = torch.concat([x_64, x_256], axis=1)
        x = self.conv_concat(x)
        x = self.upsamplex4(x)
        
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)

        # print("OUTPUT", x.shape)
        return x
'''



# unet x_4 as spatial encoding to decoder 
# without x_4 upsampling
class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )

        print("using BiT Transformer !!!!") 

        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_a_2 = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)

        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
            self.pos_embedding_decoder_2 =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)
        self.transformer_2 = Transformer(dim=dim, depth=self.enc_depth, heads=2,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder_2 = TransformerDecoder(dim=dim, depth=3,
                            heads=2, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x, level=1):
        b, c, h, w = x.shape
        if level == 1:
            spatial_attention = self.conv_a(x)
        else:
            spatial_attention = self.conv_a_2(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_reshape_tokens(self, x, level=1):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x, level=1):
        if level == 1:
            if self.with_pos:
                x += self.pos_embedding
            x = self.transformer(x)
        else:
            if self.with_pos:
                x += self.pos_embedding_2
            x = self.transformer_2(x)
        return x

    def _forward_transformer_decoder(self, x, m, level=1):
        b, c, h, w = x.shape
        if level == 1:
            if self.with_decoder_pos == 'learned':
                x = x + self.pos_embedding_decoder
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.transformer_decoder(x, m)
        else:
            if self.with_decoder_pos == 'learned':
                x = x + self.pos_embedding_decoder_2
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.transformer_decoder_2(x, m)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m, level=1):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x):
        # forward backbone resnet
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        x1, x1_up2 = self.forward_single(x1)
        x2, x2_up2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        self.tokens_ = torch.cat([token1, token2], dim=1)
        self.tokens = self._forward_transformer(self.tokens_)
        token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)
        # feature differencing
        x_out1 = torch.abs(x1 - x2)
        # x_out1 = self.upsamplex2(x)

        # print("1st: after diff and upsample2", x_out1.shape)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1_up2, level=2)
        token2 = self._forward_semantic_tokens(x2_up2, level=2)
        # forward transformer encoder
        self.tokens_ = torch.cat([token1, token2], dim=1)
        self.tokens = self._forward_transformer(self.tokens_, level=2)
        token1, token2 = self.tokens.chunk(2, dim=1)
        x1 = self._forward_transformer_decoder(x1_up2, token1, level=2)
        x2 = self._forward_transformer_decoder(x2_up2, token2, level=2)
        
        # feature differencing
        x_out2 = torch.abs(x1 - x2)
        # print("2nd: after diff and upsample2", x_out2.shape)

        x = x_out1 + x_out2
        # x = self.upsamplex2(x)

        x = self.upsamplex4(x)

        # forward small cnn
        x = self.classifier(x)
        return x
