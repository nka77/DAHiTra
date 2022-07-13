import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34
#import segmentation_models_pytorch as smp
from einops import rearrange

from importlib.machinery import SourceFileLoader
bitmodule = SourceFileLoader('bitmodule', 'zoo/bit_resnet.py').load_module()
from torchvision.models import efficientnet_b0, resnet18

import matplotlib.pyplot as plt
import random

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1))

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


class ResNet_UNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_UNet, self).__init__()
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

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x_2 = self.resnet.relu(x)
        x_2_pool = self.resnet.maxpool(x)
        
        x_4 = self.resnet.layer1(x_2_pool) # 1/4, in=64, out=64

        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        x_8_pool = self.resnet.maxpool(x_8)

        x_10 = self.resnet.layer3(x_8_pool) # 1/8, in=128, out=256

        if self.resnet_stages_num > 4:
            raise NotImplementedError

        # print(x_2.shape, x_4.shape, x_8.shape, x_10.shape)
        x = self.upsamplex2(x_10)

        return x_2, x_4, x_8, x_10


# unet x_4 as spatial encoding to decoder 
# without x_4 upsampling
class BASE_Transformer_UNet(ResNet_UNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos=None, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer_UNet, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )

        print("using UNet Transformer !!!!") 

        self.token_len = token_len
        self.tokenizer = tokenizer
        self.token_trans = token_trans
        self.with_decoder = with_decoder
        self.with_pos = with_pos

        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        
        # conv squeeze layers before transformer
        dim_5, dim_4, dim_3, dim_2 = 32, 32, 32, 32
        self.conv_squeeze_5 = nn.Sequential(nn.Conv2d(256, dim_5, kernel_size=1, padding=0, bias=False),
                                            nn.ReLU())
        self.conv_squeeze_4 = nn.Sequential(nn.Conv2d(128, dim_4, kernel_size=1, padding=0, bias=False),
                                            nn.ReLU())                                    
        self.conv_squeeze_3 = nn.Sequential(nn.Conv2d(64, dim_3, kernel_size=1, padding=0, bias=False),
                                            nn.ReLU())
        self.conv_squeeze_2 = nn.Sequential(nn.Conv2d(64, dim_2, kernel_size=1, padding=0, bias=False),
                                            nn.ReLU())
        self.conv_squeeze_layers = nn.ModuleList([self.conv_squeeze_2, self.conv_squeeze_3, self.conv_squeeze_4, self.conv_squeeze_5])
                                                                                
        self.conv_token_5 = nn.Conv2d(dim_5, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_token_4 = nn.Conv2d(dim_4, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_token_3 = nn.Conv2d(dim_3, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_token_2 = nn.Conv2d(dim_2, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_tokens_layers = nn.ModuleList([self.conv_token_2, self.conv_token_3, self.conv_token_4, self.conv_token_5])


        self.conv_decode_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.conv_decode_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.conv_decode_3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.conv_decode_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.conv_decode_layers = nn.ModuleList([self.conv_decode_2, self.conv_decode_3, self.conv_decode_4, self.conv_decode_5])


        if with_pos == 'learned':
            self.pos_embedding_5 = nn.Parameter(torch.randn(1, self.token_len*2, dim_5))
            self.pos_embedding_4 = nn.Parameter(torch.randn(1, self.token_len*2, dim_4))
            self.pos_embedding_3 = nn.Parameter(torch.randn(1, self.token_len*2, dim_3))
        
        decoder_pos_size = 256//2
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder_5 =nn.Parameter(torch.randn(1, dim_5, 16, 16))
            self.pos_embedding_decoder_4 =nn.Parameter(torch.randn(1, dim_4, 32, 32))
            self.pos_embedding_decoder_3 =nn.Parameter(torch.randn(1, dim_3, 64, 64))

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer_5 = Transformer(dim=dim_5, depth=self.enc_depth, heads=4,
                                        dim_head=self.dim_head, mlp_dim=dim_5, dropout=0)
        self.transformer_decoder_5 = TransformerDecoder(dim=dim_5, depth=4, heads=4, 
                                                        dim_head=self.decoder_dim_head, mlp_dim=dim_5, dropout=0, softmax=decoder_softmax)
        self.transformer_4 = Transformer(dim=dim_4, depth=self.enc_depth, heads=4,
                                        dim_head=self.dim_head, mlp_dim=dim_4, dropout=0)
        self.transformer_decoder_4 = TransformerDecoder(dim=dim_4, depth=4, heads=4, dim_head=self.decoder_dim_head,
                                                         mlp_dim=dim_4, dropout=0, softmax=decoder_softmax)
        self.transformer_3 = Transformer(dim=dim_3, depth=self.enc_depth, heads=8,
                                         dim_head=self.dim_head, mlp_dim=dim_3, dropout=0)
        self.transformer_decoder_3 = TransformerDecoder(dim=dim_3, depth=8, heads=8, dim_head=self.decoder_dim_head, 
                                                        mlp_dim=dim_3, dropout=0, softmax=decoder_softmax)
        self.transformer_2 = Transformer(dim=dim_2, depth=self.enc_depth, heads=1,
                                         dim_head=32, mlp_dim=dim_2, dropout=0)
        self.transformer_decoder_2 = TransformerDecoder(dim=dim_2, depth=1, heads=1, dim_head=32, 
                                                        mlp_dim=dim_2, dropout=0, softmax=decoder_softmax)                                           
        self.transformer_layers = nn.ModuleList([self.transformer_2, self.transformer_3, self.transformer_4, self.transformer_5])
        self.transformer_decoder_layers = nn.ModuleList([self.transformer_decoder_2, self.transformer_decoder_3, self.transformer_decoder_4, self.transformer_decoder_5])

        self.conv_layer2_0 = TwoLayerConv2d(in_channels=128, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                        nn.ReLU())
        self.conv_layer3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                        nn.ReLU())
        self.conv_layer4 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                        nn.ReLU())
        self.classifier = nn.Conv2d(in_channels=32, out_channels=output_nc, kernel_size=3, padding=1)

        #self.seg_head = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        #self.cls_head = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, padding=1)


    def _forward_semantic_tokens(self, x, layer=None):
        b, c, h, w = x.shape
        spatial_attention = self.conv_tokens_layers[layer](x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_transformer(self, x, layer):
        if self.with_pos:
            if layer == 5:
                x = x + self.pos_embedding_5
            if layer == 4:
                x = x + self.pos_embedding_4
            if layer == 3:
                x = x + self.pos_embedding_3
            #x += self.pos_embedding_layers[layer]
        x = self.transformer_layers[layer](x)
        return x

    def _forward_transformer_decoder(self, x, m, layer):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'learned':
            if layer == 5:
                x = x + self.pos_embedding_decoder_5
            if layer == 4:
                x = x + self.pos_embedding_decoder_4
            if layer == 3:
                x = x + self.pos_embedding_decoder_3
            #x = x + self.pos_embedding_decoder_layers[layer]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder_layers[layer](x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_trans_module(self, x1, x2, layer):
        x1 = self.conv_squeeze_layers[layer](x1)
        x2 = self.conv_squeeze_layers[layer](x2)
        token1 = self._forward_semantic_tokens(x1, layer)
        token2 = self._forward_semantic_tokens(x2, layer)
        self.tokens_ = torch.cat([token1, token2], dim=1)
        self.tokens = self._forward_transformer(self.tokens_, layer)
        token1, token2 = self.tokens.chunk(2, dim=1)
        # x1 = self._forward_transformer_decoder(x1, token1, layer)
        # x2 = self._forward_transformer_decoder(x2, token2, layer)
        # return torch.abs(x1 - x2)

        # V1, V2
        # x1 = self._forward_transformer_decoder(x1, token2, layer)
        # x2 = self._forward_transformer_decoder(x2, token1, layer)
        # return torch.add(x1, x2)

        # # V3
        diff_token = torch.abs(token2 - token1)
        diff_x = self.conv_decode_layers[layer](torch.cat([x1,x2], axis=1))
        x = self._forward_transformer_decoder(diff_x, diff_token, layer)
        return x


    def forward(self, x):
        # forward backbone resnet
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        a_128, a_64, a_32, a_16 = self.forward_single(x1)
        b_128, b_64, b_32, b_16 = self.forward_single(x2)

        #  level 5 in=256x16x16 out=32x16x16
        x1, x2 = a_16, b_16
        out_5 = self._forward_trans_module(x1, x2, layer=3)
        out_5 = self.upsamplex2(out_5)

        # level 4: in=128x32x32 out=32x32x32
        x1, x2 = a_32, b_32
        out_4 = self._forward_trans_module(x1, x2, layer=2)
        out_4 = out_4 + out_5
        # out_4 = self.conv_layer4(torch.cat([out_4, out_5], axis=1))
        out_4 = self.upsamplex2(out_4)  
        out_4 = self.conv_layer4(out_4)

        # level 3: in=64x64x64 out=32x64x64
        x1, x2 = a_64, b_64
        out_3 = self._forward_trans_module(x1, x2, layer=1)
        out_3 = out_3 + out_4
        # out_3 = self.conv_layer3(torch.cat([out_3, out_4], axis=1))
        out_3 = self.upsamplex2(out_3)
        out_3 = self.conv_layer3(out_3)

        # level 2: in=64x128x128
        out_2 = self.conv_layer2_0(torch.cat([a_128, b_128], 1))
        out_2 = out_2 + out_3
        # out_2 = self.conv_layer2(torch.cat([out_2, out_3], axis=1))
        out_2 = self.upsamplex2(out_2)
        out_2 = self.conv_layer2(out_2)
        # print(out_2.shape, out_3.shape, out_4.shape, out_5.shape)
        # forward small cnn
        x = self.classifier(out_2)
        # x_seg = self.seg_head(out_2)
        # x_cls = self.cls_head(out_2)
        # x = torch.cat([x_seg, x_cls], axis=1)
        return x




class Discriminator(torch.nn.Module):
    def __init__(self, input_nc=5):
        super(Discriminator, self).__init__()
        self.pre_process = nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, padding=0)
        self.backbone = resnet18(pretrained=True)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.backbone(x)
        return x



class UNet_Loc(ResNet_UNet):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18'):
        super(UNet_Loc, self).__init__()

    def forward(self, x):
        # forward backbone resnet
        a_128, a_64, a_32, a_16 = self.forward_single(x)

        #  level 5 in=256x16x16 out=32x16x16
        out_5 = self.upsamplex2(a_16)

        out_4 = self.conv_layer4(torch.cat([out_4, out_5], axis=1))
        out_4 = self.upsamplex2(out_4)

        # level 3: in=64x64x64 out=32x64x64
        x1, x2 = a_64, b_64
        out_3 = self._forward_trans_module(x1, x2, layer=1)
        out_3 = out_3 + out_4
        # out_3 = self.conv_layer3(torch.cat([out_3, out_4], axis=1))
        out_3 = self.upsamplex2(out_3)

        # level 2: in=64x128x128
        out_2 = self.conv_layer2_0(torch.cat([a_128, b_128], 1))
        out_2 = out_2 + out_3
        # out_2 = self.conv_layer2(torch.cat([out_2, out_3], axis=1))
        out_2 = self.upsamplex2(out_2)

        # print(out_2.shape, out_3.shape, out_4.shape, out_5.shape)
        # forward small cnn
        x = self.classifier(out_2)
        return x
