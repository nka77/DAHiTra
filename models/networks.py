import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision.models import resnet34

import functools
from einops import rearrange

import numpy as np

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from torch.nn.modules.padding import ReplicationPad2d

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimxizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, 
        milestones=[2, 4, 7, 11, 15, 25, 35, 47, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)
    elif args.net_G == 'base_transformer_pos_s4_dd8_o5':
        net = BASE_Transformer(input_nc=3, output_nc=5, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    elif args.net_G == 'base_transformer_pos_s4_dd8_t8_e2d4':
       net = BASE_Transformer(input_nc=3, output_nc=2, token_len=8, resnet_stages_num=4,
                              with_pos='learned', enc_depth=2, dec_depth=4, decoder_dim_head=8)
    elif args.net_G == 'unet_coupled_trans_256':
       net = UNet_Change_Transformer()
    elif args.net_G == 'unet_coupled_two_trans_256':
       net = UNet_Change_Two_Transformer()
    elif args.net_G == 'siamUnet_conc':
       net = SiamUnet_conc(input_nbr=3, label_nbr=2)
    elif args.net_G == 'siamUnet':
       net = Res34_Unet_Double()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


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
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
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
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


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
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
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
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
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

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
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
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
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

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)

        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x



class ConvReluBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvReluBN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ChannelAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(in_channels*2, out_channels, kernel_size, padding=padding, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        x = self.conv1(x)
        return self.tanh(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


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

'''
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi
'''
# Attention for bottleneck layers
class ChannelAttention_OnBottle(nn.Module):
    def __init__(self, in_planes, ratio=16, att_type='max'):
        super(ChannelAttention_OnBottle, self).__init__()
        self.att_type = att_type
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.min_pool = nn.AdaptiveMinPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(64, 512)
        self.fc4 = nn.Linear(96, 512)

    def forward(self, x):
        if self.att_type == 'max':
            out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        elif self.att_type == 'max_avg':
            max_out = self.relu(self.fc1(self.max_pool(x)))
            avg_out = self.relu(self.fc1(self.avg_pool(x))) 
            out = torch.cat([max_out, avg_out], 1).squeeze()
            out = self.fc3(out).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        elif self.att_type == 'avg_max_min':
            avg_out = self.relu(self.fc1(self.avg_pool(x)))
            # min_out = self.relu(self.fc1(self.min_pool(x)))
            max_out = self.relu(self.fc1(self.max_pool(x)))
            out = torch.cat([avg_out, min_out, max_out], 1).squeeze()
            out = self.fc4(out).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return self.relu(out)


class UNet_Change_Transformer(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(UNet_Change_Transformer, self).__init__()
        
        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = np.asarray([48, 64, 96, 128, 320])

        self.encoder_filters = [64, 64, 128, 256, 512]
        self.decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2]*2 , decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3]*2 , decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4]*2 , decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5]*2 , decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.conv0 = ConvRelu(6,3)
        self.conv1 = nn.Sequential(
                        encoder.conv1,
                        encoder.bn1,
                        encoder.relu)
        self.conv2 = nn.Sequential(
                        encoder.maxpool,
                        encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

        self.ca_skip_5 = ChannelAttention(encoder_filters[-1], encoder_filters[-1])
        self.ca_skip_4 = ChannelAttention(encoder_filters[-2], encoder_filters[-2])
        self.ca_skip_3 = ChannelAttention(encoder_filters[-3], encoder_filters[-3])
        self.ca_skip_2 = ChannelAttention(encoder_filters[-4], encoder_filters[-4])
        self.ca_skip_1 = ChannelAttention(encoder_filters[-5], encoder_filters[-5])

        self.ca_bottle_max = ChannelAttention_OnBottle(512, att_type='max')
        self.ca_bottle_avg_min = ChannelAttention_OnBottle(512, att_type='max_avg')
        self.sigmoid = nn.Sigmoid()
        self.linearb = nn.Linear(1024, 512)

        dim = 64
        mlp_dim = 2*dim
        enc_depth = 2
        dim_head = 64
        decoder_dim_head = 64
        decoder_softmax = True
        self.transformer = Transformer(dim=dim, depth=3, heads=4,
                                       dim_head=dim_head,
                                       mlp_dim=mlp_dim, dropout=0.05)

        self.transformer_decoder = TransformerDecoder(dim=dim, depth=2,
                            heads=8, dim_head=decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)
        self.classifier = TwoLayerConv2d(in_channels=512, out_channels=2)


    def forward_1(self, x1, x2):

        # Encoder 1
        x_1 = x1
        enc1_1 = self.conv1(x_1)
        enc2_1 = self.conv2(enc1_1)
        enc3_1 = self.conv3(enc2_1)
        enc4_1 = self.conv4(enc3_1)
        enc5_1 = self.conv5(enc4_1)

        # Encoder 2
        x_2 = x2
        enc1_2 = self.conv1(x_2)
        enc2_2 = self.conv2(enc1_2)
        enc3_2 = self.conv3(enc2_2)
        enc4_2 = self.conv4(enc3_2)
        enc5_2 = self.conv5(enc4_2)

        # Bottleneck
        enc5_1 = (self.ca_bottle_max(enc5_1)*enc5_1)
        enc5_2 = (self.ca_bottle_max(enc5_2)*enc5_2)

        enc5 = self.ca_skip_5(enc5_1,enc5_2)

        B_, C_, H_, W_ = enc5.shape
        enc5_i = enc5.view([B_, C_, H_*W_])
        enc5_i = self.transformer(enc5_i)
        enc5_i = enc5_i.view([B_, C_, H_, W_])
        enc5 = self.ca_skip_5(enc5_i,enc5)

        # Decoder
        enc4 = self.ca_skip_4(enc4_1, enc4_2)
        # enc4 = attention_block(enc4_1, enc4_2, self.encoder_filters[-2])
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        enc3 = self.ca_skip_3(enc3_1, enc3_2)
        # enc3 = attention_block(enc3_1, enc3_2, self.encoder_filters[-3])
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        enc2 = self.ca_skip_2(enc2_1, enc2_2)
        # enc2 = attention_block(enc2_1, enc2_2, self.encoder_filters[-4])
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        enc1 = self.ca_skip_2(enc1_1, enc1_2)
        # enc1 = attention_block(enc1_1, enc1_2, self.encoder_filters[-5])
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        out = self.res(dec10)
        return out

    def forward(self, x1, x2):
        # Encoder 1
        x_1 = x1
        enc1_1 = self.conv1(x_1)
        enc2_1 = self.conv2(enc1_1)
        enc3_1 = self.conv3(enc2_1)
        enc4_1 = self.conv4(enc3_1)
        enc5_1 = self.conv5(enc4_1)

        # Encoder 2
        x_2 = x2
        enc1_2 = self.conv1(x_2)
        enc2_2 = self.conv2(enc1_2)
        enc3_2 = self.conv3(enc2_2)
        enc4_2 = self.conv4(enc3_2)
        enc5_2 = self.conv5(enc4_2)

        # Bottleneck
        enc5_1 = (self.ca_bottle_max(enc5_1)*enc5_1)
        enc5_2 = (self.ca_bottle_max(enc5_2)*enc5_2)
        enc5_c = self.ca_skip_5(enc5_1,enc5_2)

        B_, C_, H_, W_ = enc5_c.shape
        enc5_i = enc5_c.view([B_, C_, H_*W_])
        enc5_i = self.transformer(enc5_i)
        enc5 = enc5_i.view([B_, C_, H_, W_])
        enc5 = self.ca_skip_5(enc5,enc5_c)

        # Decoder
        # enc4 = self.ca_skip_4(enc4_1, enc4_2)
        # enc4 = attention_block(enc4_1, enc4_2, self.encoder_filters[-2])
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4_1, enc4_2
                ], 1))

        # enc3 = self.ca_skip_3(enc3_1, enc3_2)
        # enc3 = attention_block(enc3_1, enc3_2, self.encoder_filters[-3])
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3_1, enc3_2
                ], 1))
        
        # enc2 = self.ca_skip_2(enc2_1, enc2_2)
        # enc2 = attention_block(enc2_1, enc2_2, self.encoder_filters[-4])
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2_1, enc2_2
                ], 1))

        # enc1 = self.ca_skip_2(enc1_1, enc1_2)
        # enc1 = attention_block(enc1_1, enc1_2, self.encoder_filters[-5])
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1_1, enc1_2], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        out = self.res(dec10)

        # enc5_c = enc5_c.view([B_, C_, H_*W_])
        # interim_out = self.transformer_decoder(enc5_i, enc5_c)
        # interim_out = interim_out.view([B_, C_, H_, W_])
        # interim_out = self.classifier(enc5)

        return out #, interim_out


        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNet_Change_Two_Transformer(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(UNet_Change_Two_Transformer, self).__init__()
        
        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = np.asarray([48, 64, 96, 128, 320])

        self.encoder_filters = [64, 64, 128, 256, 512]
        self.decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2]*2, decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3]*2, decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4] , decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5]*2 , decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], 5, 1, stride=1, padding=0)

        


        self._initialize_weights()

        encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.conv1 = nn.Sequential(
                        encoder.conv1,
                        encoder.bn1,
                        encoder.relu)
        self.conv2 = nn.Sequential(
                        encoder.maxpool,
                        encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

        self.ca_skip_5 = ChannelAttention(encoder_filters[-1], encoder_filters[-1])
        self.ca_skip_4 = ChannelAttention(encoder_filters[-2], encoder_filters[-2])
        self.ca_skip_3 = ChannelAttention(encoder_filters[-3], encoder_filters[-3])
        self.ca_skip_2 = ChannelAttention(encoder_filters[-4], encoder_filters[-4])
        self.ca_skip_1 = ChannelAttention(encoder_filters[-5], encoder_filters[-5])

        self.ca_bottle_max = ChannelAttention_OnBottle(512, att_type='max')
        self.ca_bottle_avg_min = ChannelAttention_OnBottle(512, att_type='max_avg')
        self.sigmoid = nn.Sigmoid()
        self.linearb = nn.Linear(1024, 512)

        dim = 64
        dim2 = 4096
        dim3 = 1024
        mlp_dim = 2*dim
        enc_depth = 2
        dim_head = 64
        decoder_dim_head = 64
        decoder_softmax = True
        self.transformer = Transformer(dim=dim, depth=3, heads=4,
                                       dim_head=dim_head,
                                       mlp_dim=mlp_dim, dropout=0.01)

        self.transformer2 = Transformer(dim=dim2, depth=2, heads=2,
                                       dim_head=dim_head,
                                       mlp_dim=dim2, dropout=0.001)

        self.transformer3 = Transformer(dim=dim3, depth=2, heads=2,
                                       dim_head=dim_head,
                                       mlp_dim=dim3, dropout=0.001)
 
        self.classifier = TwoLayerConv2d(in_channels=512, out_channels=2)
        self.convT = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)


    def forward(self, x1, x2):
        # Encoder 1
        x_1 = x1
        enc1_1 = self.conv1(x_1)
        enc2_1 = self.conv2(enc1_1)
        enc3_1 = self.conv3(enc2_1)
        enc4_1 = self.conv4(enc3_1)
        enc5_1 = self.conv5(enc4_1)

        # Encoder 2
        x_2 = x2
        enc1_2 = self.conv1(x_2)
        enc2_2 = self.conv2(enc1_2)
        enc3_2 = self.conv3(enc2_2)
        enc4_2 = self.conv4(enc3_2)
        enc5_2 = self.conv5(enc4_2)

        # Bottleneck
        # enc5_1 = (self.ca_bottle_max(enc5_1)*enc5_1)
        # enc5_2 = (self.ca_bottle_max(enc5_2)*enc5_2)
        # enc5_c = self.ca_skip_5(enc5_1,enc5_2)

        ## run 1: updating channel attention
        enc5 = self.ca_skip_5(enc5_1,enc5_2)
        B_, C_, H_, W_ = enc5.shape
        enc5_i = enc5.view([B_, C_, H_*W_]).contiguous()

        enc5_diff = (enc5_1 - enc5_2)
        spatial_attention = enc5_diff.view([B_, C_, H_*W_]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)

        enc5_i = torch.einsum('bln,bln->bln', spatial_attention, enc5_i)
        enc5_t = self.transformer(enc5_i)
        enc5 = enc5_t.view([B_, C_, H_, W_]).contiguous()
        # enc5 = self.ca_skip_5(enc5_t,enc5)

        # Decoder
        # enc4 = self.ca_skip_4(enc4_1, enc4_2)
        # B_, C_, H_, W_ = enc4.shape
        # enc4 = enc4.view([B_, C_, H_*W_]).contiguous()

        # enc4_diff = (enc4_1 - enc4_2)
        # spatial_attention = enc4_diff.view([B_, C_, H_*W_]).contiguous()
        # spatial_attention = torch.softmax(spatial_attention, dim=-1)

        # enc4 = torch.einsum('bln,bln->bln', spatial_attention, enc4)
        # enc4 = enc4.view([B_, C_, H_,W_]).contiguous()

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4_1, enc4_2], 1))

        # run4: depth=2, heads=2
        enc3 = self.ca_skip_3(enc3_1, enc3_2)
        # B_, C_, H_, W_ = enc3.shape
        # enc3_i = enc3.view([B_, C_, H_*W_]).contiguous()

        # enc3_diff = (enc3_1 - enc3_2)
        # spatial_attention = enc3_diff.view([B_, C_, H_*W_]).contiguous()
        # spatial_attention = torch.softmax(spatial_attention, dim=-1)

        # enc3_i = torch.einsum('bln,bln->bln', spatial_attention, enc3_i)
        # enc3_t = self.transformer3(enc3_i)
        # enc3_t = enc3_t.view([B_, C_, H_, W_]).contiguous()
        # enc3 = self.ca_skip_3(enc3, enc3_t)

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3_1, enc3_2], 1))
        
        ## run0: depth=2, heads=1
        enc2 = self.ca_skip_2(enc2_1, enc2_2)
        B_, C_, H_, W_ = enc2.shape
        enc2 = enc2.view([B_, C_, H_*W_]).contiguous()

        enc2_diff = (enc2_1 - enc2_2)
        spatial_attention = enc2_diff.view([B_, C_, H_*W_]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)

        enc2 = torch.einsum('bln,bln->bln', spatial_attention, enc2)
        enc2 = self.transformer2(enc2)
        enc2 = enc2.view([B_, C_, H_,W_]).contiguous()

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        # enc1 = self.ca_skip_2(enc1_1, enc1_2)
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1_1, enc1_2], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        out = self.res(dec10)

        return out #, interim_out


        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SiamUnet_conc(nn.Module):
    """SiamUnet_conc segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(SiamUnet_conc, self).__init__()

        self.input_nbr = input_nbr

        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(384, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(192, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(96, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(48, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=3, padding=1)

        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):

        """Forward method."""
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)


        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)


        ####################################################
        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43_1, x43_2), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33_1, x33_2), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22_1, x22_2), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12_1, x12_2), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return self.sm(x11d)




class Res34_Unet_Double(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(Res34_Unet_Double, self).__init__()
        
        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)
        #self.res = nn.Conv2d(decoder_filters[-5], 5, 1, stride=1, padding=0)
        self._initialize_weights()

        encoder = resnet34(pretrained=pretrained)
        self.conv0 = ConvRelu(6,3)
        self.conv1 = nn.Sequential(
                        encoder.conv1,
                        encoder.bn1,
                        encoder.relu)
        self.conv2 = nn.Sequential(
                        encoder.maxpool,
                        encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def forward1(self, x):
        batch_size, C, H, W = x.shape
        #x = self.conv0(x)
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10

    def forward(self, x1, x2):
        dec10_0 = self.forward1(x1)
        dec10_1 = self.forward1(x2)
        x = torch.cat([dec10_0, dec10_1], 1)
        #dec10 = self.sa(dec10) * dec10
        #x = self.forward1(x)
        return self.res(x)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



### Three layers UNet bottleneck -> siamese ###
''' # Encoder 1
        x_1 = x1
        enc1_1 = self.conv1(x_1)
        enc2_1 = self.conv2(enc1_1)
        enc3_1 = self.conv3(enc2_1)

        # Encoder 2
        x_2 = x2
        enc1_2 = self.conv1(x_2)
        enc2_2 = self.conv2(enc1_2)
        enc3_2 = self.conv3(enc2_2)

        enc3 = self.ca_skip_3(enc3_1, enc3_2)
        B_, C_, H_, W_ = enc3.shape
        enc3_i = enc3.view([B_, C_, H_*W_])
        enc3_i = self.transformer(enc3_i)
        enc3_i = enc3_i.view([B_, C_, H_, W_])
        enc3 = self.ca_skip_3(enc3_i,enc3)

        enc2 = self.ca_skip_2(enc2_1, enc2_2)
        dec8 = self.conv8(F.interpolate(enc3, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        enc1 = self.ca_skip_2(enc1_1, enc1_2)
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        out = self.res(dec10)
        return out
'''