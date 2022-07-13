import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34
from .senet import se_resnext50_32x4d, senet154
from einops import rearrange

from importlib.machinery import SourceFileLoader
bitmodule = SourceFileLoader('bitmodule', 'zoo/bit_resnet.py').load_module()

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


class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, channels, reduction=16, concat=False):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return chn_se + spa_se


class SeResNext50_Unet_Loc(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(SeResNext50_Unet_Loc, self).__init__()
        
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        
        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)

        # conv1_new = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # _w = encoder.layer0.conv1.state_dict()
        # _w['weight'] = torch.cat([0.5 * _w['weight'], 0.5 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward(self, x):
        batch_size, C, H, W = x.shape

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

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SeResNext50_Unet_Double(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(SeResNext50_Unet_Double, self).__init__()
        
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

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

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)

        # conv1_new = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # _w = encoder.layer0.conv1.state_dict()
        # _w['weight'] = torch.cat([0.5 * _w['weight'], 0.5 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward1(self, x):
        batch_size, C, H, W = x.shape

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


    def forward(self, x):

        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Dpn92_Unet_Loc(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(Dpn92_Unet_Loc, self).__init__()
        
        encoder_filters = [64, 336, 704, 1552, 2688]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1]+encoder_filters[-2], decoder_filters[-1]), SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2]+encoder_filters[-3], decoder_filters[-2]), SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3]+encoder_filters[-4], decoder_filters[-3]), SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4]+encoder_filters[-5], decoder_filters[-4]), SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2, decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)
        
        self._initialize_weights()

        encoder = dpn92(pretrained=pretrained)

        # conv1_new = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # _w = encoder.blocks['conv1_1'].conv.state_dict()
        # _w['weight'] = torch.cat([0.5 * _w['weight'], 0.5 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        
        self.conv1 = nn.Sequential(
                encoder.blocks['conv1_1'].conv,  # conv
                encoder.blocks['conv1_1'].bn,  # bn
                encoder.blocks['conv1_1'].act,  # relu
            )
        self.conv2 = nn.Sequential(
                encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        self.conv3 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        self.conv4 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        self.conv5 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    def forward(self, x):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        enc1 = (torch.cat(enc1, dim=1) if isinstance(enc1, tuple) else enc1)
        enc2 = (torch.cat(enc2, dim=1) if isinstance(enc2, tuple) else enc2)
        enc3 = (torch.cat(enc3, dim=1) if isinstance(enc3, tuple) else enc3)
        enc4 = (torch.cat(enc4, dim=1) if isinstance(enc4, tuple) else enc4)
        enc5 = (torch.cat(enc5, dim=1) if isinstance(enc5, tuple) else enc5)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Res34_Unet_Single(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super(Res34_Unet_Single, self).__init__()
        
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
        
        self.res = nn.Conv2d(decoder_filters[-5], 5, 1, stride=1, padding=0)
                
        self._initialize_weights()

        encoder = resnet34(pretrained=pretrained)
        tmp_conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1 = nn.Sequential(
                        #encoder.conv1,
                        tmp_conv1,
                        encoder.bn1,
                        encoder.relu)
        self.conv2 = nn.Sequential(
                        encoder.maxpool,
                        encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def forward(self, x):
        batch_size, C, H, W = x.shape
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

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Dpn92_Unet_Double(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(Dpn92_Unet_Double, self).__init__()
        
        encoder_filters = [64, 336, 704, 1552, 2688]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1]+encoder_filters[-2], decoder_filters[-1]), SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2]+encoder_filters[-3], decoder_filters[-2]), SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3]+encoder_filters[-4], decoder_filters[-3]), SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4]+encoder_filters[-5], decoder_filters[-4]), SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2, decoder_filters[-5])

        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)
        
        self._initialize_weights()

        encoder = dpn92(pretrained=pretrained)

        # conv1_new = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # _w = encoder.blocks['conv1_1'].conv.state_dict()
        # _w['weight'] = torch.cat([0.5 * _w['weight'], 0.5 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        
        self.conv1 = nn.Sequential(
                encoder.blocks['conv1_1'].conv,  # conv
                encoder.blocks['conv1_1'].bn,  # bn
                encoder.blocks['conv1_1'].act,  # relu
            )
        self.conv2 = nn.Sequential(
                encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        self.conv3 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        self.conv4 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        self.conv5 = nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])


    def forward1(self, x):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        enc1 = (torch.cat(enc1, dim=1) if isinstance(enc1, tuple) else enc1)
        enc2 = (torch.cat(enc2, dim=1) if isinstance(enc2, tuple) else enc2)
        enc3 = (torch.cat(enc3, dim=1) if isinstance(enc3, tuple) else enc3)
        enc4 = (torch.cat(enc4, dim=1) if isinstance(enc4, tuple) else enc4)
        enc5 = (torch.cat(enc5, dim=1) if isinstance(enc5, tuple) else enc5)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10


    def forward(self, x):

        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Res34_Unet_Loc(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(Res34_Unet_Loc, self).__init__()
        
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
        
        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = resnet34(pretrained=pretrained)
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

    def forward(self, x):
        batch_size, C, H, W = x.shape

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

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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
        self.sa = SpatialAttention(kernel_size=3)

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

    def forward(self, x):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])
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

class SeNet154_Unet_Loc(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(SeNet154_Unet_Loc, self).__init__()
        
        encoder_filters = [128, 256, 512, 1024, 2048]
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
        
        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = senet154(pretrained=pretrained)

        # conv1_new = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # _w = encoder.layer0.conv1.state_dict()
        # _w['weight'] = torch.cat([0.8 * _w['weight'], 0.1 * _w['weight'], 0.1 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1, encoder.layer0.conv2, encoder.layer0.bn2, encoder.layer0.relu2, encoder.layer0.conv3, encoder.layer0.bn3, encoder.layer0.relu3)
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4


    def forward(self, x):
        batch_size, C, H, W = x.shape

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

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SeNet154_Unet_Double(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(SeNet154_Unet_Double, self).__init__()
        
        encoder_filters = [128, 256, 512, 1024, 2048]
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

        self._initialize_weights()

        encoder = senet154(pretrained=pretrained)

        # conv1_new = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # _w = encoder.layer0.conv1.state_dict()
        # _w['weight'] = torch.cat([0.8 * _w['weight'], 0.1 * _w['weight'], 0.1 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1, encoder.layer0.conv2, encoder.layer0.bn2, encoder.layer0.relu2, encoder.layer0.conv3, encoder.layer0.bn3, encoder.layer0.relu3)
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def forward1(self, x):
        batch_size, C, H, W = x.shape

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

    def forward(self, x):

        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)

        return self.res(dec10)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.tanh(x)

class Deeplabv3_Double(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(Deeplabv3_Double, self).__init__()       
        self.loc_output = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None)
        #self.conv1 = nn.Conv1d(in_channels=2, out_channels=5, kernel_size=3)
        self.conv2 = ConvRelu(in_channels=2, out_channels=5)
        self.sa = SpatialAttention(kernel_size=3)


    def forward1(self, x):
        batch_size, C, H, W = x.shape
        x = self.loc_output(x)
        #x = self.sa(x) * x
        return x

    def forward(self, x):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])
        x = torch.cat([dec10_0, dec10_1], 1)
        #x = self.sa(x) * x
        x = self.conv2(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Res34_Unet_Double_Modified(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(Res34_Unet_Double_Modified, self).__init__()
        
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
        
        self.conv1d_1 = nn.Conv2d(in_channels=encoder_filters[0]*2, out_channels=encoder_filters[0], kernel_size=1)
        self.conv1d_2 = nn.Conv2d(in_channels=encoder_filters[1]*3, out_channels=encoder_filters[1], kernel_size=1)
        self.conv1d_3 = nn.Conv2d(in_channels=encoder_filters[2]*3, out_channels=encoder_filters[2], kernel_size=1)
        self.conv1d_4 = nn.Conv2d(in_channels=encoder_filters[3]*3, out_channels=encoder_filters[3], kernel_size=1)
        self.conv1d_5 = nn.Conv2d(in_channels=encoder_filters[4]*3, out_channels=encoder_filters[4], kernel_size=1)
        self.conv1d_6 = nn.Conv2d(in_channels=decoder_filters[-1]*3, out_channels=decoder_filters[-1], kernel_size=1)
        self.conv1d_7 = nn.Conv2d(in_channels=decoder_filters[-2]*3, out_channels=decoder_filters[-2], kernel_size=1)
        self.conv1d_8 = nn.Conv2d(in_channels=decoder_filters[-3]*3, out_channels=decoder_filters[-3], kernel_size=1)
        self.conv1d_9 = nn.Conv2d(in_channels=decoder_filters[-4]*3, out_channels=decoder_filters[-4], kernel_size=1)

        self.res = nn.Conv2d(decoder_filters[-5] * 3, 5, 1, stride=1, padding=0)
        

        self._initialize_weights()

        encoder = resnet34(pretrained=pretrained)
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

        return enc1, enc2, enc3, enc4, enc5, dec6, dec7, dec8, dec9, dec10


    def forward2(self, enc1_1, enc2_1, enc3_1, enc4_1, enc5_1, dec6_1, dec7_1, dec8_1, dec9_1, dec10_1,enc1_2, enc2_2, enc3_2, enc4_2, enc5_2, dec6_2, dec7_2, dec8_2, dec9_2, dec10_2):
        enc1 = torch.cat([enc1_1, enc1_2], 1)
        enc1 = self.conv1d_1(enc1)
        enc2 = self.conv2(enc1)
        enc2 = torch.cat([enc2, enc2_1, enc2_2], 1)
        enc2 = self.conv1d_2(enc2)
        enc3 = self.conv3(enc2)
        enc3 = torch.cat([enc3, enc3_1, enc3_2], 1)
        enc3 = self.conv1d_3(enc3)
        enc4 = self.conv4(enc3)
        enc4 = torch.cat([enc4, enc4_1, enc4_2], 1)
        enc4 = self.conv1d_4(enc4)
        enc5 = self.conv5(enc4)
        enc5 = torch.cat([enc5, enc5_1, enc5_2], 1)
        enc5 = self.conv1d_5(enc5)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))
        dec6 = torch.cat([dec6, dec6_1, dec6_2], 1)
        dec6 = self.conv1d_6(dec6)

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        dec7 = torch.cat([dec7, dec7_1, dec7_2], 1)
        dec7 = self.conv1d_7(dec7)
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))
        dec8 = torch.cat([dec8, dec8_1, dec8_2], 1)
        dec8 = self.conv1d_8(dec8)

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))
        dec9 = torch.cat([dec9, dec9_1, dec9_2], 1)
        dec9 = self.conv1d_9(dec9)

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        dec10 = torch.cat([dec10, dec10_1, dec10_2], 1)
        return dec10


    def forward(self, x):
        enc1_1, enc2_1, enc3_1, enc4_1, enc5_1, dec6_1, dec7_1, dec8_1, dec9_1, dec10_1 = self.forward1(x[:, :3, :, :])
        enc1_2, enc2_2, enc3_2, enc4_2, enc5_2, dec6_2, dec7_2, dec8_2, dec9_2, dec10_2 = self.forward1(x[:, 3:, :, :])
        dec10 = self.forward2(enc1_1, enc2_1, enc3_1, enc4_1, enc5_1, dec6_1, dec7_1, dec8_1, dec9_1, dec10_1,enc1_2, enc2_2, enc3_2, enc4_2, enc5_2, dec6_2, dec7_2, dec8_2, dec9_2, dec10_2)
        # dec10 = torch.cat([dec10_1, dec10_2], 1)
        return self.res(dec10)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ChannelAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(in_channels*2, out_channels, kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU()
# changed tanh to relu for two_transformer exp
    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        x = self.conv1(x)
        return self.relu(x)

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
        decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.encoder_filters = [64, 64, 128, 256, 512]
        self.decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = resnet34(pretrained=pretrained)
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


        in_dim = 1024
        mlp_dim = 2*in_dim
        enc_depth = 2
        dim_head = 64
        self.transformer = Transformer(dim=in_dim, depth=2, heads=8,
                                       dim_head=dim_head,
                                       mlp_dim=mlp_dim, dropout=0)

        


    def forward(self, x):

        # Encoder 1
        x_1 = x[:, :3, :, :]
        enc1_1 = self.conv1(x_1)
        enc2_1 = self.conv2(enc1_1)
        enc3_1 = self.conv3(enc2_1)
        enc4_1 = self.conv4(enc3_1)
        enc5_1 = self.conv5(enc4_1)

        # Encoder 2
        x_2 = x[:, 3:, :, :]
        enc1_2 = self.conv1(x_2)
        enc2_2 = self.conv2(enc1_2)
        enc3_2 = self.conv3(enc2_2)
        enc4_2 = self.conv4(enc3_2)
        enc5_2 = self.conv5(enc4_2)

        # Bottleneck
        # enc5_1 = (self.ca_bottle_max(enc5_1)*enc5_1)
        # enc5_2 = (self.ca_bottle_max(enc5_2)*enc5_2)

        # try dot product as attention??
        # enc5 = torch.einsum('bcij,bcij->bcij' , enc5_1, enc5_2)
        # enc5 = self.ca_bottle_avg_min(enc5)*enc5

        enc5 = self.ca_skip_5(enc5_1,enc5_2)

        # B_, C_, H_, W_ = enc5.shape
        # enc5_i = enc5.view([B_, C_, H_*W_])
        # enc5_i = self.transformer(enc5_i)
        # # print(enc5.shape)
        # enc5_i = enc5_i.view([B_, C_, H_, W_])
        # enc5 = self.ca_skip_5(enc5_i,enc5)

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

        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
'''

class UNet_Change_Transformer_BiT(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(UNet_Change_Transformer_BiT, self).__init__()
        
        print("using UNet_Change_Transformer !!!!")

        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = np.asarray([48, 64, 96, 128, 320])

        self.encoder_filters = [64, 64, 128, 256, 512]
        self.decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2] , decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3] , decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4] , decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5] , decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], 5, 1, stride=1, padding=0)
       

        self.conv5_2 = nn.Conv2d(1024, 512, kernel_size=5, stride=1, padding=2)
        self._initialize_weights()

        encoder = resnet34(pretrained=pretrained)
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
        self.sigmoid = nn.Sigmoid()

        dim = 64
        dim2 = 256
        dim3 = 1024
        heads = 4
        # self.dec_depth = 3
        self.enc_depth = 8
        decoder_softmax = True
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=heads,
                                       dim_head=dim, mlp_dim=dim**2, dropout=0.05)
        # self.transformer2 = Transformer(dim=dim2, depth=self.enc_depth, heads=heads,
        #                                dim_head=dim2, mlp_dim=dim2, dropout=0.05)
        # self.transformer3 = Transformer(dim=dim3, depth=self.enc_depth, heads=heads//2,
        #                                dim_head=dim3, mlp_dim=dim3, dropout=0.05)
        # self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
        #                     heads=heads, dim_head=dim, mlp_dim=dim**2, dropout=0,
        #                                               softmax=decoder_softmax)
        # self.transformer_decoder2 = TransformerDecoder(dim=dim2, depth=self.dec_depth,
        #                     heads=heads, dim_head=dim2, mlp_dim=dim2, dropout=0,
        #                                               softmax=decoder_softmax)
        # self.transformer_decoder3 = TransformerDecoder(dim=dim3, depth=self.dec_depth,
        #                     heads=heads//2, dim_head=dim3, mlp_dim=dim3//2, dropout=0,
        #                                               softmax=decoder_softmax)


    def forward(self, x):
        # Encoder 1
        x_1 = x[:, :3, :, :]
        enc1_1 = self.conv1(x_1)
        enc2_1 = self.conv2(enc1_1)
        enc3_1 = self.conv3(enc2_1)
        enc4_1 = self.conv4(enc3_1)
        enc5_1 = self.conv5(enc4_1)

        # Encoder 2
        x_2 = x[:, 3:, :, :]
        enc1_2 = self.conv1(x_2)
        enc2_2 = self.conv2(enc1_2)
        enc3_2 = self.conv3(enc2_2)
        enc4_2 = self.conv4(enc3_2)
        enc5_2 = self.conv5(enc4_2)

        # # Bottleneck
        enc5_1 = (self.ca_bottle_max(enc5_1)*enc5_1)
        enc5_2 = (self.ca_bottle_max(enc5_2)*enc5_2)

        enc5 = self.ca_skip_5(enc5_1,enc5_2)
        B_, C_, H_, W_ = enc5.shape
        enc5_i = enc5.view([B_, C_, H_*W_])
        enc5_i = self.transformer(enc5_i)
        enc5_i = enc5_i.view([B_, C_, H_, W_])
        enc5 = self.ca_skip_5(enc5_i,enc5)

        enc4 = self.ca_skip_4(enc4_1, enc4_2)
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))
        # dec6 = self.conv6_2(torch.cat([dec6, enc4_1, enc4_2], 1))

        enc3 = self.ca_skip_3(enc3_1, enc3_2)
        # B_, C_, H_, W_ = enc3.shape
        # enc3_i = enc3.view([B_, C_, H_*W_])
        # enc3_i = self.transformer3(enc3_i)
        # # enc4_i = enc4_i.view([B_, C_, H_, W_])
        # # enc4 = self.ca_skip_4(enc4_i,enc4)
        # enc3 = enc3.view([B_, C_, H_*W_])
        # enc3 = self.transformer_decoder3(enc3_i, enc3)
        # enc3 = enc3.view([B_, C_, H_, W_])

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))
        
        enc2 = self.ca_skip_2(enc2_1, enc2_2)
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))

        enc1 = self.ca_skip_2(enc1_1, enc1_2)
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        out = self.res(dec10)
        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        print("using BiT Transformer !!!!") 

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

    def forward(self, x):
        # forward backbone resnet
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
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

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
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

        encoder = resnet34(pretrained=pretrained)
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
        mlp_dim = 2*dim
        enc_depth = 2
        dim_head = 64
        decoder_dim_head = 64
        decoder_softmax = True
        self.transformer = Transformer(dim=dim, depth=6, heads=4,
                                       dim_head=dim_head,
                                       mlp_dim=mlp_dim, dropout=0)

        self.transformer2 = Transformer(dim=dim2, depth=4, heads=2, # depth=6, heads=4 leads to GPU outage
                                       dim_head=dim_head,
                                       mlp_dim=dim2, dropout=0)
 
        self.classifier = TwoLayerConv2d(in_channels=512, out_channels=2)
        self.convT = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # Encoder 1
        x_1 = x[:, :3, :, :]
        enc1_1 = self.conv1(x_1)
        enc2_1 = self.conv2(enc1_1)
        enc3_1 = self.conv3(enc2_1)
        enc4_1 = self.conv4(enc3_1)
        enc5_1 = self.conv5(enc4_1)

        # Encoder 2
        x_2 = x[:, 3:, :, :]
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

        # enc5_diff = (enc5_1 - enc5_2)
        # spatial_attention = enc5_diff.view([B_, C_, H_*W_]).contiguous()
        # spatial_attention = torch.softmax(spatial_attention, dim=-1)

        # enc5_i = torch.einsum('bln,bln->bln', spatial_attention, enc5_i)
        enc5_t = self.transformer(enc5_i)
        enc5_t = enc5_t.view([B_, C_, H_, W_]).contiguous()
        enc5 = self.ca_skip_5(enc5_t,enc5)

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

        ## run4: depth=2, heads=2
        # enc3 = self.ca_skip_3(enc3_1, enc3_2)
        # B_, C_, H_, W_ = enc3.shape
        # enc3_i = enc3.view([B_, C_, H_*W_]).contiguous()

        # # enc3_diff = (enc3_1 - enc3_2)
        # # spatial_attention = enc3_diff.view([B_, C_, H_*W_]).contiguous()
        # # spatial_attention = torch.softmax(spatial_attention, dim=-1)

        # # enc3 = torch.einsum('bln,bln->bln', spatial_attention, enc3)
        # enc3_t = self.transformer2(enc3_i)
        # enc3_t = enc3_t.view([B_, C_, H_, W_]).contiguous()
        # enc3 = self.ca_skip_3(enc3, enc3_t)

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3_1, enc3_2], 1))
        
        ## run0: depth=2, heads=1
        enc2 = self.ca_skip_2(enc2_1, enc2_2)
        B_, C_, H_, W_ = enc2.shape
        enc2_i = enc2.view([B_, C_, H_*W_]).contiguous()

        # enc2_diff = (enc2_1 - enc2_2)
        # spatial_attention = enc2_diff.view([B_, C_, H_*W_]).contiguous()
        # spatial_attention = torch.softmax(spatial_attention, dim=-1)

        # enc2_i = torch.einsum('bln,bln->bln', spatial_attention, enc2_i)
        enc2_t = self.transformer2(enc2_i)
        enc2_t = enc2_t.view([B_, C_, H_,W_]).contiguous()
        enc2 = self.ca_skip_2(enc2_t, enc2)

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



class ResNet_Encoder(torch.nn.Module):
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
        x_64 = self.resnet.maxpool(x)

        x_64_2 = self.resnet.layer1(x_64) # 1/4, in=64, out=64
        x_128 = self.resnet.layer2(x_64_2) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_256 = self.resnet.layer3(x_128) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_512 = self.resnet.layer4(x_256) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_512)
        else:
            x = x_512
        # output layers
        x = self.conv_pred(x)
        print("in forward singgleee")
        return x, x_64, x_64_2, x_128, x_256, x_512


class BASE_UNet_Transformer(ResNet):
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

    def forward(self, x):
        # forward backbone resnet
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        x1, x1_64, x1_64_2, x1_128, x1_256, x1_512 = self.forward_single(x1)
        x2, x2_64, x2_64_2, x2_128, x2_256, x2_512 = self.forward_single(x2)
        print(x1.shape, x1_64.shape, x1_64_2.shape, x1_128.shape, x1_256.shape, x1_512.shape)

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
