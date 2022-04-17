# ------------------------------------------------------------------------------
# This code is base on
# HRNet-Semantic-Segmentation (https://github.com/HRNet/HRNet-Semantic-Segmentation)
# Modified code based Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import gc

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          BatchNorm2d(reduction_dim, momentum=BN_MOMENTUM),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                BatchNorm2d(reduction_dim, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            BatchNorm2d(reduction_dim, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear')
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, n_classes=None):
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        self.transitions = nn.ModuleList()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transitions.append(self._make_transition_layer(
            [stage1_out_channel], num_channels))
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transitions.append(self._make_transition_layer(
            pre_stage_channels, num_channels))
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transitions.append(self._make_transition_layer(
            pre_stage_channels, num_channels))
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.sum_stage4_channels = np.int(np.sum(pre_stage_channels))

        if n_classes is None:
            n_classes = config.DATASET.NUM_CLASSES
        self.n_classes = n_classes
        self.final_conv_kernel = extra.FINAL_CONV_KERNEL

    def mask_last_layer(self, last_inp_channels=None):
        if last_inp_channels is None:
            last_inp_channels = self.sum_stage4_channels
        self.last_inp_channels = last_inp_channels

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=self.n_classes,
                kernel_size=self.final_conv_kernel,
                stride=1,
                padding=1 if self.final_conv_kernel == 3 else 0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transitions[0][i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transitions[1][i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transitions[2][i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)

        return x

    def init_weights(self, pretrained='', ):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                print(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class DualHRNet(nn.Module):

    def __init__(self, config, **kwargs):
        super(DualHRNet, self).__init__()

        self.is_split_loss = config.MODEL.IS_SPLIT_LOSS
        if self.is_split_loss:
            self.loc_net = HighResolutionNet(config, n_classes=2)
            self.cls_net = HighResolutionNet(config, n_classes=4)
        else:
            self.loc_net = HighResolutionNet(config, n_classes=config.DATASET.NUM_CLASSES)
            self.cls_net = HighResolutionNet(config, n_classes=None)

        self.fuse_loc_stage1 = self._make_fuse_layer([n_ch * 2 for n_ch in self.loc_net.stage2_cfg['NUM_CHANNELS']],
                                                     self.loc_net.stage2_cfg['NUM_CHANNELS'],
                                                     config)
        self.fuse_cls_stage1 = self._make_fuse_layer([n_ch * 2 for n_ch in self.cls_net.stage2_cfg['NUM_CHANNELS']],
                                                     self.cls_net.stage2_cfg['NUM_CHANNELS'],
                                                     config)

        self.fuse_loc_stage2 = self._make_fuse_layer([n_ch * 2 for n_ch in self.loc_net.stage3_cfg['NUM_CHANNELS']],
                                                     self.loc_net.stage3_cfg['NUM_CHANNELS'],
                                                     config)
        self.fuse_cls_stage2 = self._make_fuse_layer([n_ch * 2 for n_ch in self.cls_net.stage3_cfg['NUM_CHANNELS']],
                                                     self.cls_net.stage3_cfg['NUM_CHANNELS'],
                                                     config)

        self.fuse_loc_stage3 = self._make_fuse_layer([n_ch * 2 for n_ch in self.loc_net.stage4_cfg['NUM_CHANNELS']],
                                                     self.loc_net.stage4_cfg['NUM_CHANNELS'],
                                                     config)
        self.fuse_cls_stage3 = self._make_fuse_layer([n_ch * 2 for n_ch in self.cls_net.stage4_cfg['NUM_CHANNELS']],
                                                     self.cls_net.stage4_cfg['NUM_CHANNELS'],
                                                     config)

        self.is_use_fpn = config.MODEL.USE_FPN
        if self.is_use_fpn:
            self.fpn_loc = self._make_fpn_layer(self.loc_net.stage4_cfg['NUM_CHANNELS'])
            self.fpn_cls = self._make_fpn_layer(self.cls_net.stage4_cfg['NUM_CHANNELS'])
        else:
            self.fpn_loc = None
            self.fpn_cls = None

        if self.is_split_loss:
            self.loc_net.mask_last_layer()
            self.cls_net.mask_last_layer()
        else:
            self.fuse_last = self._make_fuse_layer([n_ch * 2 for n_ch in self.loc_net.stage4_cfg['NUM_CHANNELS']],
                                                   self.loc_net.stage4_cfg['NUM_CHANNELS'],
                                                   config)
            self.loc_net.mask_last_layer()

        self.is_disaster_prediction = config.MODEL.IS_DISASTER_PRED
        if self.is_disaster_prediction:
            self.disaster_layer = self._make_disaster_layer(self.cls_net.last_inp_channels)

    @staticmethod
    def _make_fuse_layer(in_channels, out_channels, config):
        num_branches = len(in_channels)

        fuse_layers = []
        for idx in range(num_branches):
            fuse_layers.append(nn.Sequential(
                nn.Conv2d(in_channels[idx], out_channels[idx], kernel_size=config.MODEL.FUSE_CONV_K_SIZE, stride=1,
                          padding=max(0, config.MODEL.FUSE_CONV_K_SIZE - 2), bias=False),
                BatchNorm2d(out_channels[idx], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))

        return nn.ModuleList(fuse_layers)

    @staticmethod
    def _make_disaster_layer(in_channels, n_disaster=6):

        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, n_disaster)
        )

    @staticmethod
    def _make_fpn_layer(in_channels, channels_per_group=8):
        fpn_layers = []
        fpn_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=1,
                          padding=1, bias=True),
                nn.GroupNorm(num_groups=in_channels[0] // channels_per_group,
                             num_channels=in_channels[0]),
                nn.ReLU(),
            )
        )
        for branch in range(1, len(in_channels)):
            fpn_layer = []
            for idx in range(branch, 0, -1):
                in_channel = in_channels[idx]
                out_channel = in_channels[idx - 1]
                fpn_layer.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.GroupNorm(num_groups=out_channel // channels_per_group, num_channels=out_channel),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='bilinear')
                    )
                )
            fpn_layers.append(nn.Sequential(*fpn_layer))

        return nn.ModuleList(fpn_layers)

    @staticmethod
    def _forward_stage1(net, x):
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.conv2(x)
        x = net.relu(x)
        x = net.bn2(x)
        x = net.relu(x)
        x = net.layer1(x)

        return x

    @staticmethod
    def _forward_transition(net, x_list, n_stage):
        n_stage = n_stage - 1
        x_trans_list = []
        for i in range(len(net.transitions[n_stage])):
            if net.transitions[n_stage][i] is not None:
                x_trans_list.append(net.transitions[n_stage][i](x_list[-1]))
            else:
                x_trans_list.append(x_list[i])

        return x_trans_list

    @staticmethod
    def _forward_fuse_layer(x_list, fuse_layers):
        x_fuse_list = []
        for x, fuse_layer in zip(x_list, fuse_layers):
            x_fuse_list.append(fuse_layer(x))

        return x_fuse_list

    @staticmethod
    def _concat_features(x1_list, x2_list):
        x_cat_list = list()
        for x1, x2 in zip(x1_list, x2_list):
            x_cat_list.append(torch.cat((x1, x2), dim=1))

        return x_cat_list

    def _upsampling(self, x_list, fpn_layers):
        x_up_list = list()
        if self.is_use_fpn:
            x = fpn_layers[0](x_list[0])
            for _x, fpn_layer in zip(x_list[1:], fpn_layers[1:]):
                x += fpn_layer(_x)
        else:
            x0_h, x0_w = x_list[0].size(2), x_list[0].size(3)
            x1 = F.interpolate(x_list[1], size=(x0_h, x0_w), mode='bilinear')
            x2 = F.interpolate(x_list[2], size=(x0_h, x0_w), mode='bilinear')
            x3 = F.interpolate(x_list[3], size=(x0_h, x0_w), mode='bilinear')
            x = torch.cat([x_list[0], x1, x2, x3], 1)
        return x

    def forward(self, x):
        x_pre = x[:,:3,:,:]
        x_post = x[:,3:,:,:]
        # Stage 1
        x_pre = self._forward_stage1(self.loc_net, x_pre)
        x_post = self._forward_stage1(self.cls_net, x_post)

        x_pre_list = self._forward_transition(self.loc_net, [x_pre], n_stage=1)
        x_post_list = self._forward_transition(self.cls_net, [x_post], n_stage=1)

        x_cat_list = self._concat_features(x_pre_list, x_post_list)
        x_pre_list = self._forward_fuse_layer(x_cat_list, self.fuse_loc_stage1)
        x_post_list = self._forward_fuse_layer(x_cat_list, self.fuse_cls_stage1)

        # Stage 2
        x_pre_list = self.loc_net.stage2(x_pre_list)
        x_post_list = self.cls_net.stage2(x_post_list)

        x_pre_list = self._forward_transition(self.loc_net, x_pre_list, n_stage=2)
        x_post_list = self._forward_transition(self.cls_net, x_post_list, n_stage=2)

        x_cat_list = self._concat_features(x_pre_list, x_post_list)
        x_pre_list = self._forward_fuse_layer(x_cat_list, self.fuse_loc_stage2)
        x_post_list = self._forward_fuse_layer(x_cat_list, self.fuse_cls_stage2)

        # Stage 3
        x_pre_list = self.loc_net.stage3(x_pre_list)
        x_post_list = self.cls_net.stage3(x_post_list)

        x_pre_list = self._forward_transition(self.loc_net, x_pre_list, n_stage=3)
        x_post_list = self._forward_transition(self.cls_net, x_post_list, n_stage=3)

        x_cat_list = self._concat_features(x_pre_list, x_post_list)
        x_pre_list = self._forward_fuse_layer(x_cat_list, self.fuse_loc_stage3)
        x_post_list = self._forward_fuse_layer(x_cat_list, self.fuse_cls_stage3)

        # Stage 4
        x_pre_list = self.loc_net.stage4(x_pre_list)
        x_post_list = self.cls_net.stage4(x_post_list)

        # if not self.training:
        #     gc.collect()

        if self.is_split_loss:
            # Upsampling
            x_pre = self._upsampling(x_pre_list, self.fpn_loc)
            x_post = self._upsampling(x_post_list, self.fpn_cls)
        else:
            x_cat_list = self._concat_features(x_pre_list, x_post_list)
            x_pre_list = self._forward_fuse_layer(x_cat_list, self.fuse_last)
            x_pre = self._upsampling(x_pre_list, self.fpn_loc)

        # # to avoid shortage of memory due to "docker run --memory=8g"
        # if not self.training:
        #     torch.save(x_post, 'x_post.ph')
        #     del x_pre_list
        #     del x_post_list
        #     del x_post
        #     gc.collect()

        # Last layer
        loc = self.loc_net.last_layer(x_pre)

        if self.is_split_loss:
            # if not self.training:
            #     del x_pre
            #     gc.collect()
            #     x_post = torch.load('x_post.ph')

            cls = self.cls_net.last_layer(x_post)
        else:
            cls = None

        pred_dict = {'loc': loc, 'cls': cls}

        # # Optional: prediction for disaster_type
        # if self.is_disaster_prediction:
        #     disaster = self.disaster_layer(x_post)
        #     pred_dict['disaster'] = disaster

        return pred_dict


def get_model(config=None, **kwargs):
    model = DualHRNet(config, **kwargs)
    return model