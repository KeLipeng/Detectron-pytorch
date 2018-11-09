import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron.utils.torchsummary import summary

from detectron.core.config import cfg
import detectron.nn as mynn
import detectron.utils.net as net_utils
from detectron.utils.resnet_weights_helper import convert_state_dict

# ---------------------------------------------------------------------------- #
# VGG16 and VGG19 network
# by Liu Liu
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (VGG16, VGG19)
# ---------------------------------------------------------------------------- #

def VGG11_conv5_body():
    return VGG_conv5_body((1,1,2,2,2))

def VGG13_conv5_body():
    return VGG_conv5_body((2,2,2,2,2))

def VGG16_conv5_body():
    return VGG_conv5_body((2,2,3,3,3))

def VGG19_conv5_body():
    return VGG_conv5_body((2,2,4,4,4))

# ---------------------------------------------------------------------------- #
# Generic VGG components
# ---------------------------------------------------------------------------- #

cfg = {
    'vgg11': [1, 1, 2, 2, 2],
    'vgg13': [2, 2, 2, 2, 2],
    'vgg16': [2, 2, 3, 3, 3],
    'vgg19': [2, 2, 4, 4, 4],
}


class VGG_conv5_body(nn.Module):
    def __init__(self, block_counts):
        super(VGG_conv5_body, self).__init__()
        dim_in = 3
        self.conv1, dim_in = self._make_block(block_counts[0], dim_in, 64)
        self.conv2, dim_in = self._make_block(block_counts[1], dim_in, 128)
        self.conv3, dim_in = self._make_block(block_counts[2], dim_in, 256)
        self.conv4, dim_in = self._make_block(block_counts[3], dim_in, 512)
        self.conv5, dim_in = self._make_block(block_counts[4], dim_in, 512)

        self.dim_out = dim_in
        self.spatial_scale = 1. / 32

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def _make_block(self, n, dim_in, dim_out):
        layers = []
        for i in range(n):
            if i == 0:
                layers += [nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
                           nn.BatchNorm2d(dim_out),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
                           nn.BatchNorm2d(dim_out),
                           nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers), dim_out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG16_conv5_body().cuda()
    print(net)
    summary(net,(3,32,32))
    x = torch.randn(2,3,32,32).cuda()
    y = net(x)
    print(y.size())
test()






