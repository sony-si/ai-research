"""
 This file is copied from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
 and modified for this project needs.

 The Licence of the torch vision project is shown in:https://github.com/pytorch/vision/blob/master/LICENSE
"""
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from networks.layers.non_linear import NonLinear
from networks.layers.conv_bn import ConvBN


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3_bn(nc, in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding and Batch normalization"""
    return ConvBN(nc, in_planes, out_planes, kernel_size=3,
                  stride=stride,
                  padding=dilation, dilation=dilation,
                  group=groups)


def conv1x1_bn(nc, in_planes, out_planes, stride=1):
    """1x1 convolution with padding and Batch normalization"""
    return ConvBN(nc, in_planes, out_planes, kernel_size=1,
                  stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, nc, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1_bn = conv3x3_bn(nc, inplanes, planes, stride)
        self.relu1 = NonLinear(nc, planes)

        self.conv2_bn = conv3x3_bn(nc, planes, planes)
        self.relu2 = NonLinear(nc, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_bn(x)
        out = self.relu1(out)

        out = self.conv2_bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, nc, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1_bn = conv1x1_bn(nc, inplanes, width)
        self.relu1 = NonLinear(nc, width)

        self.conv2_bn = conv3x3_bn(nc, width, width, stride, groups, dilation)
        self.relu2 = NonLinear(nc, width)

        self.conv3_bn = conv1x1_bn(nc, width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = NonLinear(nc, width)

        # self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_bn(x)
        out = self.relu1(out)

        out = self.conv2_bn(out)
        out = self.relu2(out)

        out = self.conv3_bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, nc, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResNet, self).__init__()

        self.nc = nc

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1_bn = ConvBN(nc, 3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.relu = NonLinear(nc, self.inplanes)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1_bn(self.nc, self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.nc, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.nc, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, nc, block, layers, pretrained, progress, **kwargs):
    model = ResNet(nc, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)

        def rename(x):
            return x.replace('conv1', 'conv1_bn.conv').replace('conv3', 'conv3_bn.conv').replace('conv2',
                                                                                                 'conv2_bn.conv').replace(
                'bn1',
                'conv1_bn.bn').replace(
                'bn2', 'conv2_bn.bn').replace(
                'bn3', 'conv3_bn.bn').replace('downsample.0', 'downsample.conv').replace('downsample.1',
                                                                                         'downsample.bn')

        state_dict = {rename(k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    return model


def resnet50(nc, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', nc, Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
