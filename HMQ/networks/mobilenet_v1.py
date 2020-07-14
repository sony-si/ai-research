import torch
import torch.nn as nn
from networks import layers


class MobileNetV1ImageNet(nn.Module):
    def __init__(self, nc, n_classes=1000, p_drop_out=0.2):
        """
        The init function of the MobileNet-V1 Module

        :param nc: Network controller
        :param n_classes: the number of output classes
        :param p_drop_out: The dropout probability
        """
        super(MobileNetV1ImageNet, self).__init__()

        def conv_bn(channels_in, channels_out, stride):
            conv_block = nn.Sequential()

            conv_block.add_module('conv2d',
                                  layers.ConvBN(nc, in_channels=channels_in, out_channels=channels_out, kernel_size=3,
                                                stride=stride,
                                                padding=1))
            conv_block.add_module('relu', layers.NonLinear(nc, channels_out))
            return conv_block

        def conv_dw(channels_in, channels_out, stride):
            conv_block = nn.Sequential()
            conv_block.add_module('dw_conv',
                                  layers.ConvBN(nc, in_channels=channels_in, out_channels=channels_in, kernel_size=3,
                                                stride=stride,
                                                padding=1,
                                                group=channels_in))

            conv_block.add_module('relu_0', layers.NonLinear(nc, channels_in))
            conv_block.add_module('conv',
                                  layers.ConvBN(nc, in_channels=channels_in, out_channels=channels_out, kernel_size=1,
                                                stride=1,
                                                padding=0))

            conv_block.add_module('relu_1', layers.NonLinear(nc, channels_out))
            return conv_block

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )

        self.dropout = nn.Dropout(p=p_drop_out)
        self.fc = layers.FullyConnected(nc, in_channels=1024, out_channels=n_classes)

        # manual initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.fc.fc.weight)
        nn.init.constant_(self.fc.fc.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def mobilenet_v1(nc, pretrained=False, progress=True, **kwargs):
    """
    An implementation of Mobile-Net-V1 model from
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/pdf/1704.04861.pdf>`_
    :param nc: Network Controller class
    :param pretrained: Load pretrained weights
    :param progress: Show download progress bar
    :param kwargs: A kwargs dict for controlling module config
    :return: A PyTorch module
    """

    net = MobileNetV1ImageNet(nc, **kwargs)
    if pretrained:
        state_dict = torch.load(
            './models/mobilenet_v1_imagenet.pth',
            map_location='cpu')
        state_dict = state_dict.get('state_dict')

        def rename(x):
            return x.replace('module.', '').replace('conv.', 'conv.conv.').replace('conv2d.', 'conv2d.conv.').replace(
                'bn.', 'conv2d.bn.').replace('bn_0.', 'dw_conv.bn.').replace('bn_1.', 'conv.bn.').replace('fc.',
                                                                                                          'fc.fc.')

        state_dict = {rename(k): v for k, v in state_dict.items()}

        net.load_state_dict(state_dict)
    return net
