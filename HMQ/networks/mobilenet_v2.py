"""
 This file is copied from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
 and modified for this project needs.

 The Licence of the torch vision project is shown in:https://github.com/pytorch/vision/blob/master/LICENSE
"""

from torch import nn
from torchvision.models.utils import load_state_dict_from_url
from networks import layers
from networks.blocks import InvertedResidual, ConvBNNonLinear

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):
    def __init__(self, nc, num_classes=1000, width_mult=1.0, inverted_residual_setting=None,
                 round_nearest=8):
        """
        The init function of the  MobileNet V2 Module

        :param nc: Network controller
        :param num_classes: the number of output classes
        :param width_mult: The width multiple
        :param inverted_residual_setting: A list of the block configurations
        :param round_nearest: Rounding to nearest value
        """

        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNNonLinear(nc, 3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(nc, input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNNonLinear(nc, input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            layers.FullyConnected(nc, self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mobilenet_v2(nc, pretrained=False, progress=True, **kwargs):
    """
    An implementation of Mobile-Net-V2 model from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_
    :param nc: Network Controller class
    :param pretrained: Load pretrained weights
    :param progress: Show download progress bar
    :param kwargs: A kwargs dict for controlling module config
    :return: A PyTorch module
    """
    model = MobileNetV2(nc, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)

        def rename(k):
            k = k.replace('.0.0.', '.0.0.conv.').replace('.0.1.', '.0.0.bn.').replace('.1.conv.1',
                                                                                      '.1.conv.1.0.conv').replace(
                '.1.conv.2', '.1.conv.1.0.bn')
            k = k.replace('.conv.1.0', '.conv.1.0.conv').replace('.conv.1.1', '.conv.1.0.bn').replace('.conv.2',
                                                                                                      '.conv.2.0.conv').replace(
                '.conv.3', '.conv.2.0.bn')
            k = k.replace('features.18.0', 'features.18.0.conv').replace('features.18.1', 'features.18.0.bn')
            k = k.replace('classifier.1.weight', 'classifier.1.fc.weight').replace('classifier.1.bias',
                                                                                   'classifier.1.fc.bias')
            k = k.replace('.conv.1.0.conv.bn', '.conv.1.0.bn').replace('.conv.1.0.conv.conv', '.conv.1.0.conv')
            return k

        state_dict = {rename(k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    return model
