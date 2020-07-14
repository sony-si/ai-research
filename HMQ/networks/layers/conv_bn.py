from torch import nn
from torch.nn import functional as F
from networks.controller.network_controller import NetworkQuantizationController
from networks.layers.non_linear import Identity
from networks.layers.quantization import Quantization


class ConvBN(nn.Module):
    def __init__(self, network_controller: NetworkQuantizationController, in_channels, out_channels, kernel_size,
                 stride=1,
                 padding=0, dilation=1,
                 group=1, disable_bn=False,
                 batch_norm_epsilon=1e-5, batch_norm_momentum=0.1, tf_padding=False):
        """
        A joint 2d convolution with batch normalization module with HMQ quantization of the convolution weights.
        :param network_controller: The network quantization controller
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        :param kernel_size: The kernel size
        :param stride: The convolution stride
        :param padding: The convolution padding
        :param dilation: The convolution dilation
        :param group: The convolution group size
        :param disable_bn: Disable the batch normalization
        :param batch_norm_epsilon: The batch normalization epsilon
        :param batch_norm_momentum: The batch normalization momentum
        :param tf_padding: Use TensorFlow padding (for EfficientNet)
        """
        super(ConvBN, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.network_controller = network_controller
        self.tf_padding = tf_padding

        if not tf_padding:
            self.pad_tensor = Identity()
            self.padding_conv = self.padding

        else:
            padding = 0
            self.padding_conv = padding
            pad_h = self.padding
            pad_w = self.padding
            self.pad_tensor = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group,
                              bias=disable_bn)
        if disable_bn:
            self.bn = Identity()
        else:
            self.bn = nn.BatchNorm2d(out_channels, eps=batch_norm_epsilon, momentum=batch_norm_momentum)
        self.q = Quantization(network_controller, is_signed=True,
                              weights_values=self.conv.weight.detach())

    def forward(self, x):
        """
        The forward function of the ConvBN module

        :param x: Input tensor x to be convolved
        :return: A tensor after convolution and batch normalization
        """
        x = self.pad_tensor(x)
        if self.network_controller.is_float_coefficient:
            return self.bn(self.conv(x))
        else:
            res = F.conv2d(x, self.q(self.conv.weight), self.conv.bias, self.stride,
                           self.padding_conv, self.dilation, self.group)
            return self.bn(res)
