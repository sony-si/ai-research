import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.uniform import Uniform
from networks.layers.non_linear import NonLinear, NonLinearType
from networks.layers.conv_bn import ConvBN


class DropConnect(nn.Module):
    def __init__(self, survival_prob):
        """
        A module that implements drop connection
        :param survival_prob: the probability if connection survival
        """
        super(DropConnect, self).__init__()
        self.survival_prob = survival_prob
        self.u = Uniform(0, 1)

    def forward(self, x):
        """
        The forward function of the DropConnect module

        :param x: Input tensor x
        :return: A tensor after drop connection
        """
        if self.training:
            random_tensor = self.u.sample([x.shape[0], 1, 1, 1]).cuda()
            random_tensor += self.survival_prob
            binary_tensor = torch.floor(random_tensor)
            return x * binary_tensor / self.survival_prob
        else:
            return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """
        Global Average pooling module
        """
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        """
        The forward function of the GlobalAvgPool2d module

        :param x: Input tensor x
        :return: A tensor after average pooling
        """
        return F.avg_pool2d(x, (x.shape[2], x.shape[3]))


class SEBlock(nn.Module):
    def __init__(self, nc, in_channels, reduce_channels):
        """
        An implantation of Squeeze Excite block

        :param nc: Input network controller
        :param in_channels: the number of input channels
        :param reduce_channels: the number of channels after reduction
        """
        super(SEBlock, self).__init__()
        self.gap = GlobalAvgPool2d()
        self.conv_reduce = nn.Sequential(
            ConvBN(nc, in_channels, reduce_channels, 1, disable_bn=True),
            NonLinear(nc, reduce_channels, NonLinearType.SWISH))
        self.conv_expand = nn.Sequential(
            ConvBN(nc, reduce_channels, in_channels, 1, disable_bn=True),
            NonLinear(nc, in_channels, NonLinearType.SIGMOID))

    def forward(self, x):
        """
        The forward function of the SEBlock module
        :param x: Input tensor x
        :return: A tensor after SE Block
        """
        return x * self.conv_expand(self.conv_reduce(self.gap(x)))


class ConvBNNonLinear(nn.Sequential):
    def __init__(self, nc, in_planes, out_planes, kernel_size=3, stride=1, groups=1, nl_type=NonLinearType.RELU6,
                 batch_norm_epsilon=1e-5, batch_norm_momentum=0.1, tf_padding=False):
        """
        A joint block of 2d convolution with batch normalization and non linear function modules
        with HMQ quantization of both the convolution weights and activation function
        :param nc: The network quantization controller
        :param in_planes: The number of input channels
        :param out_planes: The number of output channels
        :param kernel_size: The kernel size
        :param stride: The convolution stride
        :param groups: The convolution group size
        :param nl_type: enum that state the non-linear type.
        :param batch_norm_epsilon: The batch normalization epsilon
        :param batch_norm_momentum: The batch normalization momentum
        :param tf_padding: Use TensorFlow padding (for EfficientNet)
        """
        padding = kernel_size - stride if tf_padding else (kernel_size - 1) // 2
        super(ConvBNNonLinear, self).__init__(
            ConvBN(nc, in_planes, out_planes, kernel_size, stride, padding, group=groups,
                   batch_norm_epsilon=batch_norm_epsilon, batch_norm_momentum=batch_norm_momentum,
                   tf_padding=tf_padding),
            NonLinear(nc, out_planes, nl_type)
        )


class InvertedResidual(nn.Module):
    def __init__(self, nc, inp, oup, stride, expand_ratio, kernel_size=3, nl_type=NonLinearType.RELU6, se_ratio=0,
                 survival_prob=0, batch_norm_epsilon=1e-5, batch_norm_momentum=0.1, tf_padding=False):
        """
        A Inverted Residual block use in Efficient-Net

        :param nc: The network quantization controller
        :param inp: The number of input channels
        :param oup: The number of output channels
        :param stride: The depth wise convolution stride
        :param expand_ratio: The block expand ratio for depth-wise convolution
        :param kernel_size: The kernel size
        :param nl_type:  enum that state the non-linear type.
        :param se_ratio: the ratio between the number of input channel and mid channels in SE Bloock
        :param survival_prob: the probability if connection survival
        :param batch_norm_epsilon: The batch normalization epsilon
        :param batch_norm_momentum: The batch normalization momentum
        :param tf_padding: Use TensorFlow padding (for EfficientNet)
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = kernel_size

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNNonLinear(nc, inp, hidden_dim, kernel_size=1, nl_type=nl_type,
                                          batch_norm_epsilon=batch_norm_epsilon,
                                          batch_norm_momentum=batch_norm_momentum))
        layers.append(
            ConvBNNonLinear(nc, hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim,
                            nl_type=nl_type, batch_norm_epsilon=batch_norm_epsilon,
                            batch_norm_momentum=batch_norm_momentum, tf_padding=tf_padding))
        if se_ratio != 0:
            layers.append(SEBlock(nc, hidden_dim, int(inp * se_ratio)))

        layers.append(ConvBNNonLinear(nc, hidden_dim, oup, kernel_size=1, stride=1, nl_type=NonLinearType.IDENTITY,
                                      batch_norm_epsilon=batch_norm_epsilon,
                                      batch_norm_momentum=batch_norm_momentum))

        if survival_prob != 0 and self.use_res_connect:
            layers.append(DropConnect(survival_prob))
        self.conv = nn.Sequential(*layers)
        self.output_q = NonLinear(nc, oup, nl_type=NonLinearType.IDENTITY)

    def forward(self, x):
        """
        The forward function of the InvertedResidual module
        :param x: Input tensor x
        :return: A tensor after InvertedResidual
        """
        if self.use_res_connect:
            y = self.conv(x)
            return self.output_q(x + y)
        else:
            x = self.conv(x)
            return self.output_q(x)


class RepeatedInvertedResidual(nn.Module):
    def __init__(self, nc, n_repeat, in_channels, out_channels, stride_first, expand_ratio, kernel_size=3,
                 nl_type=NonLinearType.RELU6,
                 se_ratio=0,
                 survival_prob_start=0, drop_rate=0, batch_norm_epsilon=1e-5, batch_norm_momentum=0.1,
                 tf_padding=False):
        """
        A block the repeatedly run the InvertedResidual block
        :param nc:The network quantization controller
        :param n_repeat:
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        :param stride_first: The depth wise convolution stride in the first block
        :param expand_ratio: The block expand ratio for depth-wise convolution
        :param kernel_size: The kernel size
        :param nl_type:  enum that state the non-linear type.
        :param se_ratio: the ratio between the number of input channel and mid channels in SE Bloock
        :param survival_prob_start: the probability if connection survival in the first block
        :param batch_norm_epsilon: The batch normalization epsilon
        :param batch_norm_momentum: The batch normalization momentum
        :param tf_padding: Use TensorFlow padding (for EfficientNet)
        """
        super(RepeatedInvertedResidual, self).__init__()
        layers = []
        for i in range(n_repeat):
            if survival_prob_start > 0 and drop_rate > 0:
                survival_prob = survival_prob_start - drop_rate * float(i)
            else:
                survival_prob = 0
            block = InvertedResidual(nc, in_channels if i == 0 else out_channels, out_channels,
                                     stride_first if i == 0 else 1, expand_ratio, kernel_size=kernel_size,
                                     nl_type=nl_type, se_ratio=se_ratio, survival_prob=survival_prob,
                                     batch_norm_epsilon=batch_norm_epsilon, batch_norm_momentum=batch_norm_momentum,
                                     tf_padding=tf_padding)
            layers.append(block)
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward function of the RepeatedInvertedResidual module
        :param x: Input tensor x
        :return: A tensor after RepeatedInvertedResidual
        """
        return self.blocks(x)
