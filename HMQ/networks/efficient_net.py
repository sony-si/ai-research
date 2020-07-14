import torch
from torch import nn
from dataclasses import dataclass
from networks.blocks import ConvBNNonLinear, RepeatedInvertedResidual
from networks.blocks import GlobalAvgPool2d
from networks import layers
from networks.layers.non_linear import NonLinearType


@dataclass
class StageConfig(object):
    stem: bool  # Is stem block
    output_channels: int  # Number of output channels
    expand_ratio: int  # Block expand ratio
    stride_first: int  # The first block stride
    kernel_size: int  # The depth-wise kernel size
    n_repeat: int  # The number of repletion of the block


DEFAULT_CONFIG = [
    StageConfig(stem=True, output_channels=32, expand_ratio=-1, stride_first=2, kernel_size=3, n_repeat=-1),  # 112
    StageConfig(stem=False, output_channels=16, expand_ratio=1, stride_first=1, kernel_size=3, n_repeat=1),
    StageConfig(stem=False, output_channels=24, expand_ratio=6, stride_first=2, kernel_size=3, n_repeat=2),  # 56
    StageConfig(stem=False, output_channels=40, expand_ratio=6, stride_first=2, kernel_size=5, n_repeat=2),  # 28
    StageConfig(stem=False, output_channels=80, expand_ratio=6, stride_first=2, kernel_size=3, n_repeat=3),  # 14
    StageConfig(stem=False, output_channels=112, expand_ratio=6, stride_first=1, kernel_size=5, n_repeat=3),
    StageConfig(stem=False, output_channels=192, expand_ratio=6, stride_first=2, kernel_size=5, n_repeat=4),  # 7
    StageConfig(stem=False, output_channels=320, expand_ratio=6, stride_first=1, kernel_size=3, n_repeat=1)]


class EfficientNet(nn.Module):
    def __init__(self, nc, input_channels=3, n_classes=1000, stage_config=DEFAULT_CONFIG, survival_prob=0.8,
                 se_ratio=0.25,
                 p_drop_out=0.2,
                 nl_type=NonLinearType.SWISH, batch_norm_epsilon=1e-3, batch_norm_momentum=0.01):
        """
        The init function of the EfficientNet Module

        :param nc: Network controller
        :param input_channels: Input number channels
        :param n_classes: the number of output classes
        :param stage_config: A list of Stage configs
        :param survival_prob: The stochastic depth survival probability
        :param se_ratio: The ratio of the Squeeze Excite block
        :param p_drop_out: The dropout probability
        :param nl_type: The non-linear function type
        :param batch_norm_epsilon: The batch normalization epsilon value
        :param batch_norm_momentum: The batch normalization momentum value
        """
        super(EfficientNet, self).__init__()

        blocks_list = []
        n_channels = input_channels
        base_drop_rate = 1.0 - survival_prob
        n_blocks = sum([sc.stem * sc.n_repeat for sc in stage_config])
        drop_rate = base_drop_rate / n_blocks
        past_index = 0
        for i, sc in enumerate(stage_config):
            if sc.stem:
                blocks_list.append(ConvBNNonLinear(nc, n_channels, sc.output_channels, kernel_size=sc.kernel_size,
                                                   stride=sc.stride_first,
                                                   nl_type=nl_type, batch_norm_epsilon=batch_norm_epsilon,
                                                   batch_norm_momentum=batch_norm_momentum, tf_padding=True))
            else:
                survival_prob_start = 1.0 - drop_rate * past_index
                blocks_list.append(
                    RepeatedInvertedResidual(nc, sc.n_repeat, n_channels, sc.output_channels, sc.stride_first,
                                             expand_ratio=sc.expand_ratio,
                                             kernel_size=sc.kernel_size,
                                             nl_type=nl_type,
                                             se_ratio=se_ratio,
                                             survival_prob_start=survival_prob_start, drop_rate=drop_rate,
                                             batch_norm_epsilon=batch_norm_epsilon, tf_padding=True))
                past_index += sc.n_repeat
            n_channels = sc.output_channels
        self.conv_blocks = nn.Sequential(*blocks_list)
        self.conv_blocks_list = blocks_list

        self.conv_head = ConvBNNonLinear(nc, n_channels, 1280, kernel_size=1, nl_type=nl_type,
                                         batch_norm_epsilon=batch_norm_epsilon)

        self.gap = GlobalAvgPool2d()
        self.drop_out = nn.Dropout(p=p_drop_out)
        self.fc = layers.FullyConnected(nc, 1280, n_classes)

    def forward(self, x):
        res_list = []
        for i, b in enumerate(self.conv_blocks_list):
            x = b(x)
            res_list.append(x)

        x = self.conv_head(x)
        x = self.gap(x)
        x = self.drop_out(x.squeeze(dim=-1).squeeze(dim=-1))
        return self.fc(x)


def efficient_net(nc, pretrained=False, progress=True, **kwargs):
    """
    An implementation of Efficient-Net-B0 model from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_
    :param nc: Network Controller class
    :param pretrained: Load pretrained weights
    :param progress: Show download progress bar
    :param kwargs: A kwargs dict for controlling module config
    :return: A PyTorch module
    """

    net = EfficientNet(nc, **kwargs)
    if pretrained:
        state_dict = torch.load('./models/EfficientNet_B0_IMAGENET.pt',
                                map_location='cpu')
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
    return net
