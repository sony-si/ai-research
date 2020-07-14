from torch import nn
from torch.nn import functional as F
from networks.layers.quantization import Quantization
from networks.controller.network_controller import NetworkQuantizationController


class FullyConnected(nn.Module):
    def __init__(self, network_controller: NetworkQuantizationController, in_channels, out_channels):
        """
        A fully connected module with HMQ quantization of the weights.
        :param network_controller: The network quantization controller
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        """
        super(FullyConnected, self).__init__()
        self.network_controller = network_controller
        self.fc = nn.Linear(in_channels, out_channels)
        self.q = Quantization(network_controller, is_signed=True,
                              weights_values=self.fc.weight.detach())

    def forward(self, x):
        """
        The forward function of the FullyConnected module

        :param x: Input tensor x
        :return: A tensor after fully connected
        """
        if self.network_controller.is_float_coefficient:
            return self.fc(x)
        else:
            return F.linear(x, self.q(self.fc.weight), self.fc.bias)
