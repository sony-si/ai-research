import torch
from torch import nn
from networks.controller.network_controller import NetworkQuantizationController
from networks.layers.quantization import Quantization
from enum import Enum


class NonLinearType(Enum):
    RELU = 0
    RELU6 = 1
    IDENTITY = 2
    SIGMOID = 3
    SWISH = 4
    HSWISH = 5


class HSwish(nn.Module):
    def __init__(self):
        """
        An HSwish module
        :param inplace: A boolean stating if the operation is inplace
        """
        super(HSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        """
        The forward function of the HSwish module
        :param x: Input tensor x
        :return: A tensor after HSwish
        """
        return x * self.relu6(x + 3.0) / 6.0


class Swish(nn.Module):
    def __init__(self, inplace):
        """
        An Swish module
        :param inplace: A boolean stating if the operation is inplace
        """
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        The forward function of the Swish module
        :param x: Input tensor x
        :return: A tensor after Swish
        """
        return x * self.sigmoid(x)


class Identity(nn.Module):

    def __init__(self, inplace=False):
        """
        An Identity module (just for adding a fake inplace flag)
        :param inplace: A boolean stating if the operation is inplace
        """
        super(Identity, self).__init__()

    def forward(self, input):
        """
        The forward function of the Identity module
        :param input_tensor: Input tensor x
        :return: A tensor after Identity
        """
        return input


class Sigmoid(nn.Module):
    def __init__(self, inplace=False):
        """
        An Sigmoid module (just for adding a fake inplace flag)
        :param inplace: A boolean stating if the operation is inplace
        """
        super(Sigmoid, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        The forward function of the Sigmoid module
        :param input_tensor: Input tensor x
        :return: A tensor after Sigmoid
        """
        return self.s(input_tensor)


class _SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def __init__(self, inplace):
        """
        An MemoryEfficientSwish module
        :param inplace: A boolean stating if the operation is inplace
        """
        super(MemoryEfficientSwish, self).__init__()

    def forward(self, x):
        """
        The forward function of the MemoryEfficientSwish module
        :param x: Input tensor x
        :return: A tensor after MemoryEfficientSwish
        """
        return _SwishImplementation.apply(x)


NL_DICT = {NonLinearType.RELU: nn.ReLU,
           NonLinearType.IDENTITY: Identity,
           NonLinearType.RELU6: nn.ReLU6,
           NonLinearType.SIGMOID: Sigmoid,
           NonLinearType.SWISH: MemoryEfficientSwish,
           NonLinearType.HSWISH: HSwish}


class NonLinear(nn.Module):
    def __init__(self, network_controller: NetworkQuantizationController, out_channels,
                 nl_type: NonLinearType = NonLinearType.RELU):
        """
        A non-linear module with HMQ quantization after the non-linear.
        :param network_controller: The network quantization controller
        :param out_channels: The number of output channels
        :param nl_type: enum that state the non-linear type.
        """
        super(NonLinear, self).__init__()
        self.network_controller = network_controller
        self.nl_type = nl_type
        self.nl_func = NL_DICT[nl_type](inplace=True)
        is_signed_list = [NonLinearType.IDENTITY, NonLinearType.SWISH, NonLinearType.HSWISH]
        self.q = Quantization(network_controller, is_signed=nl_type in is_signed_list)

    def forward(self, x):
        """
        The forward function of the NonLinear module

        :param x: Input tensor x
        :return: A tensor after non-linear
        """
        if self.network_controller.is_float_activation:
            return self.nl_func(x)
        elif self.nl_type != NonLinearType.RELU:
            x = self.nl_func(x)
        return self.q(x)
