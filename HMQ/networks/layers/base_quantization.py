import torch
from torch import nn


class RoundSTE(nn.Module):
    def __init__(self):
        """
        This module perform element-wise rounding with straight through estimator (STE).
        """
        super(RoundSTE, self).__init__()

    def forward(self, x):
        """
        The forward function of the rounding module

        :param x: Input tensor to be rounded
        :return: A rounded tensor
        """
        x_error = torch.round(x) - x
        return x + x_error.detach()


class Clipping(nn.Module):
    def __init__(self):
        """
        This module perform element-wise clipping.
        """
        super(Clipping, self).__init__()

    def forward(self, x, max_value, min_value):
        """
        The forward function of the clipping module

        :param x:  Input tensor to be clipped
        :param max_value: The maximal value of the tensor after clipping
        :param min_value: The minimal value of the tensor after clipping
        :return: A clipped tensor
        """
        x = torch.min(x, max_value)
        x = torch.max(x, min_value)
        return x


class BaseQuantization(nn.Module):
    def __init__(self):
        """
        This module perform element-wise quantization.
        """
        super(BaseQuantization, self).__init__()
        self.round = RoundSTE()
        self.clip = Clipping()
        self.symmetric = False

    def forward(self, x, delta, q_p, is_signed):
        """
        The forward function of the quantization module

        :param x: Input tensor to be quantized
        :param delta: The quantization step size
        :param q_p: The number of quantization step's
        :param is_signed: is quantization signed
        :return: A quantized tensor
        """
        symmetirc_q_p = (q_p - int(not self.symmetric)) * is_signed
        return delta * self.clip(self.round(x / delta), q_p, - symmetirc_q_p)
