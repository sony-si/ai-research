import torch
from torch import nn
import numpy as np


class _EfficientBaseQuantizationFunction(torch.autograd.Function):

    @staticmethod
    def clip(x, min_value, max_value):
        x = torch.min(x, max_value)
        x = torch.max(x, min_value)
        return x

    @staticmethod
    def forward(ctx, x, delta, q_p, is_signed):
        q_int_float = x / delta
        q_int = torch.round(q_int_float)
        q_int_clip = _EfficientBaseQuantizationFunction.clip(q_int, -(q_p - 1.0) * is_signed, q_p)
        result = delta * q_int_clip
        ctx.save_for_backward(q_int_float, delta, q_p, is_signed)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        q_int_float = ctx.saved_tensors[0]
        delta = ctx.saved_tensors[1]
        q_p = ctx.saved_tensors[2]
        is_signed = ctx.saved_tensors[3]

        q_p_shape_array = np.asarray(q_p.shape)
        delta_shape_array = np.asarray(delta.shape)
        index2sum_q_p = list(range(len(q_int_float.shape))) if np.prod(q_p_shape_array) == 1 else list(
            np.where(q_p_shape_array == 1)[0])
        index2sum_delta = list(range(len(q_int_float.shape))) if np.prod(delta_shape_array) == 1 else list(
            np.where(delta_shape_array == 1)[0])
        q_int = torch.round(q_int_float)
        upper_bound = q_p
        lower_bound = -(q_p - 1.0) * is_signed
        enable_low = (q_int < upper_bound).float()
        enable_high = (q_int > lower_bound).float()
        not_enable_low = 1 - enable_low
        not_enable_high = 1 - enable_high
        #############################################################
        grad_output_x = grad_output * enable_low * enable_high
        #############################################################
        delta_mid = grad_output_x * (q_int - q_int_float)
        delta_high = grad_output * not_enable_high * lower_bound
        delta_low = grad_output * not_enable_low * upper_bound
        grad_output_delta = (delta_mid + delta_high + delta_low).sum(dim=index2sum_delta).reshape(delta.shape)
        #############################################################
        # grad_output_c
        #############################################################

        grad_output_c = delta * (grad_output * (not_enable_low - is_signed * not_enable_high)).sum(
            dim=index2sum_q_p).reshape(
            q_p.shape)
        #############################################################
        # is signed grad
        #############################################################
        grad_output_p = -(delta * (grad_output * (q_p - 1) * not_enable_high)).sum(dim=index2sum_q_p)
        if is_signed.shape[0] != 1:
            # import pdb
            # pdb.set_trace()
            grad_output_p = grad_output_p.reshape(is_signed.shape)

        return grad_output_x, grad_output_delta, grad_output_c, grad_output_p


class EfficientBaseQuantization(nn.Module):
    def __init__(self):
        """
        Memory Efficient implementation of the base quantization module
        """
        super(EfficientBaseQuantization, self).__init__()

    def forward(self, x, delta, q_p, is_signed):
        """
        The forward function of the quantization module

        :param x: Input tensor to be quantized
        :param delta: The quantization step size
        :param q_p: The number of quantization step's
        :param is_signed: is quantization signed
        :return: A quantized tensor
        """
        return _EfficientBaseQuantizationFunction.apply(x, delta, q_p, is_signed)
