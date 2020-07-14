import torch
from networks import layers
from quantization_common.quantization_enums import TensorType
import numpy as np


def calculate_weight_compression(model):
    """
    This function calculate the weight compression rate of a model
    :param model: Input model
    :return: a floating point number
    """
    float_size = 0
    fxp_size = 0
    for m in model.modules():
        if isinstance(m, layers.Quantization) and m.is_coefficient():
            float_size += m.get_float_size()
            fxp_size += m.get_fxp_size()
    return float_size / fxp_size


def calculate_activation_max_compression(model):
    """
    This function calculate the activation max compression rate of a model
    :param model: Input model
    :return: a floating point number
    """
    float_size = 0
    fxp_size = 0
    for m in model.modules():
        if isinstance(m, layers.Quantization) and m.is_activation():
            float_size = max(m.get_float_size(), float_size)
            fxp_size = max(m.get_fxp_size(), fxp_size)
    return float_size / fxp_size


def calculate_expected_weight_compression(model):
    """
    This function returns the expected weight compression of the entire module
    :param model: An Input PyTorch module with HMQ quantization
    :return: A Tensor with a single value of the expected weight compression
    """
    reg_list_coeff = []
    total_max_size_coeff = []

    for n, m in model.named_modules():
        if isinstance(m, layers.Quantization):
            if len(m.bits_vector) >= 1:
                if m.is_coefficient():
                    reg_list_coeff.append(m.get_expected_bits() * m.get_expected_tensor_size())
                    total_max_size_coeff.append(
                        torch.Tensor([m.get_float_size()]).cuda())
    cr = torch.stack(total_max_size_coeff).sum() / torch.stack(reg_list_coeff).sum()
    return cr


def get_thresholds_list(nc, model) -> list:
    """
    This function return the thresholds values of all layers
    :param nc: The network controller
    :param model: The PyTorch network module
    :return: A list of thresholds
    """
    list_of_maps = []
    for m in model.modules():
        if isinstance(m, layers.Quantization):
            if (m.tensor_type == TensorType.ACTIVATION and not nc.is_float_activation) or (
                    m.tensor_type == TensorType.COEFFICIENT and not nc.is_float_coefficient):
                list_of_maps.append(np.squeeze(m.base_thresholds.detach().cpu().numpy(), axis=0))
    return list_of_maps
