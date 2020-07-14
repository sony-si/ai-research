import torch
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from networks.controller.network_controller import NetworkQuantizationController
from quantization_common.quantization_enums import TensorType
from networks.layers.base_quantization import BaseQuantization
from networks.layers.efficient_quantization import EfficientBaseQuantization
from networks.layers.gumbel_softmax import GumbelSoftmax


class Quantization(nn.Module):
    def __init__(self, network_controller: NetworkQuantizationController, is_signed: bool,
                 alpha: float = 0.9, weights_values=None, efficient=True):
        """
        HMQ Block
        :param network_controller: The network controller
        :param is_signed: is this tensor signed
        :param alpha: the thresholds I.I.R value
        :param weights_values: In the case of weights quantized this is the tensors values
        :param efficient: Boolean flag stating to use the memory efficient
        """
        super(Quantization, self).__init__()
        self.weights_values = weights_values
        if weights_values is None:
            self.tensor_type = TensorType.ACTIVATION
            self.tensor_size = None
        else:
            self.tensor_type = TensorType.COEFFICIENT
            self.tensor_size = np.prod(weights_values.shape)

        self.network_controller = network_controller
        self.alpha = alpha
        self.is_signed_tensor = torch.Tensor([float(is_signed)]).cuda()

        if efficient:
            self.base_q = EfficientBaseQuantization()
        else:
            self.base_q = BaseQuantization()
        self.gumbel_softmax = GumbelSoftmax(ste=network_controller.ste)

        self.bits_vector = None
        self.mv_shifts = None
        self.base_thresholds = None
        self.nb_shifts_points_div = None
        self.search_matrix = None

    def init_quantization_coefficients(self):
        """
        This function initlized the HMQ parameters
        :return: None
        """
        init_threshold = 0
        n_bits_list, thresholds_shifts = self.network_controller.quantization_config.get_thresholds_bitwidth_lists(self)
        if self.is_coefficient():
            init_threshold = torch.pow(2.0, self.weights_values.abs().max().log2().ceil() + 1).item()
        if self.is_activation():
            n_bits_list = [8]

        self._init_quantization_params(n_bits_list, thresholds_shifts, init_threshold)
        self._init_search_matrix(self.network_controller.p, n_bits_list, len(thresholds_shifts))

    def _init_quantization_params(self, bit_list, thresholds_shifts, init_thresholds):
        self.update_bits_list(bit_list)
        self.mv_shifts = Parameter(torch.Tensor(thresholds_shifts), requires_grad=False)
        self.thresholds_shifts_points_div = Parameter(torch.pow(2.0, self.mv_shifts), requires_grad=False)
        self.base_thresholds = Parameter(torch.Tensor(1), requires_grad=False)
        init.constant_(self.base_thresholds, init_thresholds)

    def _init_search_matrix(self, p, n_bits_list, n_thresholds_options):
        n_channels = 1
        sm = -np.random.rand(n_channels, len(n_bits_list), n_thresholds_options, 1)
        n = np.prod(sm.shape)
        sm[:, 0, 0, 0] = np.log(p * n / (1 - p))  # for single channels
        self.search_matrix = Parameter(torch.Tensor(sm))

    def _get_quantization_probability_matrix(self, batch_size=1, noise_disable=False):
        return self.gumbel_softmax(self.search_matrix, self.network_controller.temperature, batch_size=batch_size,
                                   noise_disable=noise_disable)

    def _get_bits_probability(self, batch_size=1, noise_disable=False):
        p = self._get_quantization_probability_matrix(batch_size=batch_size, noise_disable=noise_disable)
        return p.sum(dim=4).sum(dim=3).sum(dim=1)

    def _update_iir(self, x):  # update scale using statistics
        if self.is_activation():
            if self.tensor_size is None:
                self.tensor_size = np.prod(x.shape[1:])  # Remove batch axis
            max_value = x.abs().max()
            self.base_thresholds.data.add_(self.alpha * (max_value - self.base_thresholds))

    def _calculate_expected_delta(self, p, max_scale):
        max_scales = max_scale / (self.thresholds_shifts_points_div.reshape(1, -1))
        max_scales = max_scales.reshape(1, 1, 1, -1, 1)

        nb_shifts = self.nb_shifts_points_div.reshape(1, 1, -1, 1, 1) * torch.pow(2.0, -self.is_signed_tensor)
        delta = (max_scales / nb_shifts) * p
        return delta.sum(dim=-1).sum(dim=-1).sum(dim=-1).sum(dim=-1)

    def _calculate_expected_threshold(self, p, max_threshold):
        p_t = p.sum(dim=4).sum(dim=2).sum(dim=1)
        thresholds = max_threshold / (self.thresholds_shifts_points_div.reshape(1, -1))
        return (p_t * thresholds).sum(dim=-1)

    def _calculate_expected_q_point(self, p, max_threshold, expected_delta, param_shape):
        t = self._calculate_expected_threshold(p, max_threshold=max_threshold).reshape(*param_shape)
        return t / expected_delta

    def _built_param_shape(self, x):
        random_size = x.shape[0] if self.is_activation() else x.shape[1]  # select random
        if len(x.shape) == 4:
            param_shape = [random_size, -1, 1, 1] if self.is_activation() else [-1, random_size, 1, 1]
        elif len(x.shape) == 2:
            param_shape = [random_size, -1] if self.is_activation() else [-1, random_size]
        else:
            raise NotImplemented
        return random_size, param_shape

    def forward(self, x):
        """
        The forward function of the HMQ module

        :param x: Input tensor x
        :return: A tensor after quantization
        """
        if self.network_controller.statistics_update:
            self._update_iir(x)
        max_threshold = torch.pow(2.0,
                                  torch.ceil(torch.log2(self.base_thresholds.detach().abs()))).detach()  # read scale
        if self.training and self.network_controller.temperature > 0:
            random_size, param_shape = self._built_param_shape(x)
            # axis according to tensor type (activation randomization is done over the batch axis,
            # coeff the randomization is done over the input channel axis)
            p = self._get_quantization_probability_matrix(batch_size=random_size)
            delta = self._calculate_expected_delta(p, max_threshold).reshape(*param_shape)
            q_points = self._calculate_expected_q_point(p, max_threshold, delta,
                                                        param_shape).reshape(*param_shape)
            return self.base_q(x, delta, q_points, self.is_signed_tensor)
        else:  # negative temperature/ infernce
            p = self._get_quantization_probability_matrix(batch_size=1, noise_disable=True).squeeze(dim=0)
            bits_index = torch.argmax(self._get_bits_probability(batch_size=1, noise_disable=True).squeeze(dim=0))
            max_index = torch.argmax(p[:, bits_index, :, 0], dim=-1)
            q_points = self.nb_shifts_points_div[bits_index] * torch.pow(2.0,
                                                                         -self.is_signed_tensor)
            max_scales = (max_threshold / self.thresholds_shifts_points_div.reshape(1, -1)).detach()
            delta = torch.stack(
                [(max_scales[i, mv] / q_points) for i, mv in enumerate(max_index)]).flatten().detach()
            return self.base_q(x, delta, q_points, self.is_signed_tensor)

    def get_bit_width(self):
        """
        This function return the selected bit-width
        :return: the bit-width of the HMQ
        """
        return self.bits_vector[torch.argmax(self._get_bits_probability(noise_disable=True).flatten())].item()

    def get_expected_bits(self):
        """
        This function return the expected bit-width
        :return: the expected bit-width of the HMQ
        """
        return (self.bits_vector * self._get_bits_probability(noise_disable=True)).sum()

    def get_float_size(self):
        """
        This function return the size of floating point tensor in bits
        Note: we assume 32 bits for floating point values
        :return: the floating point tensor size
        """
        return 32 * self.tensor_size

    def get_fxp_size(self):
        """
        This function return the size of quantized tensor in bits
        :return: the quantized tensor size
        """
        return self.get_bit_width() * self.tensor_size

    def is_activation(self):
        """
        This function return the boolean stating if this module quantize activation
        :return: a boolean flag stating if this activation quantization
        """
        return self.tensor_type == TensorType.ACTIVATION

    def is_coefficient(self):
        """
        This function return the boolean stating if this module quantize coefficient
        :return: a boolean flag stating if this coefficient quantization
        """
        return self.tensor_type == TensorType.COEFFICIENT

    def get_expected_tensor_size(self):
        """
         This function return the expected size of quantized tensor in bits
         :return: the expected size of quantized tensor
         """
        return torch.Tensor([self.tensor_size]).cuda()

    def update_bits_list(self, bits_list):
        """
        This function update the HMQ bit-width list
        :param bits_list: A list of new bit-widths
        :return: None
        """
        if self.bits_vector is None:
            self.bits_vector = Parameter(torch.Tensor(bits_list), requires_grad=False)
            self.nb_shifts_points_div = Parameter(
                torch.pow(2.0, self.bits_vector),  # - int(q_node.is_signed)
                requires_grad=False)  # move to init
        else:
            self.bits_vector.add_(torch.Tensor(bits_list).cuda() - self.bits_vector)
            self.nb_shifts_points_div.add_(torch.pow(2.0, self.bits_vector) - self.nb_shifts_points_div)
