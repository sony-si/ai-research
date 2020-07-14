from typing import List


class BaseQuantizationConfig(object):
    def __init__(self, bit_list: List[int], thresholds_shift: List[int]):
        """
        A class of base quantization config (This class defined for each group of layer types)
        :param bit_list: A list of bit width to be used during the search phase
        :param thresholds_shift: A list of threshold shift used during the search phase
        """
        self.bit_list = bit_list
        self.thresholds_shift = thresholds_shift

    def update_bit_list(self, bit_list):
        """
        This function update ths bit width list
        :param bit_list: a list of bit widths
        :return: None
        """
        self.bit_list = bit_list


class QuantizationConfig(object):
    def __init__(self, weights_config: BaseQuantizationConfig,
                 activation_config: BaseQuantizationConfig):
        """
        This class contains the configure of quantization search
        :param weights_config: The Quantization for weights
        :param activation_config: The Quantization for activation
        """
        self.config_dict_weights = weights_config
        self.config_dict_activation = activation_config

    def get_thresholds_bitwidth_lists(self, m):
        """
        This function return the layer quantization config for the search which a tuple of list of bit widths and
        list of thresholds shift
        :param m: An input module to be quantized
        :return: A tuple of size two, with list of bit widths and list of thresholds shift
        """
        if hasattr(m, 'q'):
            if m.q.is_activation():
                layer_cfg = self.config_dict_activation
            else:
                layer_cfg = self.config_dict_weights
            return layer_cfg.bit_list, layer_cfg.thresholds_shift
        else:
            if m.is_activation():
                layer_cfg = self.config_dict_activation
            else:
                layer_cfg = self.config_dict_weights
            return layer_cfg.bit_list, layer_cfg.thresholds_shift


def get_q_config_same(activation_bits_list, weights_bit_list, n_thresholds_shift):
    """
    This function create a quantization config with the same parameter across all layer types.
    :param activation_bits_list: A list of bit width to bit widths for activation
    :param weights_bit_list: A list of bit width to bit widths for weights
    :param n_thresholds_shift: the number of threshold shift
    :return: Quantization config
    """
    threshold_shift = list(range(n_thresholds_shift))
    return QuantizationConfig(BaseQuantizationConfig(weights_bit_list, threshold_shift),
                              BaseQuantizationConfig(activation_bits_list, threshold_shift))
