import math
from networks import layers


def update_network_activation(input_nc, input_net, input_target_compression):
    """
    This function modified the bit widths of activation to achieve a target compression
    :param input_nc: Input network config
    :param input_net: Input network module
    :param input_target_compression: A Target compression for activation quantization
    :return: None
    """
    float_max = max([m.tensor_size for n, m in input_net.named_modules() if
                     isinstance(m, layers.Quantization) and m.is_activation()])

    for n, m in input_net.named_modules():
        if isinstance(m, layers.Quantization) and m.is_activation():
            n_bits_list, _ = input_nc.quantization_config.get_thresholds_bitwidth_lists(m)
            n_bits = max(
                min(int(math.floor((32 * float_max) / (input_target_compression * m.tensor_size))), max(n_bits_list)),
                min(n_bits_list))
            m.update_bits_list([n_bits])
