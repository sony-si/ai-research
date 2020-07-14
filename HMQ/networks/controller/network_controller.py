from quantization_common.quantization_enums import QUANTIZATION
from quantization_common.quantization_config import QuantizationConfig


class NetworkQuantizationController(object):
    def __init__(self, quantization_config: QuantizationConfig, quantization_part: QUANTIZATION, p_init: float = 0.95,
                 ste: bool = False):
        """
        The NetworkQuantizationController handel's the configuration of quantization of a network
        :param quantization_config: A class that hold the quantization config (bit-width lists and thresholds lists)
        :param quantization_part: An enum stating which part of the network to quantized [ACTIVATION, COEFFICIENT, BOTH]
        :param p_init: The initial probability of the highest bit-width of the HMQ
        :param ste: A Boolean flag that enable an STE on the gumbel softmax
        """
        self.quantization_config: QuantizationConfig = quantization_config
        self._quantization_part: QUANTIZATION = quantization_part
        self.p: float = p_init
        self.ste: bool = ste
        self.statistics_update: bool = False
        self._is_float_coefficient: bool = True
        self._is_float_activation: bool = True
        self._temperature: float = 1.0

    @property
    def is_float_activation(self) -> bool:
        """
        This property indicate if activation are running in floating point
        :return: Boolean stating if the activation are running in floating point
        """
        return self._is_float_activation

    @property
    def is_float_coefficient(self):
        """
        This property indicate if coefficient are running in floating point
        :return: Boolean stating if the coefficient are running in floating point
        """
        return self._is_float_coefficient

    def enable_statistics_update(self):
        """
        This function enables statistics collection of thresholds I.I.R
        :return: None
        """
        self.statistics_update = True

    def disable_statistics_update(self):
        """
        This function disable statistics collection of thresholds I.I.R
        :return: None
        """
        self.statistics_update = False

    def apply_fix_point(self):
        """
        This function apply quantization operation across all HMQ's
        :return: None
        """
        self._is_float_activation = not (self._quantization_part in [QUANTIZATION.ACTIVATION, QUANTIZATION.BOTH])
        self._is_float_coefficient = not (self._quantization_part in [QUANTIZATION.COEFFICIENT, QUANTIZATION.BOTH])

    def set_temperature(self, t: float):
        """
        This function set the gumbel softmax temperature
        :param t: A floating point value of gumbel softmax temperature.
        :return:  None
        """
        self._temperature = t

    @property
    def temperature(self) -> float:
        """
        This function return the current gumbel softmax temperature
        :return: A floating point value of the current gumbel softmax temperature.
        """
        return self._temperature
