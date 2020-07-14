from enum import Enum


class TensorType(Enum):
    ACTIVATION = 0
    COEFFICIENT = 1


class QUANTIZATION(Enum):
    ACTIVATION = 0
    COEFFICIENT = 1
    BOTH = 2
