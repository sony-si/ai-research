from enum import Enum
import torchvision.transforms as transforms


class PreProcessing(Enum):
    CIFAR = 0
    ONE = 1
    IMAGENET = 2


def get_preprocessing(preprocessing: PreProcessing, mean=None, std=None):
    """
    This function return the normalization transform for image normalization.
    There are three pre-processing type
        1. Base on the empirical value from CIFAR dataset
        2. Setting the input range input -1 to 1
        3. Base on the empirical value from ImageNet dataset
    :param preprocessing: Pre processing type enum
    :param mean: overriding parameter of the mean values
    :param std: overriding parameter of the std values
    :return: a Normalize object
    """
    if mean is not None and std is not None:
        normalize = transforms.Normalize(mean, std)
    else:
        if preprocessing == PreProcessing.CIFAR:
            normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        elif preprocessing == PreProcessing.ONE:
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        elif preprocessing == PreProcessing.IMAGENET:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        else:
            raise NotImplemented
    return normalize
