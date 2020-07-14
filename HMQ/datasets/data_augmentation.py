from enum import Enum
import torchvision.transforms as transforms
from datasets.auto_augmentation_transform import CIFAR10_AUGMENT_POLICY, IMAGENET_AUGMENT_POLICY
from PIL import Image


class Augmentation(Enum):
    CropAndHorizontalFlip = 0
    NoAugmentation = 1
    ResizeCropAndHorizontalFlip = 2
    ResizeCenterCrop = 3
    CropAndHorizontalFlipVerticalFlipRotation = 4
    ToTensor = 5


def get_augmentation(augmentation_type: Augmentation, crop_size: int = 32, padding_size: int = 4,
                     resize_size: int = 256, distributed=True, enable_auto_augmentation=False):
    """
    Thins function return a compose of transforms for training data augmentation.
    :param augmentation_type: an enum of the augmentation try that should run on each image.
    :param crop_size: The size of the randomized crop
    :param padding_size: The size of the padding before the random crop
    :param resize_size: The size of the output image using  resize
    :param distributed: is distributed flag for multiple gpu's
    :param enable_auto_augmentation: A boolean flag the enable auto augment.
    :return: A compose of transforms
    """
    train_transform = transforms.Compose([])

    if augmentation_type in [Augmentation.CropAndHorizontalFlip,
                             Augmentation.CropAndHorizontalFlipVerticalFlipRotation]:
        train_transform.transforms.append(transforms.RandomCrop(crop_size, padding=padding_size))
    if augmentation_type == Augmentation.ResizeCropAndHorizontalFlip:
        train_transform.transforms.append(transforms.RandomResizedCrop(crop_size))
    if augmentation_type == Augmentation.ResizeCenterCrop:
        train_transform.transforms.append(transforms.Resize(resize_size, interpolation=Image.BICUBIC))
        train_transform.transforms.append(transforms.CenterCrop(crop_size))
    if augmentation_type in [Augmentation.CropAndHorizontalFlip,
                             Augmentation.ResizeCropAndHorizontalFlip,
                             Augmentation.CropAndHorizontalFlipVerticalFlipRotation]: train_transform.transforms.append(
        transforms.RandomHorizontalFlip())
    if augmentation_type in [Augmentation.CropAndHorizontalFlipVerticalFlipRotation]:
        train_transform.transforms.append(transforms.RandomVerticalFlip())
        train_transform.transforms.append(transforms.RandomRotation([-90, 90]))
    #########################
    # Auto Augmentation
    #########################
    if enable_auto_augmentation and crop_size == 32:
        train_transform.transforms.append(CIFAR10_AUGMENT_POLICY)

    if enable_auto_augmentation and crop_size == 224:  # ImageNet Networks
        train_transform.transforms.append(IMAGENET_AUGMENT_POLICY)

    if not distributed:
        train_transform.transforms.append(transforms.ToTensor())
    return train_transform
