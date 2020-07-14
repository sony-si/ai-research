import os
import torch
import torchvision
import numpy as np
from enum import Enum
from datasets.data_preprocessing import get_preprocessing, PreProcessing
from datasets.data_augmentation import get_augmentation, Augmentation
from torch.utils.data import DataLoader


class Dataset(Enum):
    CIFAR10 = 0
    ImageNet = 1


CIFAR_IMAGE_SIZE = 32
IMAGENET_CROP_SIZE = 224
IMAGENET_RESIZE_SIZE = 256


def _fast_collate(batch):
    imgs = [img[0] for img in batch]
    c = 1 if len(np.asarray(imgs[0]).shape) == 2 else 3
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), c, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def _get_dataset_augmentation_normalization(dataset_enum, distributed=True, enable_auto_augmentation=False):
    if dataset_enum == Dataset.CIFAR10:
        normalization = get_preprocessing(PreProcessing.CIFAR)
        train_transform = get_augmentation(Augmentation.CropAndHorizontalFlip,
                                           crop_size=CIFAR_IMAGE_SIZE,
                                           distributed=distributed,
                                           enable_auto_augmentation=enable_auto_augmentation)
        validation_transform = get_augmentation(Augmentation.NoAugmentation,
                                                crop_size=CIFAR_IMAGE_SIZE,
                                                distributed=distributed)
    elif dataset_enum == Dataset.ImageNet:
        normalization = get_preprocessing(PreProcessing.IMAGENET)
        train_transform = get_augmentation(Augmentation.ResizeCropAndHorizontalFlip, crop_size=IMAGENET_CROP_SIZE,
                                           distributed=distributed)
        validation_transform = get_augmentation(Augmentation.ResizeCenterCrop, crop_size=IMAGENET_CROP_SIZE,
                                                resize_size=IMAGENET_RESIZE_SIZE, distributed=distributed)
    else:
        raise NotImplemented
    if not distributed and normalization is not None:
        train_transform.transforms.append(normalization)
        validation_transform.transforms.append(normalization)
    return train_transform, validation_transform


def get_dataset(dataset: Dataset, data_path: str, batch_size: int, num_workers: int = 4, distributed=True,
                enable_auto_augmentation=False):
    """
    This function return the dataset loaders for the validation and training sets also
    with training sampler for multiple gpu usage
    :param dataset: the dataset enum (CIFAR10 or ImageNet)
    :param data_path: the data folder in ImageNet
    :param batch_size: the training and validation batch size
    :param num_workers: the number of working
    :param distributed: working in distributed mode
    :param enable_auto_augmentation: this flag enable the auto augmentation
    :return: train loader, validation loader and training sampler.
    """
    train_transform, test_transform = _get_dataset_augmentation_normalization(dataset,
                                                                              distributed=distributed,
                                                                              enable_auto_augmentation=enable_auto_augmentation)
    if dataset == Dataset.CIFAR10:
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                download=True,
                                                transform=train_transform)  # transformation (preporcess and augmentation)

        testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                               download=True, transform=test_transform)
    elif dataset == Dataset.ImageNet:
        trainset = torchvision.datasets.ImageFolder(os.path.join(data_path, 'train'),
                                                    transform=train_transform)

        testset = torchvision.datasets.ImageFolder(os.path.join(data_path, 'validation'),
                                                   transform=test_transform)
    else:
        raise NotImplemented

    train_sampler = None
    val_sampler = None
    if distributed:
        print("Starting Distributed Datasets")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler,
                              collate_fn=_fast_collate if distributed else None)  # loading data using multipy therd
    test_loader = None
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True,
                                 sampler=val_sampler,
                                 collate_fn=_fast_collate if distributed else None
                                 )
    return train_loader, test_loader, train_sampler
