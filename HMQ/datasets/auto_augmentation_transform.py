import random
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
import torchvision.transforms as transforms

PARAMETER_MAX = 10  # What is the max 'level' a transform could be predicted


def int_parameter(level, maxval) -> int:
    """
    A function to scale between zero to max val with casting to int
    :param level: The level of augmentation an integer between 0 to 9
    :param maxval: The maximal output value
    :return: A int value
    """
    return int(level * maxval / PARAMETER_MAX)


def float_parameter(level, maxval):
    """
    A function to scale between zero to max val
    :param level: The level of augmentation an integer between 0 to 9
    :param maxval: The maximal output value
    :return: A int value
    """
    return float(level) * maxval / PARAMETER_MAX


class BaseAugmentation(object):
    def __init__(self, p, level):
        """
        Base Augmentation class which performed data augmentation with probability p and severity level

        :param p: The probability that the augmentation is performed
        :param level: The severity of data augmentation
        """
        self.p = p
        self.level = level

    def __call__(self, img):
        if random.random() < self.p:
            img = self._augment(img)
        return img

    def _augment(self, img):
        raise NotImplemented

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, level={})'.format(self.p, self.level)


class AutoContrast(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform auto contrast.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        return ImageOps.autocontrast(img)


class Equalize(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Equalization.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        return ImageOps.equalize(img)


class Invert(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Invert.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        return ImageOps.invert(img)


class Blur(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Image Blur.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        return img.filter(ImageFilter.BLUR)


class Smooth(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Smoothing.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        return img.filter(ImageFilter.SMOOTH)


class Rotate(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform rotation.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        degrees = int_parameter(self.level, 30)
        if random.random() > 0.5:
            degrees = -degrees
        return img.rotate(degrees)


class Posterize(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Posterize.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        level = int_parameter(self.level, 4)
        return ImageOps.posterize(img, 4 - level)


class ShearX(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform ShearX.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        level = float_parameter(self.level, 0.3)
        if random.random() > 0.5:
            level = -level
        return img.transform((32, 32), Image.AFFINE, (1, level, 0, 0, 1, 0))


class ShearY(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform ShearY.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        level = float_parameter(self.level, 0.3)
        if random.random() > 0.5:
            level = -level
        return img.transform((32, 32), Image.AFFINE, (1, 0, 0, level, 1, 0))


class TranslateX(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform TranslateX.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        level = int_parameter(self.level, 10)
        if random.random() > 0.5:
            level = -level
        return img.transform((32, 32), Image.AFFINE, (1, 0, level, 0, 1, 0))


class TranslateY(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform TranslateY.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        level = int_parameter(self.level, 10)
        if random.random() > 0.5:
            level = -level
        return img.transform((32, 32), Image.AFFINE, (1, 0, 0, 0, 1, level))


class Solarize(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Solarize.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        level = int_parameter(self.level, 256)
        return ImageOps.solarize(img, 256 - level)


class Color(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Color.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        v = float_parameter(self.level, 1.8) + .1  # going to 0 just destroys it
        return ImageEnhance.Color(img).enhance(v)


class Contrast(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Contrast change.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        v = float_parameter(self.level, 1.8) + .1  # going to 0 just destroys it
        return ImageEnhance.Contrast(img).enhance(v)


class Brightness(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Brightness change.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        v = float_parameter(self.level, 1.8) + .1  # going to 0 just destroys it
        return ImageEnhance.Brightness(img).enhance(v)


class Sharpness(BaseAugmentation):
    def _augment(self, img):
        """
        Data augmentation function that perform Sharpness change.

        :param img: PIL Image to be augmented.
        :return: PIL Image after augmentation
        """
        v = float_parameter(self.level, 1.8) + .1  # going to 0 just destroys it
        return ImageEnhance.Sharpness(img).enhance(v)


CIFAR10_AUGMENT_POLICY = transforms.RandomChoice([transforms.Compose([Invert(0.1, 7), Contrast(0.2, 6)]),
                                                  transforms.Compose([Rotate(0.7, 2), TranslateX(0.3, 9)]),
                                                  transforms.Compose([Sharpness(0.8, 1), Sharpness(0.9, 3)]),
                                                  transforms.Compose([ShearY(0.5, 8), TranslateY(0.7, 9)]),
                                                  transforms.Compose([AutoContrast(0.5, 8), Equalize(0.9, 2)]),
                                                  transforms.Compose([ShearY(0.2, 7), Posterize(0.3, 7)]),
                                                  transforms.Compose([Color(0.4, 3), Brightness(0.6, 7)]),
                                                  transforms.Compose([Sharpness(0.3, 9), Brightness(0.7, 9)]),
                                                  transforms.Compose([Equalize(0.6, 5), Equalize(0.5, 1)]),
                                                  transforms.Compose([Contrast(0.6, 7), Sharpness(0.6, 5)]),
                                                  transforms.Compose([Color(0.7, 7), TranslateY(0.5, 8)]),
                                                  transforms.Compose([Equalize(0.3, 7), AutoContrast(0.4, 8)]),
                                                  transforms.Compose([TranslateY(0.4, 3), Sharpness(0.2, 6)]),
                                                  transforms.Compose([Brightness(0.9, 6), Color(0.2, 8)]),
                                                  transforms.Compose([Solarize(0.5, 2), Invert(0.0, 3)]),
                                                  transforms.Compose([Equalize(0.2, 0), AutoContrast(0.6, 0)]),
                                                  transforms.Compose([Equalize(0.2, 8), Equalize(0.6, 4)]),
                                                  transforms.Compose([Color(0.9, 9), Equalize(0.6, 6)]),
                                                  transforms.Compose([AutoContrast(0.8, 4), Solarize(0.2, 8)]),
                                                  transforms.Compose([Brightness(0.1, 3), Color(0.7, 0)]),
                                                  transforms.Compose([Solarize(0.4, 5), AutoContrast(0.9, 3)]),
                                                  transforms.Compose([TranslateY(0.9, 9), TranslateY(0.7, 9)]),
                                                  transforms.Compose([AutoContrast(0.9, 2), Solarize(0.8, 3)]),
                                                  transforms.Compose([Equalize(0.8, 8), Invert(0.1, 3)]),
                                                  transforms.Compose([TranslateY(0.7, 9), AutoContrast(0.9, 1)])])

IMAGENET_AUGMENT_POLICY = transforms.RandomChoice([transforms.Compose([Posterize(0.4, 8), Rotate(0.6, 9)]),
                                                   transforms.Compose([Solarize(0.6, 5), AutoContrast(0.6, 5)]),
                                                   transforms.Compose([Equalize(0.8, 18), Equalize(0.6, 3)]),
                                                   transforms.Compose([Posterize(0.6, 7), Posterize(0.6, 6)]),
                                                   transforms.Compose([Equalize(0.4, 7), Solarize(0.2, 4)]),
                                                   transforms.Compose([Equalize(0.4, 4), Rotate(0.8, 8)]),
                                                   transforms.Compose([Solarize(0.6, 3), Equalize(0.6, 7)]),
                                                   transforms.Compose([Posterize(0.8, 5), Equalize(1.0, 2)]),
                                                   transforms.Compose([Rotate(0.2, 3), Solarize(0.6, 8)]),
                                                   transforms.Compose([Equalize(0.6, 8), Posterize(0.4, 6)]),
                                                   transforms.Compose([Rotate(0.8, 8), Color(0.4, 0)]),
                                                   transforms.Compose([Rotate(0.4, 9), Equalize(0.6, 2)]),
                                                   transforms.Compose([Equalize(0.0, 7), Equalize(0.8, 8)]),
                                                   transforms.Compose([Invert(0.6, 4), Equalize(1.0, 8)]),
                                                   transforms.Compose([Color(0.6, 4), Color(1.0, 8)]),
                                                   transforms.Compose([Rotate(0.8, 8), Color(1.0, 2)]),
                                                   transforms.Compose([Color(0.8, 8), Solarize(0.8, 7)]),
                                                   transforms.Compose([Sharpness(0.4, 7), Invert(0.6, 8)]),
                                                   transforms.Compose([ShearX(0.6, 5), Equalize(1.0, 9)]),
                                                   transforms.Compose([Color(0.4, 0), Equalize(0.6, 3)]),
                                                   transforms.Compose([Equalize(0.4, 7), Solarize(0.2, 4)]),
                                                   transforms.Compose([Solarize(0.6, 5), AutoContrast(0.6, 5)]),
                                                   transforms.Compose([Invert(0.6, 4), Equalize(1.0, 8)]),
                                                   transforms.Compose([Color(0.6, 4), Contrast(1.0, 8)]),
                                                   transforms.Compose([Equalize(0.8, 8), Equalize(0.6, 3)])])
