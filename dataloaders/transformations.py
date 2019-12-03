import torch
from torchvision.transforms import Compose
# from albumentations.core.composition import Compose
import numpy as np
import cv2
from albumentations import Normalize

from dataloaders.aug import get_transforms


class FromNumpy(object):
    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("input image should be of type `np.ndarray`")
        shape = image.shape
        if len(shape) == 2:
            return torch.from_numpy(np.expand_dims(image, 0))
        elif len(shape) == 3 and shape[-1] == 3:
            return torch.from_numpy(image.transpose((2, 0, 1)))
        raise NotImplementedError("shape `{}`. Only grayscale images implemented.".format(shape))


class CustomNormalize(object):
    def __init__(self, threshold, left_mean, left_std, right_mean, right_std):
        self.threshold = threshold
        self.left_mean = left_mean
        self.left_std = left_std
        self.right_mean = right_mean
        self.right_std = right_std

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise ValueError("input image should be of type `np.ndarray`")
        img = img.astype(np.float32)
        if img.mean() < self.threshold:
            img -= self.left_mean
            img /= self.left_std
        else:
            img -= self.right_mean
            img /= self.left_std
        return img


class AllToSingleObject(object):
    def __call__(self, mask):
        if not isinstance(mask, np.ndarray):
            raise ValueError("input image should be of type `np.ndarray`")
        mask = (mask > 0).astype(np.uint8)
        return mask


class ToLong(object):
    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            raise ValueError("input data should be of type `torch.tensor`")
        return data.long()


class ToFloat(object):
    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            raise ValueError("input data should be of type `torch.tensor`")
        return data.float()


class CV2ToGray(object):
    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("input image should be of type `np.ndarray`")
        return cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)


class CV2ToColor(object):
    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("input image should be of type `np.ndarray`")
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)


class Gray2RGBTriple(object):
    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("input image should be of type `np.ndarray`")
        if len(data.shape) != 2:
            raise ValueError("input data shape should be `HxW`")
        return np.stack([data, data, data], axis=2)


class CV2HistNorm(object):
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise ValueError("input image should be of type `np.ndarray`")
        if len(img.shape) != 2:
            raise ValueError("input data shape should be `HxW`")
        img = cv2.equalizeHist(img)
        return img


class CLAHE(object):
    def __init__(self, clipLimit=2.0, tileGridSize=(16, 16)):
        print("CLAHE ->>", tileGridSize)
        self.clipLimit = clipLimit
        self.tileGridSize = tuple(tileGridSize)
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tuple(tileGridSize))

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise ValueError("input image should be of type `np.ndarray`")
        if len(img.shape) != 2:
            raise ValueError("input data shape should be `HxW`")
        img = self.clahe.apply(img)
        return img


class Dilate(object):
    def __call__(self, img):
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=2)
        return dilation


class Erode(object):
    def __call__(self, img):
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=2)
        return erosion


# class Augmentations(object):
#     def __init__(self, size, scope, crop):
#         self.size = size
#         self.scope = scope
#         self.crop = crop
#
#     def __call__(self, img, trg):
#         return get_transforms(self.size, self.scope, self.crop)(img, trg)


class DefaultNormalise(object):
    def __init__(self, mean, std):
        self.norm = Normalize(mean=mean, std=std)

    def __call__(self, x):
        return self.norm(image=x)["image"]


class ByDIMNormalise(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    def __call__(self, x):
        x = x.astype(np.float32)
        if self.mean:
            mean = self.mean
        else:
            mean = np.mean(x, axis=(0, 1))
        if self.std:
            std = self.std
        else:
            std = np.std(x, axis=(0, 1))
        return (x - mean) / std
#130.70300678664125 57.0585038384225

OBJECT_MAPPING = dict(
    alltosingleobject=AllToSingleObject,
    normalize=CustomNormalize,
    cv2tocolor=CV2ToColor,
    cv2togray=CV2ToGray,
    fromnumpy=FromNumpy,
    gray2rgbtriple=Gray2RGBTriple,
    tofloat=ToFloat,
    tolong=ToLong,
    histnorm=CV2HistNorm,
    clahe=CLAHE,
    dilate=Dilate,
    defaultnorm=DefaultNormalise,
    bydimnormalise=ByDIMNormalise
    # augmentations=Augmentations
)


def create_transform(config):
    transformations_list = []
    for obj in config:
        name = obj["name"]
        cls = OBJECT_MAPPING.get(name, None)
        if cls is None:
            continue
        params = obj.get("params", {})
        transformations_list.append(cls(**params))
    return Compose(transformations_list)
