from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, Rotate,
    RandomScale, RandomContrast, RandomBrightness, RandomBrightnessContrast, OneOf, Compose, RandomGamma, VerticalFlip,
    HorizontalFlip, PadIfNeeded, RandomCrop, RGBShift
)
import cv2

# took from https://www.kaggle.com/sinitame/classification-with-transfert-learning

"""
Useful augmentation for medical image classification papers:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5977656/
https://www.quantib.com/blog/image-augmentation-how-to-overcome-small-radiology-datasets
https://www.accentsjournals.org/PaperDirectory/Journal/TIPCV/2019/2/2.pdf
"""


class StandartAugmentation:

    def __init__(self, p=0.8, size=(256, 256)):
        self.p = p
        self.width, self.height = size
        self.aug = self.__build_augmentator()

    def __call__(self, x, y):
        augm_res = self.aug(image=x, mask=y)
        return augm_res['image'], augm_res['mask']

    def __build_augmentator(self):
        return Compose([
            # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, p=0.3),
            # OneOf([
            #     RandomScale(scale_limit=0.05, interpolation=1, p=0.5),
            #     Rotate(limit=7, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5)
            # ], p=0.5),
            # PadIfNeeded(always_apply=True, min_width=self.width, min_height=self.height),
            # RGBShift(p=0.3),
            # RandomCrop(width=self.width, height=self.height),
            OneOf([
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
            ], p=0.5)
            # OneOf([
            #     # RandomBrightness(limit=0.2, always_apply=False, p=0.5),
            #     # RandomContrast(),
            #     RandomGamma()
            # ], p=0.7),
        ], p=self.p)


class FrogAugmentation:
    def __init__(self, p=0.8, size=(256, 256)):
        self.p = p
        self.width, self.height = size
        self.aug = self.__build_augmentator()

    def __call__(self, x, y):
        augm_res = self.aug(image=x, mask=y)
        return augm_res['image'], augm_res['mask']

    def __build_augmentator(self):
        return Compose([
            ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=13,
                             interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT,
                             mask_value=0, value=0.0, p=0.95),
            VerticalFlip(p=0.5),
            RandomContrast(limit=0.4, p=0.95),
        ],)
