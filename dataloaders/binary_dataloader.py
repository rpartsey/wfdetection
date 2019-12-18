from __future__ import print_function, division
import os
import sys

import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from skimage import io, transform
import numpy as np
import cv2
from .transformations import FromNumpy
from .align import AlignTransform
import rasterio
from torchvision import transforms


def read_tif(file_path):
    with rasterio.open(file_path) as dataset:
        bands = dataset.read()
        return bands


def cutBorders(X, y):
    shape = X.shape
    mask = X != 0
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    X = X[x0:x1, y0:y1]
    y = y[x0:x1, y0:y1]
    X = cv2.resize(X, shape, interpolation=cv2.INTER_NEAREST)
    y = cv2.resize(y, shape, interpolation=cv2.INTER_NEAREST)
    return X, y, [x0, x1, y0, y1]


def uncut(mask, cut_options):
    x0, x1, y0, y1 = cut_options
    res_mask = np.zeros_like(mask)
    mask = cv2.resize(mask, (y1 - y0, x1 - x0))
    res_mask[x0:x1, y0:y1] = mask
    return res_mask


# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

class BinaryLoader(Dataset):
    IMAGE_ID_COLUMN = "ImageId"
    MASK_EXISTS_COLUMN = "mask_exists"

    def __init__(self, csv_file, root_dir, image_transform=None, mask_transform=None, aug_transform=None, align=False,
                 cut_borders=False):
        self.input_df = csv_file
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.aug = aug_transform
        self.cut_borders = cut_borders
        self.align = None
        if align:
            # TODO: download: https://www.kaggleusercontent.com/kf/6702147/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..jONz7097UrpecDRNw3sGfQ.29OWj09H9U86_jZk9f5MLi-fvxu-QsrL2Mt7lVCrl1O2BBBZTawUA3rgBFJard-6vplDy5eIXMdqd9NqWtJmudbs0H8xXyX6gI1sAQgGHfSrdbpInKkvsBJ8CxcdBAvOLIxV74sx5P1MeviH_eoZK8PTqV2MatqrZTvNJoBafoXld_SLAG9BSrKZy5UfAooM.CU43Z5_wtrUdEgJVcCzB3w/unet_lung_seg.hdf5
            self.align = AlignTransform()

    def image_id(self, idx):
        return str(self.input_df.iloc[idx][self.IMAGE_ID_COLUMN])

    def __len__(self):
        return len(self.input_df)

    def mask_exists(self, idx):
        return self.input_df.iloc[idx][self.MASK_EXISTS_COLUMN]

    def __getitem__(self, idx):
        try:
            cur_id = self.input_df.iloc[idx][self.IMAGE_ID_COLUMN]
        except:
            print(idx, self.IMAGE_ID_COLUMN, self.input_df.shape)
            raise ValueError
        mask_exists = self.input_df.iloc[idx][self.MASK_EXISTS_COLUMN]

        # fn = "{}.png".format(cur_id)
        img_path = os.path.join(self.root_dir, 'img', cur_id)
        mask_path = os.path.join(self.root_dir, 'mask', cur_id)

        # Reading

        # original shape C x H x W
        X = np.array(read_tif(img_path)).transpose((1, 2, 0))
        X = (X / np.max(X) * 255).astype(np.uint8)
        y = read_tif(mask_path)[0]  # H x W

        X = cv2.resize(X, (256, 256))
        input_img = X.copy()

        transformations = transforms.Compose([transforms.ToTensor()])
        X = transformations(X)



        y = cv2.resize(y, (256, 256)).astype(np.uint8)
        y = torch.from_numpy(np.expand_dims(y, 0))



        # y = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # X = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


        # if self.align:
        #     X = self.align(X, y)
        #
        # if self.cut_borders:
        #     X, y, _ = cutBorders(X, y)
        #
        # if self.aug:
        #     X, y = self.aug(X, y)

        # if self.image_transform:
        #     X = self.image_transform(X)

        # if self.mask_transform:
        #     y = self.mask_transform(y)

        # X = X.float()
        # y = y.long()

        return X, y, input_img, int(mask_exists)

    def __str__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    pass
