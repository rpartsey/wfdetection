import torch
import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import pandas as pd

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.torchhub_unet import UNet
from dataloaders.binary_dataloader import BinaryLoader

import rasterio
from rasterio.windows import Window

weights_path = '/home/rpartsey/code/wfdetection/experiments/open_cities_1/last.h5'
state_dict = torch.load(weights_path, map_location='cuda:0')
model = UNet(3, 1, pretrained=False)
model.load_state_dict(state_dict['model'])

import numpy as np
import torch


class FromNumpy(object):
    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("input image should be of type `np.ndarray`")
        shape = image.shape
        if len(shape) == 2:
            return torch.from_numpy(np.expand_dims(image, 0))
        elif len(shape) == 3:
            return torch.from_numpy(image.transpose((2, 0, 1)))
        raise NotImplementedError("shape `{}`. Only grayscale images implemented.".format(shape))


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


from torchvision.transforms import Compose

image_transforms = Compose([FromNumpy(), ToFloat()])
mask_transforms = Compose([FromNumpy(), ToLong()])


def generate_window_offsets(image_h, image_w, window_h, window_w):
    """
    Returns iterable with window column and row offsets(top left corner).

    :param image_h: height of raster image
    :param image_w: width of raster image
    :param window_h: window height
    :param window_w: window width
    :return: iterable
    """

    def shift(raster_size, window_size):
        return (raster_size % window_size) // 2

    row_coord = -shift(image_h, window_h)
    col_coord = -shift(image_w, window_w)

    rows = np.arange(row_coord, image_h, window_h)
    cols = np.arange(col_coord, image_w, window_w)

    rows, cols = np.meshgrid(rows, cols, indexing='ij')

    return zip(rows.ravel(), cols.ravel())

from glob import glob

IMAGE_H = 1024
IMAGE_W = 1024

WINDOW_H = 256
WINDOW_W = 256

THRESHOLD = 0.5

window_offsets = list(generate_window_offsets(IMAGE_H, IMAGE_W, WINDOW_H, WINDOW_W))

dest_mask_dir = '/datasets/rpartsey/open_cities/output/unet_pred'

test_img_dir = '/datasets/rpartsey/open_cities/test/*/*.tif'
counter = 0

to_process = sorted(glob(test_img_dir))[4000:6000]
for path in to_process:

    dest_path = os.path.join(dest_mask_dir, os.path.basename(path))

    with rasterio.open(path) as source:
        dest_meta = {
            **source.meta,
            'dtype': rasterio.uint8,
            'count': 1
        }
        with rasterio.open(dest_path, 'w', **dest_meta) as dest:
            for row_off, col_off in window_offsets:
                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=WINDOW_W,
                    height=WINDOW_H
                )

                r, g, b, alpha = source.read(window=window, boundless=True, fill_value=0)

                X_np = np.array((r, g, b)).transpose((1, 2, 0))
                #         print(X_np.shape)
                X = image_transforms(X_np)
                with torch.no_grad():
                    processed_logits = model(X.unsqueeze(0))
                    processed_logits = processed_logits.sigmoid()

                mask = (processed_logits.numpy() > THRESHOLD).astype(np.uint8)

                dest.write(mask[0][0], window=window, indexes=1)
