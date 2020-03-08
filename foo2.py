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
from glob import iglob


# import sys
# sys.path.insert(0, '../')

from torch import nn
from models import SmpUnet
from dataloaders.binary_dataloader import BinaryLoader

import rasterio
from rasterio.windows import Window


# ids_file_path = '/datasets/rpartsey/open_cities/output/chip/1/chips/open_cities_valid.csv'
# data_dir = '/home/partsey/data/separated_tif'
weights_path = '/home/rpartsey/code/foo/experiments/open_cities_smp_unet_binary_focal_dice_075_huge_dataset_300samples_4stable/last.h5'
# df = pd.read_csv(ids_file_path)

state_dict = torch.load(weights_path, map_location='cuda')
model = nn.DataParallel(SmpUnet(3, 1, pretrained=False))
model.load_state_dict(state_dict['model'])
model.cuda()


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

IMAGE_H = 1024
IMAGE_W = 1024

WINDOW_H = 256
WINDOW_W = 256

THRESHOLD = 0.5

# window_offsets = list(generate_window_offsets(IMAGE_H, IMAGE_W, WINDOW_H, WINDOW_W))

dest_mask_dir = '/datasets/rpartsey/open_cities/output/open_cities_smp_unet_binary_focal_dice_075_huge_dataset_300samples_4stable'

test_img_dir = '/datasets/rpartsey/open_cities/test/*/*.tif'
counter = 0
for path in iglob(test_img_dir):

    if counter % 100 == 0:
        print(counter)
    counter += 1

    dest_path = os.path.join(dest_mask_dir, os.path.basename(path))

    with rasterio.open(path) as source:
        dest_meta = {
            **source.meta,
            'dtype': rasterio.uint8,
            'count': 1
        }
        with rasterio.open(dest_path, 'w', **dest_meta) as dest:
            r, g, b, alpha = source.read()

            X_np = np.array((r, g, b)).transpose((1, 2, 0))

            X = image_transforms(X_np)
            with torch.no_grad():
                processed_logits = model(X.unsqueeze(0))
                processed_logits = processed_logits.sigmoid()

            mask = (processed_logits.cpu().numpy() > THRESHOLD).astype(np.uint8)

            dest.write(mask[0][0], indexes=1)

