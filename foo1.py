import os
import glob
import rasterio
import rasterio.mask
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from dataloaders.binary_dataloader import BinaryLoader
from dataloaders.augmentation import StandartAugmentation


augmentations = StandartAugmentation(p=0.9, size=(1024, 1024))

all_df = pd.read_csv('/datasets/rpartsey/open_cities/output/chip/1/chips/open_cities_all.csv')
train_aug_data = BinaryLoader(all_df, '/foo', aug_transform=augmentations)

chips_per_scene = 10

base_dir = '/datasets/rpartsey/open_cities/output/cropped_chips'

print(len(train_aug_data))
for idx, (x, y, meta) in enumerate(train_aug_data):

    file_name = os.path.basename(meta['img_id']).split('.')[0]
    image_name = '{file_name}_{idx}'.format(file_name=file_name, idx=idx % chips_per_scene)
    mask_name = '{}_mask_exists'.format(image_name) if np.sum(y) > 0 else image_name

    image_path = os.path.join(base_dir, 'images', '{}.png'.format(image_name))
    mask_path = os.path.join(base_dir, 'labels', '{}.png'.format(mask_name))

    cv2.imwrite(image_path, x)
    cv2.imwrite(mask_path, y)