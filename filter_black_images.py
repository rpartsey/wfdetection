import os
import glob

import numpy as np
import pandas as pd

import rasterio


experiments_root = '/datasets/rpartsey/open_cities/experiments_data'

cropped_images_root = os.path.join(experiments_root, 'cropped_images')
cropped_masks_root = os.path.join(experiments_root, 'cropped_masks')

dest_images_path = os.path.join(cropped_images_root, 'train_1')
dest_masks_path = os.path.join(cropped_masks_root, 'train_1')

images_glob = os.path.join(dest_images_path, '*')
masks_glob = os.path.join(dest_masks_path, '*')

images = sorted(glob.glob(images_glob))
masks = sorted(glob.glob(masks_glob))

df = pd.DataFrame()
df['images'] = images
df['masks'] = masks

black_list = []
max_values = []

for indx, (image_path, _) in df.iterrows():
    with rasterio.open(image_path) as img_src:
        img = img_src.read()[:3]

    if not img.any():
        black_list.append(indx)
        max_values.append(img.max())

    if indx % 5000 == 0:
        print(indx)

print("Max of all black images is: ", max(max_values))
print("Num of black images: ", len(black_list))

black_indx = np.array(black_list)

white_mask = np.ones(df.shape[0]).astype(bool)
white_mask[black_indx] = False

print("Saving dataframe.")
white_df = df[white_mask]
white_df.to_csv('/datasets/rpartsey/open_cities/experiments_data/white_images.csv', index=False)