import os
import glob
import rasterio
import rasterio.mask
from rasterio.windows import Window
import numpy as np


open_cities_root = '/datasets/rpartsey/open_cities'
train_1_root = os.path.join(open_cities_root, 'train_tier_1')
train_2_root = os.path.join(open_cities_root, 'train_tier_2')


train_1_images_glob = os.path.join(train_1_root, '???', '??????', '*.tif')
train_2_images_glob = os.path.join(train_2_root, '???', '??????', '*.tif')

train_1_labels_glob = os.path.join(train_1_root, '???', '??????-labels', '*.geojson')
train_2_labels_glob = os.path.join(train_2_root, '???', '??????-labels', '*.geojson')

experiments_root = '/datasets/rpartsey/open_cities/experiments_data'
masks_root = os.path.join(experiments_root, 'masks')
cropped_images_root = os.path.join(experiments_root, 'cropped_images')
cropped_masks_root = os.path.join(experiments_root, 'cropped_masks')


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



WINDOW_H = 1024
WINDOW_W = 1024

for indx, image_path in enumerate(glob.glob(os.path.join(train_1_root, '???', '??????', '*.tif'))):
    file_name = os.path.basename(image_path)
    print(indx, file_name)

    with rasterio.open(image_path) as src:
        print(src.meta['height'], src.meta['width'])

        window_offsets = generate_window_offsets(
            image_h=src.meta['height'],
            image_w=src.meta['width'],
            window_h=WINDOW_H,
            window_w=WINDOW_W
        )

        for row_off, col_off in window_offsets:
            window = Window(
                col_off=col_off,
                row_off=row_off,
                width=WINDOW_W,
                height=WINDOW_H
            )

            dest_meta = {
                **src.meta,
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, src.transform)
            }

            dest_file_name = '{}_{}_{}.tif'.format(file_name.split('.')[0], (row_off // WINDOW_H) + 1, (col_off // WINDOW_W) + 1)
            dest_path = os.path.join(cropped_images_root, 'train_1', dest_file_name)

            with rasterio.open(dest_path, 'w', **dest_meta) as dest:
                dest.write(src.read(window=window, boundless=True, fill_value=0))