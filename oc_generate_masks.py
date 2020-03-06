import os
import glob
import rasterio
import rasterio.mask
from rasterio.features import rasterize
import geopandas as gpd

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

image_tifs = sorted(glob.glob(train_1_images_glob))
label_tifs = sorted(glob.glob(train_1_labels_glob))

for indx, (image_path, label_path) in enumerate(zip(image_tifs, label_tifs)):
    shapes_df = gpd.read_file(label_path)
    file_name = os.path.basename(image_path)
    mask_path = os.path.join(masks_root, 'train_1', file_name)

    print(indx, mask_path)
    with rasterio.open(image_path) as source:
        shapes_df = shapes_df.to_crs({'init': source.meta['crs']['init']})

        im_size = (source.meta['height'], source.meta['width'])

        bitmask = rasterize(
            shapes=shapes_df.geometry,
            out_shape=im_size,
            transform=source.meta['transform']
        )

        bitmask_meta = {
            **source.meta,
            'dtype': rasterio.uint8,
            'count': 1
        }

        with rasterio.open(mask_path, 'w', **bitmask_meta) as dest:
            dest.write(bitmask, indexes=1)