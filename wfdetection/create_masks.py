import os
import glob
import rasterio
import rasterio.mask
from rasterio.features import rasterize
import geopandas as gpd


def create_shapes_bitmask(source_meta, shapes, mask_file_name):
    """
    Creates bit masks for shapes that intersect with raster image coordinates.

    :param source_meta: opened raster image
    :param shapes: iterable with shapes
    :param mask_file_name: file name to store mask on disk
    :return: None
    """
    im_size = (source_meta['height'], source_meta['width'])

    bitmask = rasterize(
        shapes=shapes,
        out_shape=im_size,
        transform=source_meta['transform']
    )

    bitmask_meta = {
        **source_meta,
        'dtype': rasterio.uint8,
        'count': 1
    }

    with rasterio.open('{}.tif'.format(mask_file_name), 'w', **bitmask_meta) as dest:
        dest.write(bitmask, indexes=1)


geojson_path = os.environ.get('GEOJSON_PATH') # path to file with shapes
files_dir = os.environ.get('IMG_DIR') # path to directory with raster images
dest_dir = os.environ.get('DEST_DIR') # path to directory where to store masks

shapes_df = gpd.read_file(geojson_path)
tif_files = sorted(glob.glob('{}/*.tif'.format(files_dir)))

for tif_file in tif_files:
    file_name = os.path.splitext(os.path.basename(tif_file))[0]
    mask_file_name = '{}/{}_mask'.format(dest_dir, file_name)

    with rasterio.open(tif_file) as src:
        shapes_df = shapes_df.to_crs({'init' : src.meta['crs']['init']})

        create_shapes_bitmask(
            source_meta=src.meta,
            shapes=shapes_df.geometry,
            mask_file_name=mask_file_name
        )
