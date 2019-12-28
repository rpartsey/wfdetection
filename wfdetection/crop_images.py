import glob
import os
import numpy as np
from shapely.geometry import Polygon
import rasterio
import rasterio.mask
from rasterio.windows import Window
import geopandas as gpd


def generate_window_coords(raster_h, raster_w, window_h, window_w):
    all_dims_are_even = raster_h % 2 == 0 and raster_w % 2 == 0 and window_h % 2 == 0 and window_w % 2 == 0
    assert all_dims_are_even, 'Assume raster height and width and window height and width are even numbers'

    if raster_h % window_h == 0:
        start_row_coord = 0
        stop_row_coord = raster_h
    else:
        delta_h = ((raster_h // window_h + 1) * window_h - raster_h) // 2
        start_row_coord = -delta_h
        stop_row_coord = raster_h + delta_h

    if raster_w % window_w == 0:
        start_col_coord = 0
        stop_col_coord = raster_w
    else:
        delta_w = ((raster_w // window_w + 1) * window_w - raster_w) // 2
        start_col_coord = -delta_w
        stop_col_coord = raster_w + delta_w

    rows = np.arange(start_row_coord, stop_row_coord, window_h)
    cols = np.arange(start_col_coord, stop_col_coord, window_w)

    rows, cols = np.meshgrid(rows, cols, indexing='ij')

    return list(zip(rows.ravel(), cols.ravel()))


def build_coords_polygon(i, j, h, w, transform):
    """
    Returns Polygon representing rectangle with (i, j) - top left corner, height h and width w.

    :param i: row index top left corner
    :param j: column index
    :param h: height
    :param w: width
    :param transform: functions that transforms index location to geo coordinates
    :return:  shapely.geometry.Polygon
    """

    points = [
        (i, j),
        (i + h, j),
        (i + h, j + w),
        (i, j + w)
    ]
    return Polygon([transform(*p) for p in points])


def cover_shapes(shapes, windows_coords, window_h, window_w, pixel_loc_to_coords):
    """
    Returns list of (Window, window_index) tuples.

    For each window coordinates builds shapely Polygon and checks if it intersects with any shape.
    If so, builds rasterio Window for such polygon.

    :param shapes: iterable with shapely shapes
    :param windows_coords: coordinates of top left corner
    :param window_h: window height
    :param window_w: window width
    :param pixel_loc_to_coords: functions that transforms index location to geo coordinates
    :return: list of tuples (Window, window_index)
    """

    windows = []
    for row, col in windows_coords:
        poly = build_coords_polygon(row, col, window_h, window_w, pixel_loc_to_coords)
        if any(shape.intersects(poly) for shape in shapes):
            windows.append((
                Window(
                    col_off=col,
                    row_off=row,
                    width=window_w,
                    height=window_h
                ),
                f'_{row // window_h}_{col // window_w}'
            ))
    return windows


def save_cropped_rasters(source, file_name, dest_dir, windows):
    """
    Stores cropped images on disc.

    :param source: opened raster
    :param file_name: raster file name
    :param dest_dir: path to directory where to store cropped images
    :param windows: windows representing rectangles that intersect with shapes
    :return: None
    """

    for window, index in windows:
        meta = {
            **source.meta,
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)
        }
        with rasterio.open(f'{dest_dir}/{file_name}{index}.tif', 'w', **meta) as dest:
            dest.write(src.read(window=window, boundless=True, fill_value=0))


geojson_path = os.environ.get('GEOJSON_PATH')
files_dir = os.environ.get('IMG_DIR')
dest_dir = os.environ.get('DEST_DIR')
window_h = os.environ.get('WINDOW_H')
window_w = os.environ.get('WINDOW_W')

if not (geojson_path and files_dir and dest_dir and window_h and window_w):
    raise ValueError(
        "'GEOJSON_PATH' and 'IMG_DIR' and 'DEST_DIR' and 'WINDOW_H' and 'WINDOW_W' env variables must be specified"
    )

shapes_df = gpd.read_file(geojson_path)
tif_files = sorted(glob.glob(f'{files_dir}/*.tif'))

for tif_file in tif_files:
    with rasterio.open(tif_file) as src:
        shapes_df = shapes_df.to_crs({'init': src.meta['crs']['init']})
        shapes = shapes_df['geometry']

        windows_coords = generate_window_coords(
            raster_h=src.meta['height'],
            raster_w=src.meta['width'],
            window_h=window_h,
            window_w=window_w
        )
        windows = cover_shapes(
            shapes=shapes,
            windows_coords=windows_coords,
            window_h=window_h,
            window_w=window_w,
            pixel_loc_to_coords=src.xy
        )

        file_name = os.path.splitext(os.path.basename(tif_file))[0]
        save_cropped_rasters(src, file_name, dest_dir, windows)