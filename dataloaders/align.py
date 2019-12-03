from keras.models import load_model
import cv2
from scipy import ndimage
import numpy as np
from skimage import measure


def crop_segmentation(mask, *others, width=512, height=512, extra_space=0.1):
    '''
    Crop using `mask` as input. `others` are optional arguments that will be croped using `mask`
    as reference.
    '''
    # Declare structure used in morphotology opening
    morph_structure = np.ones((11, 11))

    # Binarize mask
    mask_bin = np.squeeze(mask) > 0.5

    # Use morphology opening to reduce small structures detected.
    mask_bin = ndimage.morphology.binary_opening(mask_bin, morph_structure)

    mask_bin_labeled = measure.label(mask_bin, background=0)
    mask_bin = np.zeros_like(mask_bin)
    unique_values, counts = np.unique(mask_bin_labeled, return_counts=True)
    for i in np.argsort(counts)[-3:]:
        val = unique_values[i]
        if val == 0:
            continue
        mask_bin[mask_bin_labeled == val] = 1

    morph_structure = np.ones((22, 22))
    mask_bin = ndimage.morphology.binary_closing(mask_bin, morph_structure)


    # Squeeze horizontal and vertical dimention to find where mask begins and ends
    mask_bin_hor = mask_bin.any(axis=0)
    mask_bin_ver = mask_bin.any(axis=1)

    # Find index of first and last positive pixel
    xmin, xmax = np.argmax(mask_bin_hor), len(mask_bin_hor) - np.argmax(mask_bin_hor[::-1])
    ymin, ymax = np.argmax(mask_bin_ver), len(mask_bin_ver) - np.argmax(mask_bin_ver[::-1])

    # Add extra space
    xextra = int((xmax - xmin) * extra_space)
    yextra = int((ymax - ymin) * extra_space)
    xmin -= xextra
    xmax += xextra
    ymin -= yextra
    ymax += yextra

    # We will use affine transform to crop image. It will deal with padding image if necessary
    # Note: `pts` will follow a L shape: top left, bottom left and bottom right
    # For details see: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#affine-transformation
    pts1 = np.float32([[xmin, ymin], [xmin, ymax], [xmax, ymax]])
    pts2 = np.float32([[0, 0], [0, height], [width, height]])
    M = cv2.getAffineTransform(pts1, pts2)

    # Crop mask
    mask_crop = cv2.warpAffine(mask_bin.astype(np.float), M, (height, width), flags=cv2.INTER_AREA, borderValue=0)

    M_inv = cv2.invertAffineTransform(M)
    inverse = lambda x: cv2.warpAffine(x, M_inv, (height, width))

    if len(others) > 0:
        # Crop others
        others_crop = tuple(
            cv2.warpAffine(np.squeeze(other), M, (height, width), flags=cv2.INTER_AREA, borderValue=0) for other in
            others)

        return (mask_crop,) + others_crop, inverse
    else:
        return mask_crop, inverse


class AlignTransform(object):
    def __init__(self, unet_path='unet_lung_seg.hdf5'):
        self.unet = load_model(unet_path, compile=False)

    def __call__(self, x, *args):
        orig_w, orig_h = x.shape
        width, height = 512, 512
        x_orig = x.copy()
        x = cv2.resize(x, (width, height), interpolation=cv2.INTER_AREA)

        # Normalize
        x = np.float32(x) / 255.

        # Add channel axis
        x = x[np.newaxis, ..., np.newaxis]

        mask = self.unet.predict(x)[0, ..., 0]
        mask = cv2.resize(mask, (orig_w, orig_h))
        out, _ = crop_segmentation(mask, x_orig, *args, width=orig_w, height=orig_h)
        return out[1:]
