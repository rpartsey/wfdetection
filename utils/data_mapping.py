# Convert DICOM to PNG via openCV
import os
import shutil
import glob

from tqdm import tqdm
import pydicom
import cv2
import pandas as pd

from utils.mask_functions import rle2mask, blank_mask

WIDTH, HEIGHT = 1024, 1024
DATA_ROOT = "/datasets"



def id_from_filename(filename):
    return os.path.splitext(filename)[0]


def _iterate_dcm_dir(in_dir, prefix="*/*/*"):
    if not in_dir.endswith("/"):
        in_dir += "/"
    # TODO: change here
    for fp in glob.glob(in_dir + '{}.dcm'.format(prefix)):
        yield fp


def read_dcm(fp):
    """
    Read dcom file
    :param fp: str: dcom file path
    :return: np.ndarray: img as numpy array
    """
    ds = pydicom.read_file(fp)  # read dicom image
    img = ds.pixel_array  # get image array
    return img


def dcm2png_file(dcm_path, out_dir, resize=None):
    """
    Read `dcim image` and writes corresponding .png
    file to `out_dir`
    :param dcm_path: str: path to dcim file
    :param out_dir: str: path to out dir
    :param resize: tuple(int, int): if provided,
            saves resized version of img
    :return: NoneType: None object
    """
    img = read_dcm(dcm_path)
    if resize:
        img = cv2.resize(img, resize)
    dcm_fn = os.path.basename(dcm_path)
    sample_id = os.path.splitext(dcm_fn)[0]
    img_fn = "{}.png".format(sample_id)
    img_path = os.path.join(out_dir, img_fn)
    cv2.imwrite(img_path, img)


def dcm2png(in_dir, out_dir, resize=None, prefix=None, v=0):
    """
    Converts all dcim images to png images
    :param in_dir: str: path to dir with input dcim images
    :param out_dir: str: path to out dir where to save .png images
    :param resize: tuple(int, int): if provided saves
            resized images
    :param v: int: verbosity, if 1 shows `tqdm`
    :return: NoneType: None object
    """
    it = _iterate_dcm_dir(in_dir, prefix=prefix)
    if v:
        it = tqdm(it)
    for fp in it:
        img = read_dcm(fp)
        out_fname = "{}.png".format(os.path.basename(os.path.splitext(fp)[0]))
        out_fp = os.path.join(out_dir, out_fname)
        if resize:
            img = cv2.resize(img, resize)
        cv2.imwrite(out_fp, img)


def csv2mask(in_path, out_dir, resize=None, v=0):
    """
    Converts given .csv annotations, creates masks in out_dir
    :param in_path: str: file path to `.csv`
    :param out_dir: str: path to out dir where to save .png images
    :param resize: tuple(int, int): if provided saves
            resized images
    :param v: int: verbosity, if 1 shows `tqdm`
    :return: NoneType: None object
    """
    df = pd.read_csv(in_path, dtype=str, sep=", ")
    df = df.groupby('ImageId')["EncodedPixels"].apply(list)
    it = df.items()
    if v:
        it = tqdm(it)
    for image_id, row in it:
        mask = blank_mask(WIDTH, HEIGHT)
        for i, rle_mask in enumerate(row, 1):
            if rle_mask == "-1":
                continue
            i_mask = rle2mask(rle_mask, WIDTH, HEIGHT)
            mask[i_mask > 125] = i
        out_fname = "{}.png".format(image_id)
        out_fp = os.path.join(out_dir, out_fname)
        if resize:
            mask = cv2.resize(mask, resize)
        cv2.imwrite(out_fp, mask)


def ask_delete_path(path):
    question = "Path [{}] exists, do you want to delete it? [y/n]:".format(path)
    answer = input(question)
    while answer != "y" and answer != "n":
        answer = input(question)
    return answer


def map_dataset(dim=1024, dtype="train", v=0):
    """
    Creates dataset representation in dir `$DATA_ROOT/processed_$dim_$dtype`
    :param dim: tuple(int, int): size of created representation
    :param dtype: str: train or test. important for creation masks or not
    :param v: int: verbosity, if 1 shows `tqdm`
    :return:
    """
    if dtype not in ["train", "test", "private", "train-2"]:
        raise ValueError("dtype=`{}` not in possible values ['test', 'train']".format(dtype))
    root = os.path.join(DATA_ROOT, "processed_{}_{}".format(dim, dtype))
    in_img = os.path.join(DATA_ROOT, "siim-dataset", "dicom-images-{}".format(dtype))
    if dtype == "train-2":
        csv_path = os.path.join(DATA_ROOT, "siim-dataset", "stage_2_train.csv")
    else:
        csv_path = os.path.join(DATA_ROOT, "siim-dataset", "train-rle.csv")
    if not os.path.exists(in_img):
        raise ValueError("Path for dataset input images doesn't exist: [{}]".format(in_img))
    if os.path.exists(root):
        ans = ask_delete_path(root)
        if ans == "n":
            return
        shutil.rmtree(root)
    os.mkdir(root)
    img_dir = os.path.join(root, "img")
    os.mkdir(img_dir)
    prefix = "*/*/*"
    if dtype == "private":
        prefix = "*"
    dcm2png(in_img, img_dir, resize=(dim, dim), prefix=prefix, v=v)

    if dtype == "train" or dtype == "train-2":
        mask_dir = os.path.join(root, "mask")
        os.mkdir(mask_dir)
        csv2mask(csv_path, mask_dir, resize=(dim, dim), v=v)


if __name__ == "__main__":
    map_dataset(dim=1024, dtype="train-2", v=1)
    map_dataset(dim=256, dtype="train-2", v=1)
