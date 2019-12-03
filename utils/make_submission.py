"""
Module for creating submissions.
"""

import os
import shutil

import pandas as pd
import cv2
from skimage import measure
import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


from .mask_functions import better_mask2rle
from .data_mapping import map_dataset, id_from_filename
from .data_mapping import ask_delete_path, DATA_ROOT



DEVICE = os.getenv('DEVICE', None)
if DEVICE is None:
    raise ValueError("please specify the device in OS.ENV using "
                     "`>> DEVICE=cuda:0 python {file you running}` or doing"
                     "`export DEVICE=cuda:0\npython{file you running}`"
                     " so we can easily distribute GPUs")



def create_submittion_mask_job(model_run_path, pred_name, postprocessing_config):
    preds = np.load(os.path.join(model_run_path, pred_name), allow_pickle=True)

    # preds.shape 1, W, H
    image_id = id_from_filename(pred_name)

    preds = postprocessing(preds, postprocessing_config)

    # Setting -1 value if there was no instances found
    # Going mask-by-mask
    unique_values = np.unique(preds)
    if len(unique_values) == 1 and unique_values[0] == 0:
        return [[image_id, " -1"]]

    mask = preds > 0
    mask = mask * 1 * 255

    return [[image_id, " " + better_mask2rle(mask)]]


def create_submission(model_run_path, result_filename_appendix, postprocessing_config, v=0):
    """
    Creates submission `.csv`.
    :param path_to_masks: Path to folder with `.png` masks.
    :param result_filename_appendix: Appendix to the submission filename.
    """
    it = os.listdir(model_run_path)
    print()

    if v:
        it = tqdm(it, position=0, desc=".csv creation")

    raw_submissions = Parallel(n_jobs=8, backend='multiprocessing')(delayed(
        create_submittion_mask_job)(model_run_path, pred_name, postprocessing_config) for pred_name in it)
    submissions = [mask for mask_list in raw_submissions for mask in mask_list]

    # Creating pd.DataFrame from list of lists
    submission_df = pd.DataFrame(submissions, columns=['ImageId', 'EncodedPixels'])

    # Creating folder for submissions `.csv`
    if not os.path.exists('submissions'):
        os.makedirs('submissions')

    # Saving results to .`csv`
    submission_df.to_csv(os.path.join('submissions', 'submission_{}.csv'.format(result_filename_appendix))
                         , index=False)

    print('Submission is saved to submission_{}.csv successfully!!!'.format(result_filename_appendix))

# TODO: fix
def mult(*args):
    res = 1
    for el in args:
        res *= el
    return res


def forward_model(model, img, tta=False):
    saved = None
    if model.cut_borders:
        img, _, saved = model.action_cut_borders(img, np.zeros_like(img))

    if tta:
        img_f = img.copy()
        img_f = img[:, ::-1].copy()

    img = model.transformations["image"](img)
    if tta:
        img_f = model.transformations["image"](img_f)
        img_f = img_f.unsqueeze(0)
        img_f = img_f.to(DEVICE)
    # Create batch
    img = img.unsqueeze(0)
    img = img.to(DEVICE)

    with torch.no_grad():
        logits = model.predict(img)
        if tta:
            logits_f = model.predict(img_f)

    softmax_logits = logits.detach().cpu().numpy()
    if tta:
        softmax_logits_f = logits_f.detach().cpu().numpy()
        softmax_logits = np.stack([softmax_logits, softmax_logits_f[:, :, :, ::-1]]).mean(0)

    if model.cut_borders:
        softmax_logits[0, 0, :, :] = model.uncut_borders(softmax_logits[0, 0, :, :], saved)
    # B, C, W, H
    # 1, 2, W, H
    return softmax_logits


def forward_model_path(model, img_path, tta=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return forward_model(model, img, tta=tta)


def model_run_test(models, dim, model_run, tta=False, v=0):
    print("TTA", tta)
    in_root = os.path.join(DATA_ROOT, "processed_{}_private".format(dim), "img")
    # model_run = "submissions/model-run-{}".format(dim)
    print("[", model_run, "]")
    tq = os.listdir(in_root)

    if v:
        tq = tqdm(tq, desc="model run")
    for fn in tq:
        preds = []
        fp = os.path.join(in_root, fn)  # ToDo: add TTA
        for m in models:  # avg the predictions of several models
            preds.append(forward_model_path(m, fp, tta=tta))
        # pred.shape -> N, 1, W, H
        preds = np.concatenate(preds)
        if preds.shape[2] == 2:
            raise NotImplementedError("Doesn't work")
            # p = np.expand_dims(np.array([np.mean(preds[:, 0, 0], axis=0), np.mean(preds[:, 0, 1], axis=0)]), axis=0)
        # else:
        p = np.mean(preds, axis=0)
        # p.shape -> 1, W, H
        p.dump(os.path.join(model_run, id_from_filename(fn) + '.dat'))


def postprocessing(preds, config): #add all quick fixes options
    """
    :param preds: np.ndarray shape (1, W, H)
    :param min_area_removal:
    :return:
    """
    # 1. choose class
    def class_threshhold(preds, thr=0.5):
        """
        Make binary pred by given theshold.
        Default is 0.5 what is `argmax`
        """
        print("Thresholding", thr)
        preds = (preds > thr).astype(np.uint8)
        return preds

    preds = class_threshhold(preds, thr=config["class_threshold"])

    # 2. resize
    def resize(preds, w=1024, h=1024):
        dim = int(mult(*preds.shape) ** 0.5)
        preds = preds.reshape(dim, dim)
        if preds.shape[0] != w:
            print("Resizing from", preds.shape, "to", (w, h))
            preds = cv2.resize(preds, (w, h), interpolation=cv2.INTER_NEAREST)
            preds = preds > 0
        return preds.astype(np.uint8)

    preds = resize(preds)

    # 3. various quick fixes
    def remove_min_area(preds, min_area=100.0):
        print("Removing min area", min_area)
        num_component, component = cv2.connectedComponents(preds.astype(np.uint8))
        predict = np.zeros_like(preds)
        num = 0
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_area:
                predict[p] = 1
                num += 1
        return predict

    def erode_mask(preds):
        print("Eroding masks")
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(preds, kernel, iterations=2)
        return erosion

    if config['erode_mask']:
        preds = erode_mask(preds)

    if config["min_area_subtraction"]["use"]:
        preds = remove_min_area(preds, min_area=config["min_area_subtraction"]["area"])

    return preds


def main(model, dim, appendix, config, v=0):
    """
    Does the whole process of submittion:
    1. Create imgs from dcim of correct `DIM`
    2. Run model on this images
    3. Create final submittion .csv
    :param model: torch.Module: model object
    :param model_channels: int: number of channels on input of img
    :param dim: int: size of images
    :param appendix: str: appendix for `.csv` file name
    :param v: int: verbosity
    :return: NoneType: None object
    """
    if not os.path.exists("submissions"):
        os.mkdir("submissions")
    map_dataset(dim, dtype="private", v=v)

    model_run = os.path.join("submissions", "model-run-{}-{}".format(dim, appendix))
    answer_ok = True
    if os.path.exists(model_run):
        answer_ok = ask_delete_path(model_run) == "y"
        if answer_ok:
            shutil.rmtree(model_run)
    os.makedirs(model_run, exist_ok=True)
    if answer_ok:
        model_run_test(model, dim, model_run=model_run, tta=config.get("do_tta", False), v=v)

    create_submission(model_run, appendix, config['postprocessing'], v=v)


if __name__ == '__main__':
    # PATH_TO_MASKS = '/datasets/processed/mask'
    model = None  # load 'weights/unet_baseline_weighted_crossentropy0.233407.pth'
