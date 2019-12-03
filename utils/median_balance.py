#!/usr/bin/env python

import sys, os
import glob

import numpy as np
import cv2
from tqdm import tqdm

np.set_printoptions(precision=15)


def main(folder):
    print("[FOLDER {}]".format(folder))
    files = glob.glob(os.path.join(folder, "*.png"))
    nfiles = len(files)

    lbl_counts = {}

    for f in tqdm(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = (img > 0).astype(np.uint8)

        id, counts = np.unique(img, return_counts=True)
        # normalize on image
        counts = counts / float(sum(counts))
        for i in range(len(id)):
            if id[i] in lbl_counts.keys():
                lbl_counts[id[i]] += counts[i]
            else:
                lbl_counts[id[i]] = counts[i]

    # normalize on training set
    for k in lbl_counts:
        lbl_counts[k] /= nfiles

    print("##########################")
    print("class probability:")
    for k in lbl_counts:
        print("%i: %f" % (k, lbl_counts[k]))
    print("##########################")

    # normalize on median freuqncy
    med_frequ = np.median(list(lbl_counts.values()))
    lbl_weights = {}
    for k in lbl_counts:
        lbl_weights[k] = med_frequ / lbl_counts[k]

    print("##########################")
    print("median frequency balancing:")
    for k in lbl_counts:
        print("%i: %f" % (k, lbl_weights[k]))
    print("##########################")


if __name__ == '__main__':
    folder = sys.argv[1]
