import numpy as np
from itertools import groupby


def mask2rle(img, width, height):
    img = img.T
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)


def better_mask2rle(img):
    flat = list(img.T.reshape(-1))
    start = 0
    data = []
    prev = 0
    for val, lst in groupby(flat):
        length = len(list(lst))
        if val > 0:
            data.append(prev)
            data.append(length)
        prev = length
        start += length
    if len(data) == 0:
        return "-1"
    return " ".join(map(str, data))


def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height).T


def blank_mask(width, height):
    return np.zeros((width, height))
