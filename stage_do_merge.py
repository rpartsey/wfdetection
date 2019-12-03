from utils.mask_functions import better_mask2rle, rle2mask



import pandas as pd
from matplotlib import pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

# experiments/smp_unet_sampler_1024accum_bug/best.h5 0.9 3800 do_tta False cut_borders False[change in conf]
our_best = pd.read_csv("./submissions/submission_stage_best_without_tta.csv", dtype=str)

# smp_unet_sampler_alignaug_1024 5.0000000e-01 4.5000000e+03 do_tta True cut_borders from config
best_align = pd.read_csv("./submissions/submission_stage_align.csv", dtype=str)
# smp_unet_sampler_alignaug_1024_div 5.5000001e-01 3.3000000e+03 do_tta True cut_borders from config
best_align_div = pd.read_csv("./submissions/submission_stage_align_div.csv", dtype=str)
# best_align_div = pd.read_csv("./submissions/submission_private_align_aug_div_tta_best.csv", dtype=str)
# experiments/smp_unet_sampler_1024accum_bug/best.h5 5.0000000e-01 3.6000000e+03 do_tta True cut_borders True[change in conf]
accu_bug_best = pd.read_csv("./submissions/submission_stage_best.csv", dtype=str)

best_new_dfs = [accu_bug_best, best_align, best_align_div]

dfs = [our_best] + best_new_dfs
out_values = []
for img_id in tqdm(our_best.ImageId.unique()):
    or_between = None
    skip = False
    for i, df in enumerate(dfs):
        val = df[df.ImageId == img_id]["EncodedPixels"].values[0]
        if val.strip() == "-1":
            skip = True
            break
        mask = (rle2mask(val, 1024, 1024) > 0).astype(np.uint8)
        if or_between is None:
            or_between = np.zeros_like(mask, dtype=np.bool)
        or_between |= mask > 0.5
    if skip:
        out_values.append([img_id, "-1"])
    else:
        out_values.append([img_id, better_mask2rle(or_between * 1 * 255)])

out_best_values = out_values.copy()

out_values_df = pd.DataFrame(out_best_values, columns=['ImageId', 'EncodedPixels'])
print((out_values_df.EncodedPixels == "-1").sum())
out_values_df.to_csv("stage_strange.csv", index=None)
