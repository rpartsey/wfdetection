#!/usr/bin/env python
# coding: utf-8

# In[6]:




# In[8]:




# In[17]:


import os
import pandas as pd
from utils.data_mapping import id_from_filename
import cv2
import yaml


# In[10]:


data = "/datasets/processed_256_test"


# In[11]:


img_fp = list(os.listdir(os.path.join(data, "img")))
ids = [id_from_filename(fn) for fn in img_fp]
img_fp = [os.path.join(data, "img", fn) for fn in img_fp]


# In[ ]:


from dataloaders.transformations import create_transform
best_model_path = "experiments/r2_unet_classifier_and_segm_back_tp_fp_fn_focal_dicelast_r2_unet_classifier_and_segm_back_tp_fp_fn_focal_dice.h5"
best_model_dir = "experiments/r2_unet_classifier_and_segm_back_tp_fp_fn_focal_dice_norm_aug"
with open(os.path.join(best_model_dir, 'config.yaml'), 'r') as f:
    config = yaml.load(f)
transform_config = config["transformations"]
transform = create_transform(transform_config["image"])


# In[18]:

import torch
import torch.nn
from models.classifier_unet import UNet
model = torch.nn.DataParallel(UNet()).to("cuda:0")
model_dict = torch.load(best_model_path)["model"]
model.load_state_dict(model_dict)
model.eval()


# In[19]:




# In[16]:
import numpy as np

full = 1024*1024
submissions = []
from tqdm import tqdm
for img_id, single_img_fp in zip(ids, tqdm(img_fp)):
    img = cv2.imread(single_img_fp, cv2.IMREAD_GRAYSCALE)
    horizontal_img = cv2.flip( img, 0 )
    vertical_img = cv2.flip( img, 1 )
    both_img = cv2.flip( img, -1 )
    imgs = [img, horizontal_img, vertical_img, both_img]
    imgs = [transform(img).numpy() for img in imgs]
    imgs = torch.from_numpy(np.array(imgs)).float()
    # print(imgs.shape)
    with torch.no_grad():
        _, out = model.forward(imgs)
        out = out.argmax(1).float()#.mean()
        # print(out)
        out = out.mean()
    out_from_nn = out # zero or one

    pred = "{} 1".format(full)
    if out_from_nn <= 0.51: # zero
        pred = "-1"
    submissions.append([img_id, pred])

submission_df = pd.DataFrame(submissions, columns=['ImageId', 'EncodedPixels'])


# In[ ]:





# In[15]:


out_fn = "bodya_classifier.csv"
submission_df.to_csv(out_fn, index=False)


# In[ ]:



