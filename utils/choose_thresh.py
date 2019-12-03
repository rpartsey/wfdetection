import os

import copy
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MODELS
from utils.train_validation_split import random_train_val_split
from dataloaders.binary_dataloader import BinaryLoader
from dataloaders.transformations import create_transform
from utils.threshold_choosing import choose_threshold

DEVICE = os.getenv('DEVICE', None)
if DEVICE is None:
    raise ValueError("please specify the device in OS.ENV using "
                     "`>> DEVICE=cuda:0 python {file you running}` or doing"
                     "`export DEVICE=cuda:0\npython{file you running}`"
                     " so we can easily distribute GPUs")


def load_data(config):
    df = pd.read_csv(config["csv_path"])
    df = df.groupby("ImageId").first().reset_index()
    df["mask_exists"] = df[' EncodedPixels'] != ' -1'
    # df = df.sample(1000)

    # and leave only ids which really have labels
    valid_file_names = os.listdir(os.path.join(config['data_path'], 'mask'))
    valid_image_ids = set(x.strip(".png") for x in valid_file_names)
    cut_df = df[df['ImageId'].isin(valid_image_ids)].reset_index(drop=True)

    print(len(cut_df))
    train_csv, val_csv = random_train_val_split(cut_df,
                                                config["validation_fraction"],
                                                config["random_state"])

    transform_config = config["transformations"]
    image_transform = create_transform(transform_config["image"])
    mask_transform = create_transform(transform_config["mask"])

    train_data = BinaryLoader(train_csv, config['data_path'],
                              image_transform=image_transform,
                              mask_transform=mask_transform)
    val_data = BinaryLoader(val_csv, config['data_path'],
                            image_transform=image_transform,
                            mask_transform=mask_transform)
    return train_data, val_data


def main(config_path, weight_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    mconfig = copy.deepcopy(config["model"])
    mtype = mconfig["type"]
    mconfig.pop("type")
    model_class = MODELS.get(mtype, None)
    if model_class is not None:
        model = model_class(device=DEVICE,
                            freeze=False,
                            **mconfig)
    else:
        raise NotImplementedError("model not Found")

    print("Best")
    state = torch.load(weight_path)
    model.load_state_dict(state["model"])
    model.to(DEVICE)

    train_data, val_data = load_data(config)

    for data, name in ((train_data, "train"), (val_data, "val")):
        tq = tqdm(
            DataLoader(data,
                       batch_size=32))

        collected_out = []
        collected_mask = []
        for i, (imgs, masks, _, _) in enumerate(tq):
            imgs = imgs.to(DEVICE)
            masks = masks.numpy().tolist()

            with torch.no_grad():
                logits = model(imgs)
                if logits.size(1) == 2:
                    softmax_logits = logits.softmax(1).cpu()
                    logits = softmax_logits[:, 1:, :, :].numpy().tolist()
                else:
                    sigmoid_logits = torch.sigmoid(logits).cpu()
                    logits = sigmoid_logits[:, 0:, :, :].numpy().tolist()

            collected_mask += masks
            collected_out += logits
        collected_out = torch.tensor(collected_out)
        collected_mask = torch.tensor(collected_mask)

        print("Name:", name, choose_threshold(collected_out, collected_mask))


if __name__ == '__main__':
    config_path = "torchhub_unet.yaml"
    weight_path = "experiments/torchhub_unet_imbsampler_histnorm/best_torchhub_unet_histnorm.h5"
    main(config_path=config_path, weight_path=weight_path)
