import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.torch import ToTensor
import os

from sklearn.model_selection import StratifiedKFold
import cv2
from tqdm import tqdm

# StratifiedKFold = None


class TestDataset(Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                Resize(size, size),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return self.num_samples

def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component


def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i - 1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                #                 HorizontalFlip(),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10,  # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                #                 GaussNoise(),
            ]
        )
    list_transforms.extend(
        [
            Resize(size, size),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms


class SIIMDataset(Dataset):
    def __init__(self, df, data_folder, size, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)
        self.gb = self.df.groupby('ImageId')
        self.fnames = list(self.gb.groups.keys())

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        annotations = df[' EncodedPixels'].tolist()
        image_path = os.path.join(self.root, image_id + ".png")
        image = cv2.imread(image_path)
        mask = np.zeros([1024, 1024])
        if annotations[0] != ' -1':
            for rle in annotations:
                mask += run_length_decode(rle)
        mask = (mask >= 1).astype('float32')  # for overlap cases
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.fnames)


def provider(
        fold,
        total_folds,
        data_folder,
        df_path,
        phase,
        size,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=4,
):
    df = pd.read_csv(df_path)
    df = df.drop_duplicates('ImageId')
    df_with_mask = df[df[" EncodedPixels"] != " -1"]
    df_with_mask['has_mask'] = 1
    df_without_mask = df[df[" EncodedPixels"] == " -1"]
    df_without_mask['has_mask'] = 0
    df_without_mask_sampled = df_without_mask.sample(len(df_with_mask))
    df = pd.concat([df_with_mask, df_without_mask_sampled])
    # NOTE: equal number of positive and negative cases are chosen.

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(
        df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    # NOTE: total_folds=5 -> train/val : 80%/20%

    image_dataset = SIIMDataset(df, data_folder, size, mean, std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    return dataloader


def save_validation(basename):
    sample_submission_path = './sample_submission.csv'
    train_rle_path = '/datasets/siim-dataset/train-rle.csv'
    data_folder = "/datasets/processed_1024_train/img/"
    test_data_folder = "/datasets/processed_1024_test/img/"

    batch_size = 4
    device = torch.device("cuda:0")

    testset = provider(
        fold=1,
        total_folds=5,
        data_folder=data_folder,
        df_path=train_rle_path,
        phase="val",
        size=512,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        batch_size=batch_size,
        num_workers=8,
    )

    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    model.eval()
    model.to(device)
    state = torch.load('./best.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    test_probability = []
    test_truth = []
    for i, (batch, mask) in enumerate(tqdm(testset)):

        with torch.no_grad():
            preds = torch.sigmoid(model(batch.to(device)))
            test_probability.append(preds.detach().cpu().numpy().astype(np.float32))
            test_truth.append(mask.numpy().astype(np.float32))

    test_truth = np.concatenate(test_truth)
    test_probability = np.concatenate(test_probability)

    truth_file_path = "{}_truth.npy".format(basename)
    probability_file_path = "{}_probability.npy".format(basename)
    np.save(truth_file_path, test_truth)
    np.save(probability_file_path, test_probability)
    print("Saved into", truth_file_path)
    print("Saved into", probability_file_path)


def save_test(best_threshold=0.5):
    sample_submission_path = './sample_submission.csv'
    train_rle_path = '/datasets/siim-dataset/train-rle.csv'
    data_folder = "/datasets/processed_1024_train/img/"
    test_data_folder = "/datasets/processed_1024_test/img/"
    size = 512
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_workers = 8
    batch_size = 16
    min_size = 3500
    device = torch.device("cuda:0")
    df = pd.read_csv(sample_submission_path)
    testset = DataLoader(
        TestDataset(test_data_folder, df, size, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    model.to(device)
    model.eval()
    state = torch.load('./best.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    encoded_pixels = []
    for i, batch in enumerate(tqdm(testset)):
        preds = torch.sigmoid(model(batch.to(device)))
        preds = preds.detach().cpu().numpy()[:, 0, :, :]  # (batch_size, 1, size, size) -> (batch_size, size, size)
        for probability in preds:
            if probability.shape != (1024, 1024):
                probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(probability, best_threshold, min_size)
            if num_predict == 0:
                encoded_pixels.append('-1')
            else:
                r = run_length_encode(predict)
                encoded_pixels.append(r)
    df['EncodedPixels'] = encoded_pixels
    df.to_csv('submission.csv', columns=['ImageId', 'EncodedPixels'], index=False)


if __name__ == '__main__':
    # save_validation("submit")
    save_test()
