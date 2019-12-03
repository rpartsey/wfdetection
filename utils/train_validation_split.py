import random

import pandas as pd
import numpy as np


def random_train_val_split(csv_file, validation_frac, random_state=None):
    """
    Splits the csv_file into train and validation parts. 
    
    Args:
        - csv_file(pandas Dataframe): csv file with patient ID and encoded pixels
        - validation_frac(float): percentage of the data, dedicated to validation set
        - random_state(int): integer to fix random seed
    Returns:
        - tuple(train_part, validation_part): tuple with two Dataframes, train and validation, which contain only patients IDs.

    Example:
        train_val_random_split(labels, 0.2, 44) # 20% of the data to validation 
    """
    random.seed(random_state)
    np.random.seed(random_state)

    imgs_with_pneumothorax = csv_file[csv_file['mask_exists']]
    imgs_without_pneumothorax = csv_file[~csv_file['mask_exists']]

    val_imgs_with_pneumothorax = imgs_with_pneumothorax.sample(frac=validation_frac, random_state=random_state)
    imgs_with_pneumothorax.drop(val_imgs_with_pneumothorax.index, inplace=True)

    val_imgs_without_pneumothorax = imgs_without_pneumothorax.sample(frac=validation_frac, random_state=random_state)
    imgs_without_pneumothorax.drop(val_imgs_without_pneumothorax.index, inplace=True)

    val_part = pd.concat([val_imgs_with_pneumothorax, val_imgs_without_pneumothorax])
    val_part.drop_duplicates(inplace=True)
    train_part = pd.concat([imgs_with_pneumothorax, imgs_without_pneumothorax])
    train_part.drop_duplicates(inplace=True)

    return train_part, val_part


if __name__ == '__main__':
    csv_file = pd.read_csv('/datasets/siim-dataset/train-rle.csv')

    d1, d2 = random_train_val_split(csv_file, 0.2, 44)

    print(d1.head())
    print(d2.head())
