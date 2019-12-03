import yaml
import os
import shutil
import random
import time

import pandas as pd

import torch
from torch.utils.data import DataLoader

import numpy as np

from dataloaders.transformations import create_transform
from dataloaders.augmentation import StandartAugmentation, FrogAugmentation
from dataloaders.binary_dataloader import BinaryLoader, cutBorders, uncut
from dataloaders import sample_tensorfboard_images
from dataloaders.sampler import ImbalancedDatasetSampler, BalanceClassSampler
from utils.train_validation_split import random_train_val_split
from utils.early_stopping import EarlyStopping

from models import MODELS, descriminator
from losses import LOSSES
from schedulers import SCHEDULERS
from optimizers import OPTIMIZERS


torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True


class Config(object):
    CONFIG_FILENAME = "config.yaml"
    LOGDIRNAME = "logdir"
    BEST_MODEL_NAME = "best_desc.h5"
    LAST_MODEL_NAME = "best_desc.h5"
    MODEL_KEY = "model"

    def __init__(self, config_instance):
        print("Loading config -->", config_instance.exp_dir)
        self.device = config_instance.device
        self.config_instance = config_instance
        self.model = descriminator.Descriminator().to(self.device)
        self.exp_dir = self.config_instance.exp_dir


    @property
    def logdir(self):
        return os.path.join(self.exp_dir, self.LOGDIRNAME)


    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def load_best_model_state(self):
        path = os.path.join(self.exp_dir, self.BEST_MODEL_NAME)
        self.__load_model_from_path(path)

    def save_best_model_state(self, name=None):
        if name is None:
            name = self.BEST_MODEL_NAME
        else:
            name = "{}.h5".format(name)
        path = os.path.join(self.exp_dir, name)
        self.__save_model_to_path(path)

    def load_last_model_state(self):
        path = os.path.join(self.exp_dir, self.LAST_MODEL_NAME)
        self.__load_model_from_path(path)

    def save_last_model_state(self, name=None):
        path = os.path.join(self.exp_dir, self.LAST_MODEL_NAME)
        self.__save_model_to_path(path)

    def __save_model_to_path(self, path):
        model_state = {self.MODEL_KEY: self.model.state_dict()}
        torch.save(model_state, path)

    def load_model_from_path(self, path):
        self.__load_model_from_path(path)

    def __load_model_from_path(self, path):
        print("Loading model from path ->", path)
        model_state = self.__load_model_state(path)
        self.model.load_state_dict(model_state)

    def __load_model_state(self, path):
        state_dict = torch.load(path, map_location=self.device)
        return state_dict[self.MODEL_KEY]

    def load_optimizer(self):
        return torch.optim.Adam(self.get_trainable_params(), lr=0.001)
