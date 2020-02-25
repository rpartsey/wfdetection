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

from models import MODELS
from losses import LOSSES
from schedulers import SCHEDULERS
from optimizers import OPTIMIZERS


torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True


class Config(object):
    CONFIG_FILENAME = "config.yaml"
    LOGDIRNAME = "logdir"
    BEST_MODEL_NAME = "best.h5"
    LAST_MODEL_NAME = "last.h5"
    MODEL_KEY = "model"

    def __init__(self, exp_dir, device):
        print("Loading config -->", exp_dir)
        self.device = device
        self.exp_dir = exp_dir
        self.config = self.__load_config()

        self.__set_seed()

        self.transformations = self.__load_transformations()
        self.augmentations = self.__load_augmentations()
        self.model = self.__load_model()
        self.stage_config = self.load_stage(1)

    def action_cut_borders(self, *arg, **kwargs):
        return cutBorders(*arg, **kwargs)
    def uncut_borders(self, *args, **kwargs):
        return uncut(*args, **kwargs)

    def __set_seed(self):
        seed = self.config["random_state"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # TODO: this is temporary method
    def predict(self, x):
        self.model.eval()
        return self.model.forward(x).sigmoid()

    def dataloaders(self, name, dataset):
        params = dict()
        if name == "train":
            if self.config.get("data_only_positive", False):
                print("Using just RandomSampler")
                params["shuffle"] = True
            elif self.config.get("data_sampler", None) == "frog_balanced_sampler":
                print("Using frog_sampler")
                params["sampler"] = BalanceClassSampler(dataset, int(len(dataset) * self.config.get("sampler_multiplier", 3.0)))
            else:
                print("Using ImbalancedDataSampler")
                params["sampler"] = ImbalancedDatasetSampler(dataset)
            params["batch_size"] = self.config["train_batch_size"]
        else:
            params["shuffle"] = False
            params["batch_size"] = self.config["val_batch_size"]
        loader = DataLoader(dataset, **params)
        return loader

    @property
    def cut_borders(self):
        return self.config.get("cut_borders", False)

    @property
    def logdir(self):
        return os.path.join(self.exp_dir, self.LOGDIRNAME)

    @property
    def epochs(self):
        return self.stage_config["num_epochs"]

    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def load_stage(self, stage_number):
        stage_name = "stage{}".format(stage_number)
        self.stage_config = self.config[stage_name]
        if self.stage_config["resume_training"]:
            print("Loading model")
            self.__load_model_from_path(self.stage_config["model_path"])
        return self.stage_config

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

    def __load_model(self):
        model_config = self.config["model"]
        model_type = model_config["type"]
        model_class = MODELS.get(model_type, None)
        model_params = model_config.get("params", {})
        if model_class is None:
            raise NotImplementedError(model_type)
        model_params["device"] = self.device
        model = model_class(**model_params)
        model = model.to(self.device)
        return model

    def __load_config(self):
        with open(os.path.join(self.exp_dir, self.CONFIG_FILENAME), 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def __load_transformations(self):
        transform_config = self.config["transformations"]
        return {
            "mask": create_transform(transform_config["mask"]),
            "image": create_transform(transform_config["image"])
        }

    def __load_augmentations(self):
        augmentations_config = self.config["augmentations"]
        if self.config.get("augmentation_class", None) == "frog":
            print("!!! Frog Augmentations")
            cls = FrogAugmentation(p=augmentations_config["p"],
                                    size=augmentations_config["size"])
        else:
            cls = StandartAugmentation(p=augmentations_config["p"],
                                    size=augmentations_config["size"])
        return cls

    @classmethod
    def create_from_path(cls, config_path, device):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls.create_from_config(config, device)

    @classmethod
    def create_from_config(cls, config, device):
        if config.get("generate_random", True):
            config["random_state"] = int(time.time())
        exp_path = config["experiment_path"]
        print("Creating Experiment path:", exp_path)

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        else:
            while True:
                ans = input("Path `{}` exists, do you want to delete it?: [y/n]".format(exp_path))
                if ans == "y":
                    shutil.rmtree(exp_path)
                    os.makedirs(exp_path)
                    break
                elif ans == "n":
                    return None
        logdir_path = os.path.join(exp_path, cls.LOGDIRNAME)
        os.makedirs(logdir_path, exist_ok=True)

        with open(os.path.join(exp_path, cls.CONFIG_FILENAME), 'w') as f:
            yaml.dump(config, f)
        return cls(exp_path, device)

    def load_datasets(self):
        # get df
        df = pd.read_csv(self.config["csv_path"])
        # df = df.groupby("ImageId").first().reset_index()
        # df["mask_exists"] = df[' EncodedPixels'] != ' -1'
        if self.config.get("data_only_positive", False):
            df = df[df["mask_exists"]]

        # train_split_name = self.config.get("train_split_name", "split/train.npy")
        # train_image_ids = np.load(train_split_name, allow_pickle=True)
        # val_image_ids = np.load("split/validation.npy", allow_pickle=True)

        # train_csv = df[df["ImageId"].isin(train_image_ids)].reset_index(drop=True)
        # val_csv = df[df["ImageId"].isin(val_image_ids)].reset_index(drop=True)

        train_csv = df.iloc[: len(df)-5, :]
        val_csv = df.iloc[-5:, :]

        cut_borders = self.config.get("cut_borders", False)

        print("Cut borders -->", cut_borders)
        train_aug_data = BinaryLoader(train_csv, self.config['data_path'],
                                  image_transform=self.transformations["image"],
                                  mask_transform=self.transformations["mask"],
                                  aug_transform=self.augmentations,
                                      cut_borders=cut_borders)

        train_data = BinaryLoader(train_csv, self.config['data_path'],
                                      image_transform=self.transformations["image"],
                                      mask_transform=self.transformations["mask"],
                                  cut_borders=cut_borders)

        val_data = BinaryLoader(val_csv, self.config['data_path'],
                                image_transform=self.transformations["image"],
                                mask_transform=self.transformations["mask"],
                                cut_borders=cut_borders)

        print('Training on : {} samples, validating on {} samples'.format(len(train_aug_data), len(val_data)))

        train_tensorboard_data = sample_tensorfboard_images.TrainLoader.build_from_binary_loader(train_data)
        val_tensorboard_data = sample_tensorfboard_images.ValLoader.build_from_binary_loader(val_data)

        data = {"train_aug": train_aug_data,
                "validation": val_data,
                "train": train_data,
                }

        tensor_data = {"train": train_tensorboard_data,
                       "validation": val_tensorboard_data}

        return data, tensor_data

    def load_criterion(self):
        loss_config = self.stage_config['loss']
        loss_type = loss_config["type"]
        loss_class = LOSSES.get(loss_type, None)
        loss_params = loss_config.get("params", {})
        if loss_class is not None:
            return loss_class(**loss_params)
        else:
            raise NotImplementedError("This type of loss is not implemented `{}`".format(loss_type))

    def load_optimizer(self):
        optim_config = self.stage_config['optimizer']
        optim_type = optim_config["type"]
        optim_class = OPTIMIZERS.get(optim_type, None)
        if optim_class is None:
            raise NotImplementedError("Optimizer [%s] not recognized." % optim_type)
        optim_params = optim_config.get("params", {})
        return optim_class(self.get_trainable_params(), **optim_params)

    def load_scheduler(self, optimizer):
        scheduler_config = self.stage_config['scheduler']
        scheduler_type = scheduler_config["type"]
        scheduler_class = SCHEDULERS.get(scheduler_type, None)
        if scheduler_class is None:
            raise NotImplementedError("Scheduler [%s] not recognized." % scheduler_type)
        scheduler_params = scheduler_config.get("params", {})
        return scheduler_class(optimizer, **scheduler_params)

    def load_early_stopper(self):
        return EarlyStopping(self.stage_config['stopper']['patience'])
