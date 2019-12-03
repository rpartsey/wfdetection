from __future__ import print_function
import os
import logging
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

from utils.schedulers import WarmRestart, LinearDecay, PlateauLRDecay
from losses import LOSSES
from utils.metric_counter import MetricCounter
from utils.metrics_evaluator import vectorize
from utils.early_stopping import EarlyStopping
from utils.train_validation_split import random_train_val_split
from utils.hard_negative_mining import OHEM
from utils.metrics import main_dice_metric
from dataloaders.binary_dataloader import BinaryLoader
from dataloaders import sample_tensorfboard_images
from dataloaders.sampler import ImbalancedDatasetSampler
from dataloaders.transformations import create_transform
cv2.setNumThreads(0)


from dataloaders.aug import get_transforms
from dataloaders.augmentation import StandartAugmentation


class Trainer(object):
    def __init__(self, configs, device, stage):
        self.config = configs
        self.stage = stage
        self.stage_config = self.config['stage%d' % self.stage]
        self.mc = MetricCounter(configs, stage, metric_name="dice", loss_name=self.stage_config["loss"])
        print("Metric Counter loaded")

        (self.train_dataset, self.val_dataset),\
        (self.train_tensorboard_data, self.val_tensorboard_data) = self._get_dataset()
        print("datasets loaded")
        self.device = device

    def _get_model(self):
        # TODO: choosing model
        mtype = self.config["model"]["type"]
        if mtype == "universal-unet":
            from models.universal_UNet import UNet
            return UNet(self.config['model']['name'], device=self.device, encoder_depth=self.config['model']['depth'],
                 num_classes=self.config['model']['num_classes'], SCSE=self.config['model']['scse_in_decoder'],
                        pretrained=self.config["model"]["pretrained"], freeze=(self.config["stage%d" % self.stage])["model_freeze"])
        elif mtype == 'torch-unet':
            from models.torchhub_unet import UNet
            return UNet(pretrained=self.config["model"]['pretrained'])
        elif mtype == 'old-unet':
            from models.unet import UNet as oldUnet
            return oldUnet()
        elif mtype == 'r2-unet':
            from models.r2_unet import R2U_Net
            return R2U_Net()
        elif mtype == 'attn-unet':
            from models.r2_unet import AttU_Net
            return AttU_Net()
        elif mtype =='r2-attn-unet':
            from models.r2_unet import R2AttU_Net
            return R2AttU_Net()
        raise NotImplementedError("This type of model is not implemented `{}`".format(mtype))

    def _get_dataset(self):
        # get df
        df = pd.read_csv(self.config["csv_path"])
        df = df.groupby("ImageId").first().reset_index()
        df["mask_exists"] = df[' EncodedPixels'] != ' -1'
        # df = df.sample(1000)

        # and leave only ids which really have labels
        valid_file_names = os.listdir(os.path.join(self.config['data_path'], 'mask'))
        valid_image_ids = set(x.strip(".png") for x in valid_file_names)
        cut_df = df[df['ImageId'].isin(valid_image_ids)].reset_index(drop=True)

        train_csv, val_csv = random_train_val_split(cut_df,
                                                    self.config["validation_fraction"],
                                                    self.config["random_state"])

        transform_config = self.config["transformations"]
        image_transform = create_transform(transform_config["image"])
        mask_transform = create_transform(transform_config["mask"])
        aug_config = self.config["augmentations"]
        aug_transform = StandartAugmentation(aug_config["p"], width=aug_config["width"], height=aug_config["height"])

        train_data = BinaryLoader(train_csv, self.config['data_path'],
                                  image_transform=image_transform,
                                  mask_transform=mask_transform,
                                  aug_transform=aug_transform)
        val_data = BinaryLoader(val_csv, self.config['data_path'],
                                image_transform=image_transform,
                                mask_transform=mask_transform)
                                # aug_transform=aug_transform)

        print('Training on : {} samples, validating on {} samples'.format(len(train_data), len(val_data)))

        train_tensorboard_data = sample_tensorfboard_images.TrainLoader.build_from_binary_loader(train_data)
        val_tensorboard_data = sample_tensorfboard_images.ValLoader.build_from_binary_loader(val_data)
        return (train_data, val_data), (train_tensorboard_data, val_tensorboard_data)

    def _get_model_parameters(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def _get_optimizer(self):
        opt_name = self.config['stage%d' % self.stage]['optimizer']['name']
        if opt_name == 'adam':
            optimizer = optim.Adam(self._get_model_parameters(),
                                   lr=self.config['stage%d' % self.stage]['optimizer']['lr'])
        elif opt_name == 'sgd':
            optimizer = optim.SGD(self._get_model_parameters(),
                                  lr=self.config['stage%d' % self.stage]['optimizer']['lr'])
        elif opt_name == 'adadelta':
            optimizer = optim.Adadelta(self._get_model_parameters(),
                                       lr=self.config['stage%d' % self.stage]['optimizer']['lr'])
        else:
            raise ValueError('Optimizer [%s] not recognized.' % opt_name)
        return optimizer

    def _get_loss(self):
        loss_name = self.config['stage%d' % self.stage]['loss']
        loss_class = LOSSES.get(loss_name, None)
        if loss_class is not None:
            return loss_class
        else:
            raise NotImplementedError("This type of loss is not implemented `{}`".format(loss_class))

    def _get_scheduler(self, optimizer):
        scheduler_name = self.config['stage%d' % self.stage]['scheduler']['name']
        if scheduler_name == 'plateau':
            scheduler = PlateauLRDecay(optimizer,
                                       mode=self.config['stage%d' % self.stage]['scheduler']['mode'],
                                       patience=self.config['stage%d' % self.stage]['scheduler']['patience'],
                                       factor=self.config['stage%d' % self.stage]['scheduler']['factor'],
                                       min_lr=self.config['stage%d' % self.stage]['scheduler']['min_lr'],
                                       eps=self.config['stage%d' % self.stage]['scheduler']['epsilon'])
        elif scheduler_name == 'warmrestart':
            scheduler = WarmRestart(optimizer,
                                    T_max=self.config['stage%d' % self.stage]['scheduler']['epochs_per_cycle'],
                                    eta_min=self.config['stage%d' % self.stage]['scheduler']['min_lr'])
        elif scheduler_name == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['stage%d' % self.stage]['scheduler']['min_lr'],
                                    num_epochs=self.config['stage%d' % self.stage]['num_epochs'],
                                    start_epoch=self.config['stage%d' % self.stage]['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % scheduler_name)
        return scheduler

    def _get_early_stopper(self):
        return EarlyStopping(self.config['stage%d' % self.stage]['stopper']['patience'])

    def _init_params(self):
        self.model = self._get_model().to(self.device)
        self.optimizer = self._get_optimizer()
        if self.config['stage%d' % self.stage]['resume_training']:
            checkpoint = torch.load(
                '{}last_{}.h5'.format(self.config['experiment_path'], self.config['experiment_desc']))
            self.model.load_state_dict(checkpoint['model'])
            if self.config["stage%d" % self.stage]["optimizer"]["load"]:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.mc.best_metric = checkpoint['best_metric']
        self.scheduler = self._get_scheduler(self.optimizer)
        self.loss_fn = self._get_loss()
        self.early_stopper = self._get_early_stopper()

    def train(self):
        print("Start params init")
        self._init_params()
        print("End params init")
        self.global_step = 0
        self._write_image(self.train_tensorboard_data, -1)
        self._write_image(self.val_tensorboard_data, -1)
        for epoch in range(0, self.config['stage%d' % self.stage]['num_epochs']):
            self._run_epoch(epoch, self.mc.train)
            val_loss = self._validate(epoch, self.mc.val)

            self.scheduler.step(metrics=val_loss)  # we can change the criteria to any metric
            self.early_stopper(val_loss)

            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_metric': self.mc.best_metric
            }, '{}last_{}.h5'.format(self.config['experiment_path'], self.config['experiment_desc']))
            if self.mc.update_best_model():
                shutil.copy(
                    '{}last_{}.h5'.format(self.config['experiment_path'], self.config['experiment_desc']),
                    '{}best_{}.h5'.format(self.config['experiment_path'], self.config['experiment_desc'])
                )

            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.mc.val.tqdm_message()))

            if self.early_stopper.early_stop:
                logging.debug('Early stopping after {} epochs'.format(epoch))
                break

    def _write_image(self, data, epoch):
        # Should be images
        t = tqdm(DataLoader(data, shuffle=False, batch_size=1),
                 desc="{}_img_e:{}".format(str(data), epoch))
        for i, (imgs, masks, real_img, _) in enumerate(t):
            imgs = imgs.to(self.device)

            with torch.no_grad():
                logits = self.model(imgs)
            softmax_logits = torch.softmax(logits, dim=1)[0, 1:, :, :].float().cpu()
            masks = masks[0].float().cpu()

            imgs = real_img[0].unsqueeze(0).float().cpu()
            if imgs.shape[0] == 3:
                imgs = imgs.numpy().transpose(1,2,0)
                imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY)
                imgs = torch.from_numpy(np.expand_dims(imgs, 0))
            grid = vutils.make_grid(
                [imgs, softmax_logits, masks],
                nrow=3
            )
            self.mc.writer.add_image("{}/image:{}".format(str(data), i), grid, epoch)

        t.close()

    def _run_epoch(self, epoch, mc):
        mc.clear()
        self.model = self.model.train()

        lr = 'undefined'
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        t = tqdm(
            DataLoader(self.train_dataset,
                       sampler=ImbalancedDatasetSampler(self.train_dataset),
                       batch_size=self.config['train_batch_size']),
            desc='Epoch {}, lr {}'.format(epoch, lr))

        for i, (imgs, masks, _, _) in enumerate(t):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            logits = self.model(imgs)

            loss = self.loss_fn(logits, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mc.add_losses(loss.item())

            with torch.no_grad():
                last_n = mc.add_batch_metrics(
                    vectorize(main_dice_metric,
                              logits.cpu(),
                              masks.cpu()))
            t.set_postfix(mc.tqdm_message(last_n))
            self.mc.write_batch(last_n, np.mean, self.global_step)
            self.global_step += 1
        t.close()

        self._write_image(self.train_tensorboard_data, epoch)

        mc.write_epoch(epoch)

    def _validate(self, epoch, mc):
        mc.clear()
        self.model = self.model.eval()
        t = tqdm(DataLoader(
            self.val_dataset,
            batch_size=self.config['val_batch_size'],
        ), desc='Validation')
        for imgs, masks, _, _ in t:
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            with torch.no_grad():
                logits = self.model(imgs)

            loss = self.loss_fn(logits, masks)
            mc.add_losses(loss.item())

            with torch.no_grad():
                last_n = mc.add_batch_metrics(vectorize(
                    main_dice_metric,
                    logits.cpu(),
                    masks.cpu()
                ))

            t.set_postfix(mc.tqdm_message(last_n))
        t.close()

        # Should do images
        self._write_image(self.val_tensorboard_data, epoch)

        mc.write_epoch(epoch)

        return mc.get_loss()
