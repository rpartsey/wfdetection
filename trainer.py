from __future__ import print_function

import logging

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import cv2
import numpy as np
from tensorboardX import SummaryWriter

from metrics import Meter

cv2.setNumThreads(0)


class Trainer(object):
    def __init__(self, config, stage_number):
        self.config = config
        self.config.load_stage(stage_number)
        self.writer = SummaryWriter(self.config.logdir)
        self.optim = self.config.load_optimizer()
        self.scheduler = self.config.load_scheduler(self.optim)
        self.early_stopper = self.config.load_early_stopper()
        self.criterion = self.config.load_criterion()
        self.accum_steps = self.config.config.get("real_train_size", 32) // self.config.config["train_batch_size"]
        print("Accumulation steps", self.accum_steps)

    @property
    def device(self):
        return self.config.device

    @property
    def lr(self):
        lr = 'undefined'
        for param_group in self.optim.param_groups:
            lr = param_group['lr']
        return lr

    @property
    def model(self):
        return self.config.model

    def train(self):
        self.global_step = 0
        data, tensor_data = self.config.load_datasets()
        self._write_image("train", tensor_data["train"], -1)
        self._write_image("validation", tensor_data["validation"], -1)

        best_loss = 1e10
        best_metrics = dict()
        print("Epochs", self.config.epochs)
        for epoch in range(self.config.epochs):
            # is writing only batch metrics
            self._train_epoch(epoch, self.config.dataloaders("train", data["train_aug"]))

            train_data = self.config.dataloaders(None, data["train"])
            validation_data = self.config.dataloaders(None, data["validation"])
            # for train
            self.calc_metrics("train", epoch, train_data)
            # for val
            val_loss, val_metrics = self.calc_metrics("validation", epoch, validation_data)
            print("Val loss", val_loss)
            print("Val metrics", val_metrics)

            self._write_image("train", tensor_data["train"], epoch)
            self._write_image("validation", tensor_data["validation"], epoch)

            # step of scheduler
            self.scheduler.step(metrics=val_loss)
            # step of early stopper
            self.early_stopper(val_loss)

            self.config.save_last_model_state()

            if val_loss < best_loss:
                print("*** !Update in BEST LOSS *** delta", best_loss - val_loss)
                best_loss = val_loss
                self.config.save_best_model_state("loss")

            for metric_name, val_metric_value in val_metrics.items():
                if np.isnan(val_metric_value):
                    continue
                best_metric_value = best_metrics.get(metric_name, 0.0)
                if best_metric_value < val_metric_value:
                    print("*** Update in BEST {} *** delta".format(metric_name), val_metric_value - best_metric_value)
                    best_metrics[metric_name] = val_metric_value
                    self.config.save_best_model_state(metric_name)

            if self.early_stopper.early_stop:
                logging.debug('Early stopping after {} epochs'.format(epoch))
                break

    def _write_image(self, name, data, epoch):
        # Should be images
        t = tqdm(DataLoader(data, shuffle=False, batch_size=1),
                 desc="{}_img_e:{}".format(str(data), epoch))
        for i, (imgs, masks, real_img, _) in enumerate(t):
            imgs = imgs.to(self.config.device)

            with torch.no_grad():
                processed_logits = self.model(imgs)
                processed_logits = processed_logits.sigmoid()
                processed_logits = processed_logits[0:, -1, :, :].float().cpu()
            # SHAPE SHOULD be 1xWxH
            masks = masks[0].float()

            imgs = real_img[0].unsqueeze(0).float()
            if imgs.shape[0] == 3:
                imgs = imgs.numpy().transpose(1, 2, 0)
                imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY)
                imgs = torch.from_numpy(np.expand_dims(imgs, 0))

            # all of inputs is 1xWxH
            grid = vutils.make_grid(
                [imgs, processed_logits, masks],
                nrow=3
            )
            self.writer.add_image("{}/image_{}".format(name, i), grid, epoch)
        t.close()

    def _train_epoch(self, epoch, dataloader):
        self.model.train()

        t = tqdm(dataloader, desc='Train Epoch {}, lr {}'.format(epoch, self.lr))

        self.optim.zero_grad()
        for i, (imgs, masks, _, _) in enumerate(t):
            imgs, masks = imgs.to(self.device), masks.float().to(self.device)

            logits = self.model(imgs)
            if logits.shape[1] != 1:
                raise ValueError("Only SIGMOID supported")
            loss = self.criterion(logits, masks) / self.accum_steps

            loss.backward()
            if self.config.config.get("bug", False):
                if (self.global_step + 1) % self.accum_steps:
                    self.optim.step()
                    self.optim.zero_grad()
            else:
                if (self.global_step + 1) % self.accum_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad()

            t.set_postfix(loss=loss.item())
            self.writer.add_scalar("batch", loss.item(), self.global_step)
            self.global_step += 1
        t.close()

    def calc_metrics(self, name, epoch, dataloader):
        self.model.eval()
        t = tqdm(dataloader, desc='Calc metr. {} e:{}'.format(name, epoch))

        meter = Meter()
        loss_list = []
        loss_length = 0.0001

        for imgs, masks, _, _ in t:
            imgs, masks = imgs.to(self.device), masks.float().to(self.device)
            batch = imgs.shape[0]

            with torch.no_grad():
                logits = self.model(imgs)
                if logits.shape[1] != 1:
                    raise ValueError("Only SIGMOID supported")
                loss = self.criterion(logits, masks)
                loss_list.append(loss.item() * batch)
                loss_length += batch

                meter.update(masks.cpu(), logits.cpu())
        t.close()

        metrics = meter.get_metrics()
        metrics["loss"] = sum(loss_list) / loss_length
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalars("epoch/{}".format(metric_name), {name: metric_value}, epoch)
        loss = metrics.pop("loss")
        return loss, metrics
