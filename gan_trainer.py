from __future__ import print_function

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import cv2
import numpy as np
from tensorboardX import SummaryWriter
from gan_config import Config

from metrics import Meter

cv2.setNumThreads(0)


class GanTrainer(object):
    def __init__(self, config, descriminator_config, stage_number, train_descriminator=True, train_generator=False):
        self.config = config
        self.config.load_stage(stage_number)
        self.writer = SummaryWriter(self.config.logdir)
        self.optim_G = self.config.load_optimizer()
        self.scheduler = self.config.load_scheduler(self.optim_G)
        self.early_stopper = self.config.load_early_stopper()
        self.criterion_G = self.config.load_criterion()
        self.accum_steps = 1
        self.descriminator_config = descriminator_config
        self.optim_D = self.descriminator_config.load_optimizer()
        self.criterion_D = nn.BCELoss()
        self.train_descriminator = train_descriminator
        self.train_generator = train_generator
        print("Descriminator train", train_descriminator)
        print("Generator train", train_generator)
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
    def model_G(self):
        return self.config.model

    @property
    def model_D(self):
        return self.descriminator_config.model

    def train(self):
        self.global_step = 0
        data, tensor_data = self.config.load_datasets()
        self._write_image("train", tensor_data["train"], -1)
        self._write_image("validation", tensor_data["validation"], -1)

        for epoch in range(self.config.epochs):
            # is writing only batch metrics
            self._train_epoch(epoch, self.config.dataloaders("train", data["train_aug"]))

            train_data = self.config.dataloaders(None, data["train"])
            validation_data = self.config.dataloaders(None, data["validation"])
            # for train
            self.calc_metrics("train", epoch, train_data)
            # for val
            val_metrics = self.calc_metrics("validation", epoch, validation_data)
            print("Val metrics", val_metrics)

            self._write_image("train", tensor_data["train"], epoch)
            self._write_image("validation", tensor_data["validation"], epoch)

            self.config.save_best_model_state("gen_{}_{}_{}".format(self.train_descriminator,
                                                                    self.train_generator, epoch))
            self.descriminator_config.save_best_model_state("desc_{}_{}_{}".format(self.train_descriminator,
                                                                                   self.train_generator, epoch))


    def _write_image(self, name, data, epoch):
        # Should be images
        t = tqdm(DataLoader(data, shuffle=False, batch_size=1),
                 desc="{}_img_e:{}".format(str(data), epoch))
        for i, (imgs, masks, real_img, _) in enumerate(t):
            imgs = imgs.to(self.config.device)

            with torch.no_grad():
                processed_logits = self.model_G(imgs)
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
        self.model_D.train()
        self.model_G.train()

        t = tqdm(dataloader, desc='Train Epoch {}, lr {}'.format(epoch, None))
        real_label = 0
        fake_label = 1

        for i, (imgs, masks, _, _) in enumerate(t):
            if self.train_descriminator:
                self.optim_D.zero_grad()
            imgs, masks = imgs.to(self.device), masks.float().to(self.device)
            real_masked = imgs * masks

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            # Format batch
            b_size = real_masked.size(0)
            label = torch.full((b_size,), real_label, device=self.device)
            # Forward pass real batch through D
            output = self.model_D.forward(real_masked).view(-1)
            # Calculate loss on all-real batch
            errD_real = self.criterion_D(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            # TODO: !
            # D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            # Generate fake image batch with G
            fake = self.model_G.forward(imgs)
            fake_masked = imgs * fake.sigmoid()
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = self.model_D(fake_masked.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion_D(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            if self.train_descriminator:
                self.optim_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if self.train_generator:
                self.optim_G.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.model_D(fake_masked).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion_D(output, label) + self.criterion_G(fake, masks) / 10.0
            # Calculate gradients for G
            errG.backward()
            # D_G_z2 = output.mean().item()
            # Update G
            if self.train_generator:
                self.optim_G.step()

            t.set_postfix(generator=errG.item(), descriminator=errD.item())
            self.writer.add_scalar("batch_generator", errG.item(), self.global_step)
            self.writer.add_scalar("batch_descriminator", errD.item(), self.global_step)
            self.global_step += 1
        t.close()

    def calc_metrics(self, name, epoch, dataloader):
        self.model_G.eval()
        self.model_D.eval()
        t = tqdm(dataloader, desc='Calc metr. {} e:{}'.format(name, epoch))

        meter = Meter()
        G_loss_list = []
        D_loss_list = []
        loss_length = 0.0001
        real_label = 0
        fake_label = 1

        for imgs, masks, _, _ in t:
            imgs, masks = imgs.to(self.device), masks.float().to(self.device)
            batch = imgs.shape[0]
            with torch.no_grad():
                real_masked = imgs * masks

                b_size = real_masked.size(0)
                label = torch.full((b_size,), real_label, device=self.device)
                output = self.model_D.forward(real_masked).view(-1)
                errD_real = self.criterion_D(output, label)
                fake = self.model_G.forward(imgs)
                fake_masked = imgs * fake.sigmoid()
                label.fill_(fake_label)
                output = self.model_D(fake_masked.detach()).view(-1)
                errD_fake = self.criterion_D(output, label)
                errD = errD_real + errD_fake

                logits = self.model_G(imgs)
                if logits.shape[1] != 1:
                    raise ValueError("Only SIGMOID supported")
                loss = self.criterion_G(logits, masks)
                G_loss_list.append(loss.item() * batch)
                D_loss_list.append(errD.item() * batch)
                loss_length += batch

                meter.update(masks.cpu(), logits.cpu())
        t.close()

        metrics = meter.get_metrics()
        metrics["G_loss"] = sum(G_loss_list) / loss_length
        metrics["D_loss"] = sum(D_loss_list) / loss_length
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalars("epoch/{}".format(metric_name), {name: metric_value}, epoch)
        return metrics
