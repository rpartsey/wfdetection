from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.utils as vutils
import cv2

from trainer import Trainer

from dataloaders.sampler import ImbalancedDatasetSampler

from models.classifier_r2_unet import Classifier_R2AttU_Net
from models.classifier_unet import UNet

from utils.metric_counter import Collector
from utils.metrics_evaluator import vectorize
from utils.metrics import main_dice_metric


class ClassTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_loss = nn.CrossEntropyLoss()

    def _get_model(self):
        m = UNet(img_ch=3, output_ch=2, num_classes=2)
        m = nn.DataParallel(m)
        return m

    def _write_image(self, data, epoch):
        # Should be images
        t = tqdm(DataLoader(data, shuffle=False, batch_size=1),
                 desc="{}_img_e:{}".format(str(data), epoch))

        for i, (imgs, masks, real_img, _) in enumerate(t):
            imgs = imgs.to(self.device)

            with torch.no_grad():
                segm_logits, class_logits = self.model(imgs)
                logits = segm_logits.softmax(1)
                logits[class_logits.argmax(1) == 0] = 0.0
                # pred_class_detach = class_logits.detach().softmax(1).unsqueeze(-1).unsqueeze(-1)
                # logits = segm_logits.softmax(1) * pred_class_detach
            # softmax_logits = torch.softmax(logits, dim=1)[0, 1:, :, :].float().cpu()
            softmax_logits = logits[0, 1:, :, :].float().cpu()
            softmax_logits = (softmax_logits >= 0.5).float()

            masks = masks[0].float().cpu()

            imgs = real_img[0].unsqueeze(0).float().cpu()
            if imgs.shape[0] == 3:
                imgs = imgs.numpy().transpose(1, 2, 0)
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

        collector = Collector()
        for i, (imgs, masks, _, classes) in enumerate(t):
            imgs, masks, classes = imgs.to(self.device), masks.to(self.device), classes.to(self.device)

            segm_logits, class_logits = self.model(imgs)

            # pred_class_detach = class_logits.detach().softmax(1).unsqueeze(-1).unsqueeze(-1)
            # segm_logits = segm_logits.softmax(1) * pred_class_detach
            detached_class = class_logits.detach().argmax(1)
            tp_tn = (detached_class == classes)
            tp_fp = (detached_class == 1)
            tn_fn = (detached_class == 0)
            tp = tp_tn & tp_fp
            tp_fn = classes == 1

            prec = (tp.float().sum() / (tp_fp.float().sum() + 1e-10)).item()
            rec = (tp.float().sum() / (tp_fn.float().sum() + 1e-10)).item()
            f1 = (2 * prec * rec) / (prec + rec + 1e-10)

            loss = torch.tensor([0.0]).float().to(segm_logits.get_device())
            segm_filter = tp_fp | tp_fn
            if segm_filter.sum() > 0.5:
                loss = self.loss_fn(segm_logits[segm_filter].softmax(1), masks[segm_filter], do_sigmoid_or_softmax=False)

            class_loss = self.class_loss(class_logits, classes)

            self.optimizer.zero_grad()
            (loss + class_loss).backward()
            self.optimizer.step()
            mc.add_losses(loss.item())

            # custom for classifier
            collector.add("class_loss", class_loss.item())
            self.mc.writer.add_scalar("train_batch/class_loss", class_loss.item(), self.global_step)
            acc = tp_tn.float().mean().item()
            collector.add("class_accuracy", acc)
            self.mc.writer.add_scalar("train_batch/class_accuracy", acc, self.global_step)
            # class_f1
            collector.add("class_f1", f1)
            self.mc.writer.add_scalar("train_batch/class_f1", f1, self.global_step)
            # class_prec
            collector.add("class_prec", prec)
            self.mc.writer.add_scalar("train_batch/class_prec", prec, self.global_step)
            # class_recall
            collector.add("class_recall", prec)
            self.mc.writer.add_scalar("train_batch/class_recall", rec, self.global_step)

            segm_logits[tn_fn, 1, :, :] = segm_logits[tn_fn, 0, :, :] - 100
            segm_logits = segm_logits.softmax(1)

            with torch.no_grad():
                last_n = mc.add_batch_metrics(
                    vectorize(main_dice_metric,
                              segm_logits.cpu(),
                              masks.cpu(),
                              do_sigmoid_or_softmax=False)
                )
            t.set_postfix(mc.tqdm_message(last_n))
            self.mc.write_batch(last_n, np.mean, self.global_step)
            self.global_step += 1
        t.close()

        self._write_image(self.train_tensorboard_data, epoch)

        self.mc.writer.add_scalars("epoch/class_loss", dict(train=np.mean(collector["class_loss"])), epoch)
        self.mc.writer.add_scalars("epoch/class_accuracy", dict(train=np.mean(collector["class_accuracy"])), epoch)
        self.mc.writer.add_scalars("epoch/class_f1", dict(train=np.mean(collector["class_f1"])), epoch)
        self.mc.writer.add_scalars("epoch/class_recall", dict(train=np.mean(collector["class_recall"])), epoch)
        self.mc.writer.add_scalars("epoch/class_prec", dict(train=np.mean(collector["class_prec"])), epoch)
        mc.write_epoch(epoch)

    def _validate(self, epoch, mc):
        mc.clear()
        self.model = self.model.eval()
        t = tqdm(DataLoader(
            self.val_dataset,
            batch_size=self.config['val_batch_size'],
        ), desc='Validation')
        collector = Collector()
        for imgs, masks, _, classes in t:
            imgs, masks, classes = imgs.to(self.device), masks.to(self.device), classes.to(self.device)

            with torch.no_grad():
                segm_logits, class_logits = self.model(imgs)
                # pred_class_detach = class_logits.detach().softmax(1).unsqueeze(-1).unsqueeze(-1)
                # segm_logits = segm_logits.softmax(1) * pred_class_detach
                detached_class = class_logits.detach().argmax(1)
                tp_tn = (detached_class == classes)
                tp_fp = (detached_class == 1)
                tn_fn = (detached_class == 0)
                tp = tp_tn & tp_fp
                tp_fn = classes == 1

                prec = (tp.float().sum() / (tp_fp.float().sum() + 1e-10)).item()
                rec = (tp.float().sum() / (tp_fn.float().sum() + 1e-10)).item()
                f1 = (2 * prec * rec) / (prec + rec + 1e-10)

                segm_logits[tn_fn, 1, :, :] = segm_logits[tn_fn, 0, :, :] - 100
                segm_logits = segm_logits.softmax(1)

            loss = self.loss_fn(segm_logits, masks, do_sigmoid_or_softmax=False)

            class_loss = self.class_loss(class_logits, classes)

            mc.add_losses(loss.item())
            # custom for classifier
            collector.add("class_loss", class_loss.item())
            collector.add("class_accuracy", tp_tn.float().mean().item())
            collector.add("class_f1", f1)
            collector.add("class_prec", prec)
            collector.add("class_recall", rec)

            with torch.no_grad():
                last_n = mc.add_batch_metrics(
                    vectorize(main_dice_metric,
                              segm_logits.cpu(),
                              masks.cpu(),
                              do_sigmoid_or_softmax=False)
                )

            t.set_postfix(mc.tqdm_message(last_n))
        t.close()

        # Should do images
        self._write_image(self.val_tensorboard_data, epoch)

        self.mc.writer.add_scalars("epoch/class_loss", dict(validation=np.mean(collector["class_loss"])), epoch)
        self.mc.writer.add_scalars("epoch/class_accuracy", dict(validation=np.mean(collector["class_accuracy"])), epoch)
        self.mc.writer.add_scalars("epoch/class_f1", dict(validation=np.mean(collector["class_f1"])), epoch)
        self.mc.writer.add_scalars("epoch/class_recall", dict(validation=np.mean(collector["class_recall"])), epoch)
        self.mc.writer.add_scalars("epoch/class_prec", dict(validation=np.mean(collector["class_prec"])), epoch)
        mc.write_epoch(epoch)

        return mc.get_loss()
