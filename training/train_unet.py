import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from tensorboardX import SummaryWriter
# from metrics_evaluator import PerformanceMetricsEvaluator
import sys
from tqdm import tqdm
import time
import cv2
import os
import skimage.io as io
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random

sys.path.insert(0, '../dataloaders/')
sys.path.insert(0, '../models')
sys.path.insert(0, '../utils')

from binary_dataloader import BinaryLoader
from train_validation_split import random_train_val_split
from metrics_evaluator import PerformanceMetricsEvaluator
from losses import dice_coeff, soft_dice_coeff
from unet import UNet

import warnings
warnings.filterwarnings("ignore")

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter(1, labels.data, 1)
    
    target = torch.tensor(target)
        
    return target

def train(model, train_loader, val_loader, optimizer, num_epochs, path_to_save_best_weights):
    model.train()

    log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()
    softmax = nn.Softmax(dim=1)

    # criterion_nlloss = nn.NLLLoss(weight=torch.tensor([0.5015290518452679, 164.00001523734954]).to(device))
    metrics_evaluator = PerformanceMetricsEvaluator()

    to_tensor = transforms.ToTensor()

    writer = SummaryWriter('runs/unet_256/')

    since = time.time()

    best_model_weights = model.state_dict()
    # best_IoU = 0.0 
    best_dice = 0.0
    best_val_loss = 1000000000

    curr_val_loss = 0.0
    curr_training_loss = 0.0
    curr_training_dice = 0.0
    curr_val_dice = 0.0
    # curr_training_IoU = 0.0
    # curr_val_IoU = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                # scheduler.step(best_val_loss)
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            # running_IoU = 0 
            running_dice = 0.

            # Iterate over data.
            for imgs, masks in tqdm(data_loader):

                imgs, masks = imgs.to(device), masks.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                logits = model(imgs)
                softmax_logits = softmax(logits)

                one_hoted_masks = make_one_hot(torch.unsqueeze(masks, 1))
                
                # loss = soft_dice_coeff(softmax_logits, one_hoted_masks)
                loss = 1 - dice_coeff(softmax_logits, one_hoted_masks)
                # log_softmax_logits = log_softmax(logits)
                # loss = criterion_nlloss(log_softmax_logits, masks)
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
                if phase == 'val':
                    name = 'ValidationEpoch'
                else:
                    name = 'TrainingEpoch'

                collapsed_softmax_logits = torch.argmax(softmax_logits, dim=1)

                empty_tensor = torch.zeros((collapsed_softmax_logits[0].size())).float()
                pred_mask_overlapping = vutils.make_grid([ collapsed_softmax_logits[0].float().cpu(),masks[0].float().cpu(),empty_tensor.cpu()], nrow=3)

                image_to_show = vutils.make_grid([imgs[0].cpu(),pred_mask_overlapping])
                writer.add_image('{}: {}'.format(name, str(epoch)), image_to_show,epoch)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.detach().item()

                # batch_IoU = 0.0
                batch_dice = 0.0
                for k in range(len(imgs)):
                    # batch_IoU += metrics_evaluator.mean_IU(collapsed_softmax_logits.numpy()[k], masks.cpu().numpy()[k])
                    batch_dice += dice_coeff(softmax_logits, one_hoted_masks).item()
                # batch_IoU /= len(imgs)
                batch_dice /= len(imgs)
                running_dice += batch_dice 
                # running_IoU += batch_IoU
            epoch_loss = running_loss / len(data_loader)
            # epoch_IoU = running_IoU / len(data_loader)
            epoch_dice = running_dice / len(data_loader)

            print('{} Loss: {:.4f} Dice: {:.4f}'.format(phase, epoch_loss, epoch_dice))
 
            # deep copy the model
            if phase == 'val' and epoch_loss < best_val_loss: # TODO add IoU
                best_val_loss = epoch_loss
                # best_IoU = epoch_IoU
                best_dice = epoch_dice
                best_model_weights = model.state_dict()
                 # Saving best model
                torch.save(best_model_weights, os.path.join(path_to_save_best_weights, 'unet_baseline_dice_256_{:2f}.pth'.format(best_val_loss)))

    
            if phase == 'val':
                # print(optimizer.param_groups[0]['lr'])
                curr_val_loss = epoch_loss
                # curr_val_IoU = epoch_IoU
                curr_val_dice = epoch_dice
            else:
                curr_training_loss = epoch_loss
                # curr_training_IoU = epoch_IoU
                curr_training_dice = epoch_dice

        writer.add_scalars('TrainVal_dice', 
                            {'train_dice': curr_training_dice,
                             'validation_dice': curr_val_dice
                            },
                            epoch
                           )
        writer.add_scalars('TrainValLoss', 
                            {'trainLoss': curr_training_loss,
                             'validationLoss': curr_val_loss
                            },
                            epoch
                           ) 
   
    # Show the timing and final statistics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss)) # TODO add IoU




# Choose free GPU
device = torch.device("cuda:{}".format(str(get_freer_gpu())))

ROOT_DIR = '/home/bohdan/SIIM-ACR_Kaggle_MedicalSegmentators/data/processed_256'
DIR_TO_CSV = '/datasets/processed_128/train-rle.csv'

# Read CSV file
csv_file = pd.read_csv(DIR_TO_CSV)
train_csv, val_csv = random_train_val_split(csv_file, 0.2, 44)

# Create Data Loaders
train_data = BinaryLoader(train_csv, ROOT_DIR)
train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=12, 
                                             shuffle=True,
                                            )
val_data = BinaryLoader(val_csv, ROOT_DIR)
val_loader = torch.utils.data.DataLoader(val_data,
                                        batch_size=2,
                                        shuffle=False
                                        )
# Create model
model = UNet((3,512,512))

# model.load_state_dict(torch.load('/home/bohdan/SIIM-ACR_Kaggle_MedicalSegmentators/weights/unet_baseline_weighted_crossentropy0.233407.pth'))
model.to(device)

# Specify optimizer and criterion
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
NUM_OF_EPOCHS = 100

#training
train(model, train_loader, val_loader, optimizer, NUM_OF_EPOCHS, '../weights/')
