from pytorch_connectomics.connectomics.model.arch.unet import UNet3D
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import tifffile
import json
import random
import pickle

class Dataset(torch.utils.data.Dataset):
  def __init__(self, list_IDs):
        self.list_IDs = list_IDs

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        ID = self.list_IDs[index]
        # print("Loading ID : ", ID)
        file = open("/mnt/ceph/users/pgrover/growth_field_dataset/sample_" + str(ID + 1) + ".pkl", 'rb')
        sample = pickle.load(file)
        input = sample['input']
        output = sample['output']
        z_offset = 0
        y_offset = 0
        x_offset = 0
        input = input[:, z_offset : z_offset + 128, y_offset : y_offset + 128, x_offset : x_offset + 128]
        output = output[:, z_offset : z_offset + 128, y_offset : y_offset + 128, x_offset : x_offset + 128]
        return input, output

# Parameters
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 0}
max_epochs = 10

# Datasets
partition = {'train' : [], 'validation' : []}
for i in range(50, 130):
    prob = random.random()
    if (prob > 0.85):
        partition['validation'].append(i)
    else:
        partition['train'].append(i)

# Generators
training_set = Dataset(partition['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'])
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

stage_pred_model = UNet3D(in_channel = 2, out_channel = 2, is_isotropic = True)
stage_pred_model = stage_pred_model.cuda()

mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(stage_pred_model.parameters(), lr=0.001)
print("Begin training.")
for e in range(1, 10+1):
    stage_pred_model.train()
    batch_num = 0
    train_loss_avg = 0.0
    val_loss_avg = 0.0

    for X_train_batch, y_train_batch in training_generator:
        batch_num += 1
        X_train_batch, y_train_batch = X_train_batch.to('cuda', dtype = torch.float), y_train_batch.to('cuda', dtype = torch.float)
        optimizer.zero_grad()
        y_train_pred = stage_pred_model(X_train_batch)
        l2_growth_pre = mse_loss(y_train_pred[0, 0, :, :, :], y_train_batch[0, 0, :, :, :])
        l2_growth_post = mse_loss(y_train_pred[0, 1, :, :, :], y_train_batch[0, 1, :, :, :])
        # print("L1 x : ", round(l2_x.item(), 3), "| L1 y: ", round(l2_y.item(), 3), "| L1 z: ", round(l2_z.item(), 3))
        train_loss = l2_growth_pre + l2_growth_post
        train_loss_avg += train_loss.item()
        train_loss.backward()
        optimizer.step()

    for X_val_batch, y_val_batch in validation_generator:
        batch_num += 1
        X_val_batch, y_val_batch = X_val_batch.to('cuda', dtype = torch.float), y_val_batch.to('cuda', dtype = torch.float)
        optimizer.zero_grad()
        y_val_pred = stage_pred_model(X_val_batch)
        l2_growth_pre = mse_loss(y_val_pred[0, 0, :, :, :], y_val_batch[0, 0, :, :, :])
        l2_growth_post = mse_loss(y_val_pred[0, 1, :, :, :], y_val_batch[0, 1, :, :, :])
        # print("L1 x : ", round(l2_x.item(), 3), "| L1 y: ", round(l2_y.item(), 3), "| L1 z: ", round(l2_z.item(), 3))
        val_loss = l2_growth_pre + l2_growth_post
        val_loss_avg += val_loss.item()
        optimizer.step()
    print("Epoch : ", e, "Train Loss : ", round(train_loss_avg/len(partition['train']), 3), "Val Loss : ", round(val_loss_avg/len(partition['validation']), 3))
torch.save(stage_pred_model.state_dict(), "/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/saved_model.pth")