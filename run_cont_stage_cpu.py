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
import sys
import builtins
import os

sys.stdout = open("/mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/continous_logging_cont_stage.txt", "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)

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
        z_offset = random.randint(5, 25)
        y_offset = random.randint(40, 120)
        x_offset = random.randint(80, 110)
        input = input[:, z_offset : z_offset + 128, y_offset : y_offset + 128, x_offset : x_offset + 128]
        output = output[:, z_offset : z_offset + 128, y_offset : y_offset + 128, x_offset : x_offset + 128]
        return input, output

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 8}
# max_epochs = 10

# Datasets
partition = {'train' : [], 'validation' : []}
for i in range(50, 130):
    prob = random.random()
    if i in [55, 60, 65, 70, 75, 80, 95, 100, 110, 120]:
        partition['validation'].append(i)
    else:
        partition['train'].append(i)

# Generators
training_set = Dataset(partition['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'])
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

stage_pred_model = UNet3D(in_channel = 2, out_channel = 2, is_isotropic = True)
# stage_pred_model = stage_pred_model.cuda()
# stage_pred_model.load_state_dict(torch.load("/mnt/ceph/users/pgrover/saved_model_.pth"))

mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(stage_pred_model.parameters(), lr=0.002)
sig = nn.Sigmoid()
print("Begin training.")
for e in range(1, 200+1):
    stage_pred_model.train()
    batch_num = 0
    train_loss_avg = 0.0
    val_loss_avg = 0.0

    for X_train_batch, y_train_batch in training_generator:
        batch_num += 1
        X_train_batch, y_train_batch = X_train_batch.to('cpu', dtype = torch.float), y_train_batch.to('cpu', dtype = torch.float)
        optimizer.zero_grad()
        y_train_pred = stage_pred_model(X_train_batch)
        l2_growth_pre = mse_loss(sig(y_train_pred[0, 0, :, :, :]), y_train_batch[0, 0, :, :, :]) * 100
        l2_growth_post = mse_loss(sig(y_train_pred[0, 1, :, :, :]), y_train_batch[0, 1, :, :, :]) * 100
        print("T : L2 Growth Pre : " + str(round(l2_growth_pre.item(), 3)) + "| L2 Growth Post : " + str(round(l2_growth_post.item(), 3)))
        train_loss = (l2_growth_pre + l2_growth_post)
        train_loss_avg += train_loss.item()
        train_loss.backward()
        optimizer.step()

    for X_val_batch, y_val_batch in validation_generator:
        batch_num += 1
        X_val_batch, y_val_batch = X_val_batch.to('cpu', dtype = torch.float), y_val_batch.to('cpu', dtype = torch.float)
        optimizer.zero_grad()
        y_val_pred = stage_pred_model(X_val_batch)
        l2_growth_pre = mse_loss(sig(y_val_pred[0, 0, :, :, :]), y_val_batch[0, 0, :, :, :]) * 100
        l2_growth_post = mse_loss(sig(y_val_pred[0, 1, :, :, :]), y_val_batch[0, 1, :, :, :]) * 100
        print("V : L2 Growth Pre : " + str(round(l2_growth_pre.item(), 3)) + "| L2 Growth Post : " + str(round(l2_growth_post.item(), 3)))
        val_loss = (l2_growth_pre + l2_growth_post)
        val_loss_avg += val_loss.item()
        optimizer.step()
    # print("Epoch : " + str(e) + "Train Loss : " + str(round(train_loss_avg/len(partition['train']), 3)) + "Val Loss : " + str(round(val_loss_avg/len(partition['validation']), 3)))
    # torch.save(stage_pred_model.state_dict(), "/mnt/ceph/users/pgrover/saved_model_" + str(e) + ".pth")

    testing_partition = []
    for i in range(130, 131):
        testing_partition.append(i)

    print("Begin Testing.")
    for ID in testing_partition[:1]:
        file = open("/mnt/ceph/users/pgrover/growth_field_dataset/sample_" + str(ID) + ".pkl", 'rb')
        sample_full = pickle.load(file)
        input = torch.Tensor(sample_full['input'].reshape((1, sample_full['input'].shape[0], sample_full['input'].shape[1], sample_full['input'].shape[2], sample_full['input'].shape[3])))
        output = torch.Tensor(sample_full['output'].reshape((1, sample_full['output'].shape[0], sample_full['output'].shape[1], sample_full['output'].shape[2], sample_full['output'].shape[3])))
        input = input[:, :, 13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
        output = output[:, :, 13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
        # input, output = input.to('cuda', dtype = torch.float), output.to('cuda', dtype = torch.float)
        output_pred = sig(stage_pred_model(input))
        # np.save("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/testing_pred_" + str(ID) + ".npy", output_pred.detach().cpu().numpy())


    for index in testing_partition[:1]:
        print("Index : " + str(index))
        # output_pred = np.load("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/testing_pred_" + str(index) + ".npy")
        file = open("/mnt/ceph/users/pgrover/growth_field_dataset/sample_" + str(ID) + ".pkl", 'rb')
        sample_full = pickle.load(file)
        output = torch.Tensor(sample_full['output'].reshape((1, sample_full['output'].shape[0], sample_full['output'].shape[1], sample_full['output'].shape[2], sample_full['output'].shape[3])))    
        output = output[:, :, 13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
        five_digit_str = str(index)
        while (len(five_digit_str) != 5):
            five_digit_str = '0' + five_digit_str
        mask_pre = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")

        five_digit_str = str(index + 1)
        while (len(five_digit_str) != 5):
            five_digit_str = '0' + five_digit_str
        mask_post = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")

        for curr_index in np.unique(mask_pre)[6: 12]:
            curr_volume = np.copy(mask_pre)[13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
            curr_volume[curr_volume != curr_index] = 0
            curr_volume[curr_volume == curr_index] = 1
            all_pixels = len(np.argwhere(curr_volume) == 1)
            curr_volume = curr_volume * output_pred[0, 0].detach().numpy()
            growth_avg = np.sum(curr_volume) * 1.0 / all_pixels
            print("Growth stage pred in pre frame for " + str(curr_index) + " is : " + str(round(growth_avg, 3)))
            # print(" ")
            curr_volume = np.copy(mask_pre)[13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
            curr_volume[curr_volume != curr_index] = 0
            curr_volume[curr_volume == curr_index] = 1
            all_pixels = len(np.argwhere(curr_volume) == 1)
            curr_volume = curr_volume * output[0, 0].detach().numpy()
            growth_avg = np.sum(curr_volume) * 1.0 / all_pixels
            print("Growth stage actual in pre frame for " + str(curr_index) + " is : " + str(round(growth_avg, 3)))
            print(" ")
        print(" ")
        print(" ")

        for curr_index in np.unique(mask_post)[6: 12]:
            curr_volume = np.copy(mask_post)[13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
            curr_volume[curr_volume != curr_index] = 0
            curr_volume[curr_volume == curr_index] = 1
            all_pixels = len(np.argwhere(curr_volume) == 1)
            curr_volume = curr_volume * output_pred[0, 1].detach().numpy()
            growth_avg = np.sum(curr_volume) * 1.0 / all_pixels
            print("Growth stage pred in post frame for " + str(curr_index) + " is : " + str(round(growth_avg, 3)))
            # print(" ")
            curr_volume = np.copy(mask_post)[13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
            curr_volume[curr_volume != curr_index] = 0
            curr_volume[curr_volume == curr_index] = 1
            all_pixels = len(np.argwhere(curr_volume) == 1)
            curr_volume = curr_volume * output[0, 1].detach().numpy()
            growth_avg = np.sum(curr_volume) * 1.0 / all_pixels
            print("Growth stage actual in post frame for " + str(curr_index) + "is :" + str(round(growth_avg, 3)))
            print(" ")
        print(" ")
        print(" ")
