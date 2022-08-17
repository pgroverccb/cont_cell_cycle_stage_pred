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

testing_partition = []
for i in range(130, 133):
    testing_partition.append(i)

stage_pred_model = UNet3D(in_channel = 2, out_channel = 2, is_isotropic = True)
# stage_pred_model = stage_pred_model.cuda()
stage_pred_model.load_state_dict(torch.load("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/saved_model.pth", map_location=torch.device('cpu')))
 
print("Begin Testing.")
for ID in testing_partition:
    file = open("/mnt/ceph/users/pgrover/growth_field_dataset/sample_" + str(ID) + ".pkl", 'rb')
    sample_full = pickle.load(file)
    input = torch.Tensor(sample_full['input'].reshape((1, sample_full['input'].shape[0], sample_full['input'].shape[1], sample_full['input'].shape[2], sample_full['input'].shape[3])))
    output = torch.Tensor(sample_full['output'].reshape((1, sample_full['output'].shape[0], sample_full['output'].shape[1], sample_full['output'].shape[2], sample_full['output'].shape[3])))
    input = input[:, :, 13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
    output = output[:, :, 13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
    # input, output = input.to('cuda', dtype = torch.float), output.to('cuda', dtype = torch.float)
    output_pred = stage_pred_model(input)
    np.save("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/testing_pred_" + str(ID) + ".npy", output_pred.detach().cpu().numpy())

for index in testing_partition:
    output_pred = np.load("/mnt/home/pgrover/cont_cell_cycle_stage_pred/utils/testing_pred_" + str(index) + ".npy")
    five_digit_str = str(index)
    while (len(five_digit_str) != 5):
        five_digit_str = '0' + five_digit_str
    mask_pre = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")

    five_digit_str = str(index + 1)
    while (len(five_digit_str) != 5):
        five_digit_str = '0' + five_digit_str
    mask_post = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")

    for curr_index in np.unique(mask_pre)[1: ]:
        curr_volume = np.copy(mask_pre)[13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
        curr_volume[curr_volume != curr_index] = 0
        curr_volume[curr_volume == curr_index] = 1
        all_pixels = len(np.argwhere(curr_volume) == 1)
        curr_volume = curr_volume * output[0, 0].detach().numpy()
        growth_avg = np.sum(curr_volume)/all_pixels
        print("Growth Stage in Pre frame for Index :", curr_index, "is :", growth_avg)
    
    for curr_index in np.unique(mask_post)[1: ]:
        curr_volume = np.copy(mask_post)[13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
        curr_volume[curr_volume != curr_index] = 0
        curr_volume[curr_volume == curr_index] = 1
        all_pixels = len(np.argwhere(curr_volume) == 1)
        curr_volume = curr_volume * output[0, 1].detach().numpy()
        growth_avg = np.sum(curr_volume)/all_pixels
        print("Growth Stage in Post frame for Index :", curr_index, "is :", growth_avg)
