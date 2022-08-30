import numpy as np
import torch
from tqdm import tqdm
from pytorch_connectomics.connectomics.model.arch.unet import UNet3D
from monai.inferers import sliding_window_inference
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage
from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import dilation
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ToTensord,
)
from monai.data import (
    DataLoader,
    CacheDataset,
)
import os
import pickle
import tifffile

dataset_path = "sample_dataset/"
test_set_paths_pairs = [['image_reg_00135.tif', 'image_reg_00136.tif']]
saved_weights_path = "utils/cell_cycle_saved_model.pth"
patch_size = (16, 128, 128)

test_files = []

for pair in test_set_paths_pairs:
      input_volume_1 = tifffile.imread(dataset_path + pair[0])
      input_volume_2 = tifffile.imread(dataset_path + pair[1])

      orig_shape = input_volume_1.shape
      input_volume_1 = (input_volume_1 - np.mean(input_volume_1))/(np.std(input_volume_1))
      scaler = MinMaxScaler()
      scaler.fit(input_volume_1.flatten().reshape(-1, 1))
      input_volume_1 = scaler.transform(input_volume_1.flatten().reshape(-1, 1)).reshape(orig_shape)

      orig_shape = input_volume_2.shape
      input_volume_2 = (input_volume_2 - np.mean(input_volume_2))/(np.std(input_volume_2))
      scaler = MinMaxScaler()
      scaler.fit(input_volume_2.flatten().reshape(-1, 1))
      input_volume_2 = scaler.transform(input_volume_2.flatten().reshape(-1, 1)).reshape(orig_shape)

      image_path = dataset_path + "normalized_image.npy"
      test_files.append({'image' : image_path})
      np.save(image_path, np.array([input_volume_1, input_volume_2]))
      print("Completed operation for ", pair)



test_files = np.array(test_files)
test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
)

test_ds = CacheDataset(
    data=test_files, 
    transform=test_transforms,
    cache_num=1, cache_rate=0.0, num_workers=2
)

test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
)

test_iterator = tqdm(test_loader, desc="Testing (X / X Steps) (dice=X.X)", dynamic_ncols=True)
stage_pred_model = UNet3D(in_channel = 2, out_channel = 2, is_isotropic = True)
stage_pred_model = stage_pred_model.cuda()
stage_pred_model.load_state_dict(torch.load(saved_weights_path))
 
print("Begin Testing.")
with torch.no_grad():
    for step, batch in enumerate(test_iterator):
        val_inputs = batch["image"].cuda()
#         print("val_inputs shape : ", val_inputs.shape)
        val_outputs = sliding_window_inference(val_inputs[:, 0], patch_size, 4, stage_pred_model)
#         print("val_outputs shape : ", val_outputs.shape)
        np.save("predicted_stage_map_" + str(step) + ".npy", val_outputs[0])
        
        mask_pre = np.load("utils/predicted_seg_mask.npy")[0]
        mask_post = np.load("utils/predicted_seg_mask.npy")[1]

        for curr_index in np.unique(mask_pre)[1: ]:
                curr_volume[curr_volume != curr_index] = 0
                curr_volume[curr_volume == curr_index] = 1
                all_pixels = len(np.argwhere(curr_volume) == 1)
                curr_volume = curr_volume * val_outputs[0, 0]
                growth_avg = np.sum(curr_volume)/all_pixels
                print("Growth Stage Prediction in Pre frame for Index :", curr_index, "is :", growth_avg)

        for curr_index in np.unique(mask_post)[1: ]:
                curr_volume[curr_volume != curr_index] = 0
                curr_volume[curr_volume == curr_index] = 1
                all_pixels = len(np.argwhere(curr_volume) == 1)
                curr_volume = curr_volume * val_outputs[0, 1]
                growth_avg = np.sum(curr_volume)/all_pixels
                print("Growth Stage Prediction in Post frame for Index :", curr_index, "is :", growth_avg)
