import numpy as np
import torch
from tqdm import tqdm
from pytorch_connectomics.connectomics.model.arch.unet import UNet3D
from monai.inferers import sliding_window_inference
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
import pickle

dataset_path = "sample_dataset/"
test_set_paths_pairs = [['1.npy', '2.npy']
saved_weights_path = "utils/cell_cycle_saved_model.pth"
patch_size = (16, 128, 128)

if not os.path.isdir(inference_path):
      os.makedirs(inference_path)
      print("Created Inference Folder", inference_path)
else:
      print("Folder already exists")

test_files = []

for pair in test_set_paths_pairs:
      input_volume_1 = np.load(pair[0])
      input_volume_2 = np.load(pair[1])

      orig_shape = input_volume_1.shape
      input_volume_1 = (input_volume_1 - np.mean(input_volume_1))/(np.std(input_volume_1))
      scaler = MinMaxScaler()
      scaler.fit(input_volume_1.flatten().reshape(-1, 1))
      input_volume_1 = scaler.transform(input_volume_1.flatten().reshape(-1, 1)).reshape(orig_shape)

      orig_shape = input_volume_2.shape
      input_volume_2 = (input_volume_2 - np.mean(input_volume_2)/(np.std(input_volume_2))
      scaler = MinMaxScaler()
      scaler.fit(input_volume_2.flatten().reshape(-1, 1))
      input_volume_2 = scaler.transform(input_volume_2.flatten().reshape(-1, 1)).reshape(orig_shape)

      image_path = inference_path + "image"
      test_files.append({'image' : image_path})
      np.save(image_path, np.array(input_volume_1, input_volume_2))
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
# stage_pred_model = stage_pred_model.cuda()
stage_pred_model.load_state_dict(torch.load(saved_weights_path, map_location=torch.device('cpu')))
 
print("Begin Testing.")
with torch.no_grad():
    for step, batch in enumerate(test_iterator):
        val_inputs = batch["image"]
        val_outputs = sliding_window_inference(val_inputs[:, :, :, :, :], patch_size, 4, model)
        np.save("predicted_stage_map_" + str(step) + ".npy", val_outputs[0])
        
        mask_pre = np.load("predicted_seg_mask_pre_" + current_label + ".npy")
        mask_post = np.load("predicted_seg_mask_post_" + current_label + ".npy")

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
