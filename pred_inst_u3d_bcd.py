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

inference_path = "pred_masks"
test_set_labels = ['F55_185']
saved_weights_path = "utils/u3d_bcd_saved_model.pth"
patch_size = (16, 128, 128)

if not os.path.isdir(inference_path):
      os.makedirs(inference_path)
      print("Created Inference Folder", inference_path)
else:
      print("Folder already exists")

test_files = []

for series in test_set_labels:
      input_volume = np.load("/content/drive/MyDrive/TestingSet_2022/" + series + "/" + series + "/images/" + series + "_image_0001.npy")
      orig_shape = input_volume.shape
      input_volume = (input_volume - np.mean(input_volume))/(np.std(input_volume))
      scaler = MinMaxScaler()
      scaler.fit(input_volume.flatten().reshape(-1, 1))
      input_volume = scaler.transform(input_volume.flatten().reshape(-1, 1)).reshape(orig_shape)
      image_path = inference_path + "image_" + series + ".npy"
      test_files.append({'image' : image_path})
      np.save(image_path, input_volume)
      print("Completed operation for ", series)

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
model = UNet3D()
model.load_state_dict(torch.load(saved_weights_path, map_location=torch.device('cpu')))
# model = model.cuda()
model.eval()

def getSegType(mid):
    m_type = np.uint64
    return m_type

def cast2dtype(segm):
    max_id = np.amax(np.unique(segm))
    m_type = getSegType(int(max_id))
    return segm.astype(m_type)
    
def remove_small_instances(segm: np.ndarray, 
                           thres_small: int = 128, 
                           mode: str = 'background'):
    assert mode in ['none', 
                    'background', 
                    'background_2d', 
                    'neighbor',
                    'neighbor_2d']
    if mode == 'none':
        return segm
    if mode == 'background':
        return remove_small_objects(segm, thres_small)
    elif mode == 'background_2d':
        temp = [remove_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)

    if mode == 'neighbor':
        return merge_small_objects(segm, thres_small, do_3d=True)
    elif mode == 'neighbor_2d':
        temp = [merge_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)

def merge_small_objects(segm, thres_small, do_3d=False):
    struct = np.ones((1,3,3)) if do_3d else np.ones((3,3))
    indices, counts = np.unique(segm, return_counts=True)
    for i in range(len(indices)):
        idx = indices[i]
        if counts[i] < thres_small:
            temp = (segm == idx).astype(np.uint8)
            coord = bbox_ND(temp, relax=2)
            cropped = crop_ND(temp, coord)

            diff = dilation(cropped, struct) - cropped
            diff_segm = crop_ND(segm, coord)
            diff_segm[np.where(diff==0)]=0

            u, ct = np.unique(diff_segm, return_counts=True)
            if len(u) > 1 and u[0] == 0:
                u, ct = u[1:], ct[1:]
            segm[np.where(segm==idx)] = u[np.argmax(ct)]
    return segm

def bcd_watershed(semantic, boundary, distance, thres1=0.9, thres2=0.8, thres3=0.85, thres4=0.5, thres5=0.0, thres_small=128, 
                  scale_factors=(1.0, 1.0, 1.0), remove_small_mode='background', seed_thres=32, return_seed=False):
    seed_map = (semantic > thres1) * (boundary < thres2) * (distance > thres4)
    foreground = (semantic > thres3) * (distance > thres5)
    seed = label(seed_map)
    seed = remove_small_objects(seed, seed_thres)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
        
    if not return_seed:
        return cast2dtype(segm)

    return cast2dtype(segm), seed

with torch.no_grad():
    for step, batch in enumerate(test_iterator):
        current_label = test_set_labels[step]
        print(" ")
        print("Processing : ", current_label)
        val_inputs = batch["image"]
        val_outputs = sliding_window_inference(val_inputs[:, :, :, :, :], patch_size, 4, model)
        np.save(inference_path + "unet_outputs_" + current_label + ".npy", val_outputs)
        out = bcd_watershed(val_outputs[0, 0], val_outputs[0, 1], val_outputs[0, 2], thres1 = 20, thres2 = -40, thres3 = -10, thres4 = -15, thres5 = -0.3, thres_small = 256, seed_thres = 64)
        np.save(inference_path + "predicted_seg_mask_" + current_label + ".npy", out)
