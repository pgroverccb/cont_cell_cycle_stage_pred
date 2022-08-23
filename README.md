## Predicting Continuous Cell Cycle Stage from Raw Images

### Running Saved Model on External Raw Images

The model is part of a bigger end-to-end system for segmentation and tracking, and thus requires instance segmentation masks as input, where background is labelled as 0, and foreground instances are labelled [1, 2, 3 ... ]. For demonstration, U3D-BCD network can be used for this purpose. Additional MONAI library is utilized for sliding window approach (available [here](https://drive.google.com/drive/u/0/folders/1A_q8lcUjO-rUbi0iwppIXCzHofhdZOFX). Saved model is placed in utils/u3d_bcd_saved_model.pth.

* **pred_inst_u3d_bcd.py** -  _generates instance segmentation masks from a raw unannotated image._ 

* **pred_external_cont_stage_sliding.py** - _uses a sliding window apporach predict stage maps for arbitrary sized volumes, coupled with predicted segmentation masks, provides stage of growth for each instance present._

---

### Methodology

#### To construct GT Dataset : 

Using Lineage Trees and Segmentation Masks, it is possible to obtain how close each nuclei in a sample volume is to its division.

* **explore_lineage.py** -  _provides a tree-like view of the dataset, with timesteps and IDs at each division._ 

~~~~
Run on cluster with scripts/script_explore_lineage.slurm, expected output for F32_40 Sequence : logs/log_explore_lineage.1542441.out, mapping between each frame is saved as utils/mapping.pkl.
~~~~

* **get_tracks.py** - _extracts tracks providing timestep-linked IDs of each unique instance present in time sequence._

~~~~
Run on cluster with scripts/script_get_tracks.slurm, expected output for F32_40 Sequence : logs/log_get_tracks.1542455.out, completed tracks are saved as utils/tracks.npy.
~~~~

~~~~
For a sequence of 200 frames, a daughter nuclei formed at 65th frame, and dividing at 105th frame, would have the following track : 
[0, 0, ... (64x) .. 0, 5, 7, 2, ... (40 numbered IDs) .., 8, 12, 2, 0, 0, 0, ... (95x) .. 0]
~~~~

~~~~
In a sequence which starts at 8 nuclei and ends at 128, the array should have 8 + 16 + 32 + 64 = 120 completed tracks, and 128 incomplete tracks.
~~~~

* **get_cont_cycle_stages.py** - _uses tracks of nuclei, to compute a continuous measure for cell cycle stage at each timestep and provides an (timesteps, unique_tracks) array for the same._

~~~~
Run on cluster with scripts/script_get_cont_cycle_stages.slurm, expected output for F32_40 Sequence : logs/log_get_cont_cycle_stages.1542465.out, completed tracks are saved as utils/growth_stages.npy.
~~~~

~~~~
In a sequence of 60 frames, timestep 35 contains two nuclei x (formed at 22 and further divides at 58), and y (formed at 31 and continues till 60), stage for x is computed as : ((35 - 22)/(58 - 22)) ^ 1.5 =  while stage for y is computed as : ((35 - 31)/(default_40)) ^ 1.5 = 
~~~~

* **form_dataset.py** - _formulates dataset from time sequence, where each sample is formed of k frames, along with their cell-cycle stage maps._

~~~~
Run on cluster with scripts/script_form_dataset.slurm, expected output for F32_40 Sequence : logs/log_form_dataset.1542472.out, dataset is saved in the form of two-frame raw image input and corresponding stage maps as output.
~~~~

#### To train and test network : 

3D U-Net is used to generate a two-frame map of stage of growth of nuclei, and their proximity to division. Architecture of U-Net is adopted from PyTorch Connectomics Library.

Supporting data including PyTorch Connectomics Framework (relevant files) and Sample Dataset Files are shared with [Google Drive](https://drive.google.com/drive/u/0/folders/1A_q8lcUjO-rUbi0iwppIXCzHofhdZOFX)

* **run_cont_stage.py** - _is the script used for training and validating the network, randomized crop to 32 x 128 x 128 is used for each volume._

* **pred_cont_stage_sliding.py** - _uses a sliding window apporach predict stage maps for arbitrary sized volumes, coupled with segmentation masks, provides stage of growth for each instance present, and displays accuracy of model._
