import json
import numpy as np
import tifffile
import pickle

import yaml
with open('config.yaml') as config_file:    
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    config_dataset = config['dataset_sequences']

print("Datasets Processing : ")
for sequence in config_dataset:
    print(sequence['name'] + ", " + str(sequence['start_timestep']) + " -> " + str(sequence['end_timestep']))

for sequence in config_dataset:
    x = list(np.load("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/tracks_" + sequence['name'] + ".npy", allow_pickle=True))
    for y in x:
        if len(y) == (sequence['end_timestep'] - sequence['start_timestep']):
            y.append(0)

    x = np.array(x)
    growth_stages = []
    for timestep in range(sequence['start_timestep'], sequence['end_timestep']):
        print("\n\n")
        print("at timestep : ", timestep)
        timestep_str = str(timestep)
        if (sequence['five_char_indexing']):
            while (len(timestep_str) != 5):
                timestep_str = '0' + timestep_str
        mask = tifffile.imread(sequence['reg_labels_path'] + sequence['label_tag'] + str(timestep_str) + ".tif")
        objects_present = np.unique(mask)[1: ]
        current_stage = []
        for object_id in objects_present:
          for i in range(0, len(x)):
              if (x[i][timestep - sequence['start_timestep']] == object_id):
                  starting = 0
                  end = 60
                  current = timestep - sequence['start_timestep']
                  seq = np.copy(x[i])
                  if (seq[0] == 0):
                      counter = 0
                      while(seq[counter] == 0):
                          counter += 1
                      # print(object_id, 'was child until', counter + sequence['start_timestep'])
                      starting = counter
                      end = starting + 60
                  if (seq[-1] == 0):
                      counter = len(seq) - 1
                      while(seq[counter] == 0):
                          counter -= 1
                      end = counter
                      # print(object_id, 'will divide at', counter + sequence['start_timestep'])
                  # print("Stage of Growth : ", pow((current - starting)/(end - starting), 2))
                  current_stage.append(pow((current - starting)/(end - starting), 2))
        growth_stages.append(current_stage)
    np.save("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/growth_stages_" + sequence['name'] + ".npy", growth_stages)
