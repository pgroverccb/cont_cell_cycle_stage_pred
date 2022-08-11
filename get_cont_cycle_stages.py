import numpy as np
import tifffile

x = list(np.load("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/tracks.npy", allow_pickle=True))
for y in x:
    if len(y) == 85:
        y.append(0)

x = np.array(x)
growth_stages = []
for timestep in range(50, 135):
    print("\n\n")
    print("at timestep : ", timestep)
    timestep_str = str(timestep)
    while (len(timestep_str) != 5):
        timestep_str = '0' + timestep_str
    mask = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(timestep_str) + ".tif")
    objects_present = np.unique(mask)[1: ]
    current_stage = []
    for object_id in objects_present:
      for i in range(0, len(x)):
          if (x[i][timestep - 50] == object_id):
              starting = 0
              end = 60
              current = timestep - 50
              seq = np.copy(x[i])
              if (seq[0] == 0):
                  counter = 0
                  while(seq[counter] == 0):
                      counter += 1
                  # print(object_id, 'was child until', counter + 50)
                  starting = counter
                  end = starting + 60
              if (seq[-1] == 0):
                  counter = len(seq) - 1
                  while(seq[counter] == 0):
                      counter -= 1
                  end = counter
                  # print(object_id, 'will divide at', counter + 50)
              # print("Stage of Growth : ", pow((current - starting)/(end - starting), 2))
              current_stage.append(pow((current - starting)/(end - starting), 2))
    growth_stages.append(current_stage)
np.save("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/growth_stages.npy", growth_stages)
