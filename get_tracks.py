import numpy as np
import tifffile
import pickle

file = open("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/mapping.pkl",'rb')
mapping = pickle.load(file)

completed_tracks = []
starting_array = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16]]
for timestep in range(50, 135):
    print("\n\n")
    starting_array.sort()
    print(timestep, "current population : ", starting_array)
    print(timestep, "mapping : ", mapping[timestep])
    timestep_str = str(timestep)
    while (len(timestep_str) != 5):
        timestep_str = '0' + timestep_str
    mask = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(timestep_str) + ".tif")
    for i in range(0, len(starting_array)):
        if (starting_array[i][timestep - 50] not in mask):
            print(starting_array[i][timestep - 50], "is not present at", timestep, "in volume.")
        else:
            points = np.argwhere(mask == starting_array[i][timestep - 50])
            z_cent, y_cent, x_cent = np.mean(points, axis = 0)
            print(starting_array[i][timestep - 50], "is present at", timestep, "with centroid", round(x_cent, 2), round(y_cent, 2), round(z_cent, 2), "and volume", len(points), "points.")
  
    to_remove_elements = []
    for i in range(0, len(starting_array)):
        # try:
          # print("Processing i", i, starting_array[i], "at timestep", timestep - 50)
          to_map = mapping[timestep][starting_array[i][timestep - 50]]
          if (len(to_map) == 0):
            while(len(starting_array[i]) != 86):
                starting_array[i].append(0)
            completed_tracks.append(starting_array[i])
            to_remove_elements.append(starting_array[i])

          if (len(to_map) == 1):
            starting_array[i].append(to_map[0])

          if (len(to_map) == 2):
            while(len(starting_array[i]) != 85):
                starting_array[i].append(0)
            completed_tracks.append(starting_array[i])
            to_remove_elements.append(starting_array[i])
            new_daughter = []
            for i in range(timestep - 50 + 1):
                new_daughter.append(0)
            new_daughter.append(to_map[0])

            new_daughter_2 = []
            for i in range(timestep - 50 + 1):
                new_daughter_2.append(0)
            new_daughter_2.append(to_map[1])

            starting_array.append(new_daughter)
            starting_array.append(new_daughter_2)
    for removal in to_remove_elements:
        starting_array.remove(removal)
        # except:
        #   print("ERROR : Lost Track at ", timestep)
    if (timestep == 134):
        for track in starting_array:
            completed_tracks.append(track)
        break
print("Completed Tracks : ", len(completed_tracks), completed_tracks)
np.save("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/tracks.npy", completed_tracks)
