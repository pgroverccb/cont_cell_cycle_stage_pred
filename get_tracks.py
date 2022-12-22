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
    f = open(sequence['lineage_graph_path'], "r")
    text = f.read()
    text = json.loads(text)
    if (sequence['key_present']):
        text = text[sequence['key_name']]
    # print(text['Edges'])
    # text['Edges'] = text['Edges'][sequence['offset'] : ]
    start_timestep = sequence['start_timestep']
    end_timestep = sequence['end_timestep']

    new_text = ""
    for timestep in range(start_timestep, end_timestep):
        for i in range(len(text['Edges'])):
            timestep_str = str(timestep)
            while (len(timestep_str) != 3):
                timestep_str = '0' + timestep_str
            if (text['Edges'][i]['EndNodes'][0][:3] == timestep_str):
                new_text += "'" + text['Edges'][i]['EndNodes'][0] + "'\t'" + text['Edges'][i]['EndNodes'][1] + "'\n"

    edges_list = new_text.split("\n")
    updated_edges_list = []
    for edge in edges_list[:-1]:
        curr_edge = edge.split("\t")
        from_point, to_point = curr_edge[0][1: - 1], curr_edge[1][1: - 1]
        updated_edges_list.append([from_point, to_point])
    updated_edges_list.sort()
    # print(updated_edges_list)

    num_nuclei_at_start = 0
    for edge in updated_edges_list:
        if (int(edge[0].split("_")[0]) == start_timestep):
            num_nuclei_at_start += 1
    
    file = open('/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/mapping_' + sequence['name'] + '.pkl','rb')
    mapping = pickle.load(file)

    completed_tracks = []
    starting_array = []
    for i in range(0, num_nuclei_at_start):
        starting_array.append([i + 1])

    for timestep in range(start_timestep, end_timestep):
        print("\n\n")
        starting_array.sort()
        print(timestep, "current population : ", starting_array)
        print(timestep, "mapping : ", mapping[timestep])
        if (sequence['five_char_indexing']):
            timestep_str = str(timestep)
            while (len(timestep_str) != 5):
                timestep_str = '0' + timestep_str
        mask = tifffile.imread(sequence['reg_labels_path'] + sequence['label_tag'] + str(timestep_str) + ".tif")
        for i in range(0, len(starting_array)):
            if (starting_array[i][timestep - start_timestep] not in mask):
                print(starting_array[i][timestep - start_timestep], "is not present at", timestep, "in volume.")
            else:
                points = np.argwhere(mask == starting_array[i][timestep - start_timestep])
                z_cent, y_cent, x_cent = np.mean(points, axis = 0)
                print(starting_array[i][timestep - start_timestep], "is present at", timestep, "with centroid", round(x_cent, 2), round(y_cent, 2), round(z_cent, 2), "and volume", len(points), "points.")
      
        to_remove_elements = []
        for i in range(0, len(starting_array)):
              # try:
              # print("Processing i", i, starting_array[i], "at timestep", timestep - start_timestep)
              to_map = mapping[timestep][starting_array[i][timestep - start_timestep]]
              if (len(to_map) == 0):
                while(len(starting_array[i]) != end_timestep - start_timestep + 1):
                    starting_array[i].append(0)
                completed_tracks.append(starting_array[i])
                to_remove_elements.append(starting_array[i])

              if (len(to_map) == 1):
                starting_array[i].append(to_map[0])

              if (len(to_map) == 2):
                while(len(starting_array[i]) != end_timestep - start_timestep):
                    starting_array[i].append(0)
                completed_tracks.append(starting_array[i])
                to_remove_elements.append(starting_array[i])
                new_daughter = []
                for i in range(timestep - start_timestep + 1):
                    new_daughter.append(0)
                new_daughter.append(to_map[0])

                new_daughter_2 = []
                for i in range(timestep - start_timestep + 1):
                    new_daughter_2.append(0)
                new_daughter_2.append(to_map[1])

                starting_array.append(new_daughter)
                starting_array.append(new_daughter_2)
        for removal in to_remove_elements:
            starting_array.remove(removal)
            # except:
            #   print("ERROR : Lost Track at ", timestep)
        if (timestep == end_timestep - 1):
            for track in starting_array:
                completed_tracks.append(track)
            break
    print("Completed Tracks : ", len(completed_tracks), completed_tracks)
    np.save("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/tracks_" + sequence['name'] + ".npy", completed_tracks)
