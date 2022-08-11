import json
import numpy as np
import tifffile
import pickle

f = open("/mnt/ceph/users/pgrover/32_40_dataset/Lineages/GT_tracking_F32_to_F40.json", "r")
text = f.read()
text = json.loads(text)
text = text['G_based_on_nn']

new_text = ""
for num in range(50, 135):
    count = 0
    for i in range(len(text['Edges'])):
        timestep_str = str(num)
        while (len(timestep_str) != 3):
            timestep_str = '0' + timestep_str
        if (text['Edges'][i]['EndNodes'][0][:3] == timestep_str):
            count += 1
            new_text += "'" + text['Edges'][i]['EndNodes'][0] + "'\t'" + text['Edges'][i]['EndNodes'][1] + "'\n"

edges_list = new_text.split("\n")
updated_edges_list = []
for edge in edges_list[:-1]:
    curr_edge = edge.split("\t")
    from_point, to_point = curr_edge[0][1: - 1], curr_edge[1][1: - 1]
    updated_edges_list.append([from_point, to_point])
updated_edges_list.sort()

centroids = {}
pixels = {}
mapping = {}

def get_div_step(timestep_start, obj_id, chain_space):
    initial_obj_id = obj_id
    for timestep in range(timestep_start, 135):
        if ((str(obj_id) + "_" + str(timestep_start)) not in centroids.keys()):
            centroids[str(obj_id) + "_" + str(timestep_start)] = []
            pixels[str(obj_id) + "_" + str(timestep_start)] = []
        timestep_str = str(timestep)
        while (len(timestep_str) != 3):
            timestep_str = '0' + timestep_str
        out_map_to = []
        for edge in updated_edges_list:
            if (edge[0][0:3] == timestep_str):
                if (int(edge[0][4:7]) == obj_id):
                    out_map_to.append(int(edge[1][4:7]))

        centroids[str(initial_obj_id) + "_" + str(timestep_start)].append([1, 1, 1])
        pixels[str(initial_obj_id) + "_" + str(timestep_start)].append([1])

        if (timestep not in mapping):
            mapping[timestep] = {}

        if (len(out_map_to) == 0):
            print(chain_space, "Ended at : ", timestep_str)
            mapping[timestep][obj_id] = []
            return 0
        
        if (len(out_map_to) == 1):
            if (int(timestep_str)%10 == 0):
                print(chain_space, "Continued to : ", timestep_str, " ... ")
            mapping[timestep][obj_id] = out_map_to
            # return 0

        if (len(out_map_to) == 2):
            print(" ")
            child_1_obj_id = out_map_to[0] 
            child_2_obj_id = out_map_to[1] 
            mapping[timestep][obj_id] = out_map_to
            return timestep, child_1_obj_id, child_2_obj_id
        
        obj_id = out_map_to[0]
    return 0


chain_space = ""
def recurs(step, id, chain_space):
    print(chain_space, "Timestep : ", step, "ID : ", id)
    zzz = get_div_step(step, id, chain_space)
    if (zzz == 0):
        print(chain_space, " >  Reached End of Tracked Sequence ")
        chain_space = chain_space[:-3]
        return
    timestep, child_1_obj_id, child_2_obj_id = zzz
    chain_space += " >  "
    print(chain_space, "Division into : ", child_1_obj_id, " and ", child_2_obj_id, " at ", timestep)
    print("   ")
    return recurs(timestep + 1, child_1_obj_id, chain_space), recurs(timestep + 1, child_2_obj_id, chain_space)

isolated_divisions_list = []
for id in range(1, 17):
    print("Processing ID : ", id)
    recurs(50, id, chain_space)

curr_file = open('/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/mapping.pkl', 'wb')
pickle.dump(mapping, curr_file)
curr_file.close()
