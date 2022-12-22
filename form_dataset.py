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
    x = np.load("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/growth_stages_" + sequence['name'] + ".npy", allow_pickle=True)
    start_timestep = sequence['start_timestep']
    end_timestep = sequence['end_timestep']
    for index in range(sequence['start_timestep'], sequence['end_timestep'] - 2):
        timestep_str = str(index)
        if (sequence['five_char_indexing']):
            while (len(timestep_str) != 5):
                timestep_str = '0' + timestep_str
        im_pre = tifffile.imread(sequence['reg_images_path'] + "/image_reg_" + str(timestep_str) + ".tif")
        lb_pre = tifffile.imread(sequence['reg_labels_path'] + "/label_reg_" + str(timestep_str) + ".tif")
        timestep_str = str(index + 1)
        if (sequence['five_char_indexing']):
            while (len(timestep_str) != 5):
                timestep_str = '0' + timestep_str
        im_post = tifffile.imread(sequence['reg_images_path'] + "/image_reg_" +  str(timestep_str) + ".tif")
        lb_post = tifffile.imread(sequence['reg_labels_path'] + "/label_reg_" +  str(timestep_str) + ".tif")

        growth_pre = np.zeros((sequence['z_dim'], sequence['y_dim'], sequence['x_dim']))
        growth_post = np.zeros((sequence['z_dim'], sequence['y_dim'], sequence['x_dim']))

        objects_pre = np.unique(lb_pre)
        mitotic_pre = []
        for object_id in objects_pre[1: ]:
            mitotic_pre.append(x[index - start_timestep][object_id - 1])

        objects_post = np.unique(lb_post)
        mitotic_post = []
        for object_id in objects_post[1: ]:
            mitotic_post.append(x[index - start_timestep + 1][object_id - 1])

        for object_id in range(1, len(mitotic_pre)):
            print("Obj Id : ", object_id, "Growth : ", mitotic_pre[object_id])
            # growth_pre[growth_pre == object_id] += mitotic_pre[object_id]
            points = np.argwhere(lb_pre == object_id + 1)
            for point in points:
                growth_pre[point[0], point[1], point[2]] = mitotic_pre[object_id]

        for object_id in range(0, len(mitotic_post)):
            print("Obj Id : ", object_id, "Growth : ", mitotic_post[object_id])
            points = np.argwhere(lb_post == object_id + 1)
            for point in points:
                growth_post[point[0], point[1], point[2]] = mitotic_post[object_id]
        sample = {'input' : np.array([im_pre, im_post]), 'output' : np.array([growth_pre, growth_post])}
        file_pointer = open("/mnt/ceph/users/pgrover/" + sequence['name'] + "_growth_field_dataset/sample_" + str(index) + ".pkl", "wb")
        pickle.dump(sample, file_pointer)
