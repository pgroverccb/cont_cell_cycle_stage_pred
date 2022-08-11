import numpy as np
import tifffile
import pickle

x = np.load("/mnt/home/pgrover/continous_cell_cycle_stage_pred/utils/growth_stages.npy", allow_pickle=True)

for index in range(50, 134):
    five_digit_str = str(index)
    while (len(five_digit_str) != 5):
        five_digit_str = '0' + five_digit_str
    im_pre = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_images/image_reg_" + str(five_digit_str) + ".tif")
    lb_pre = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")
    five_digit_str = str(index + 1)
    while (len(five_digit_str) != 5):
        five_digit_str = '0' + five_digit_str
    im_post = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_images/image_reg_" + str(five_digit_str) + ".tif")
    lb_post = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")

    growth_pre = np.zeros((160, 392, 240))
    growth_post = np.zeros((160, 392, 240))

    objects_pre = np.unique(lb_pre)
    mitotic_pre = []
    for object_id in objects_pre[1: ]:
        mitotic_pre.append(x[index - 50][object_id - 1])

    objects_post = np.unique(lb_post)
    mitotic_post = []
    for object_id in objects_post[1: ]:
        mitotic_post.append(x[index - 50 + 1][object_id - 1])

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
    file_pointer = open("/mnt/ceph/users/pgrover/growth_field_dataset/sample_" + str(index) + ".pkl", "wb")
    pickle.dump(sample, file_pointer)
