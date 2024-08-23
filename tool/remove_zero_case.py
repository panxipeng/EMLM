import os
import h5py
import numpy as np
import shutil
from tqdm import tqdm

data_path = "/data1/data/LIDC/processed/lesion_warehouse/init"
# target_path = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/new_data_2d/zero_case"
file_list = os.listdir(data_path)
all_num = len(file_list)
count = 0
for file in tqdm(file_list):
    if '.h5' not in file:
        continue
    h5f = h5py.File(os.path.join(data_path, file), "r")
    # label = h5f["label"][:]
    # image = h5f["image"][:]
    lesion_arr = h5f["lesion"][:]
    lesion_mask = h5f["lesion_mask"][:]
    if lesion_arr.shape[0] < 1:
        # print(file, ":", lesion_arr.shape)
        os.remove(os.path.join(data_path, file))
    # if np.sum(label) < 10:
    #     os.remove(os.path.join(data_path, file))

# print(f"sum:{all_num}, zero:{count}, {count/all_num}")


