import os
import numpy as np
from tqdm import tqdm
import h5py

datapath = "/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LUAD GD/My_method 3c/argument_TrainValTest/val"
file_list = os.listdir(datapath)
for file in tqdm(file_list):
    read = os.path.join(datapath, file)
    h5f = h5py.File(read, "r")
    lesion = h5f["image"][:]
    lesion_mask = h5f["label"][:]
    if type(lesion) != np.ndarray or type(lesion_mask) != np.ndarray:
        print(file)
        print("class error")
    if lesion.dtype != np.float64 or lesion_mask.dtype != np.uint8:
        print(file)
        # print(lesion.dtype)
        # print(lesion_mask.dtype)
        print("data type error")
    if lesion.shape[0] != 320 or lesion_mask.shape[0] != 320:
        # print(lesion.shape[0])
        # print(lesion_mask.shape[0])
        print(file)
        print("shape error")

    # print(type(lesion)=="<class 'numpy.ndarray'>")
    # "<class 'numpy.ndarray'>"