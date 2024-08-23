import os
import random

import numpy as np
from preprocess import Preprocesser
from tqdm import tqdm
import h5py
datapath = "/home/cmw/data/LNDb/LesionMix_km_DBI2/argument_TrainValTest/train"

file_list = os.listdir(datapath)
count_pixel = 0
count_slice = 0
all_pixel = 0
for file in tqdm(file_list):
    read = os.path.join(datapath, file)
    # read = "/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/My_data_enhancement/lesion_warehouse/class3/img10779829_lesion1.h5"
    h5f = h5py.File(read, "r")
    # print(h5f.keys())'/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/My_data_enhancement/lesion_warehouse/class3'
    image = h5f["image"][:]
    label = h5f["label"][:]
    if np.max(image) != 1:
        print(file)
    # label[label > 0] = 1
    # all_pixel += image.shape[0] ** 2
    # count_pixel += np.sum(label)
    # if np.sum(label) > 0:
    #     count_slice += 1
# slice_propotion = count_slice/len(file_list)
# pixel_propotion = count_pixel/all_pixel
# print("pixel_propotion: {}/n slice_propotion: {}".format(pixel_propotion, slice_propotion))
    # if np.sum(label) == 0 and random.random() < 0.9:
    #     os.remove(read)
    # print(np.sum(label))
    # pre = Preprocesser()
    # pre.Transform_array_to_png(image, label, idx, '/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LNDb/LesionMix3/check_image')
        # if np.sum(lesion_mask) < 200:
        #     print()
        #     print(lesion_mask.shape)
        # if np.max(lesion_mask) > 3:
        #     print(os.path.join(folder_path, file))
        # lesion = h5f["lesion"][:]
        # lesion_mask = h5f["lesion_mask"][:]
        # classes = h5f["class"][0]
        # if classes != 3:
        #     print("error")
        # source = h5f["source"][0].decode()
        # image_id
        # image_id = h5f["image_id"][0].decode()
        # lesion = h5f["lesion"][:]
        # print(classes)
        # print(classes)
        # a = image.shape
        # if image.shape[-2:] != label.shape[-2:]:
        #     print(file)

# h5f = h5py.File('/media/Deepin/003/chen_data/semi-supervision data/LIDC/new_data_2d4c/data/slices/0001a_ct_slice_86.h5', "r")
# image = h5f["image"][:]
# label = h5f["label"][:]
# print()