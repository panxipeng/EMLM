import SimpleITK as sitk
import os
from tqdm import tqdm
import numpy as np
import shutil

path = "/media/Deepin/003/chen_data/LIDC/new_nrrd"
image_path = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/raw/imagesTr"
label_path = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/raw/labelsTr"
folder_list = os.listdir(path)
folder_list.sort()
for folder in tqdm(folder_list):
    folder_path = os.path.join(path, folder)
    file_list = os.listdir(folder_path)
    label = np.zeros((5, 5))
    file_name = ""
    for file1 in file_list:
        if file1.endswith(".nrrd"):
            file_name = file1[0:8]
            break

    for file in file_list:
        file_path = os.path.join(folder_path, file)
        if file_path.endswith(".nrrd"):
            image_object = sitk.ReadImage(file_path)
            sitk.WriteImage(image_object, os.path.join(image_path, file_name + "_0000.nii.gz"))
        else:
            label_object = sitk.ReadImage(file_path)
            label_arr = sitk.GetArrayFromImage(label_object)
            if label.shape != label_arr.shape:
                label = label_arr
            else:
                label = label + label_arr
    label[label > 1] = 1
    label = sitk.GetImageFromArray(label)
    sitk.WriteImage(label, os.path.join(label_path, file_name + ".nii.gz"))
