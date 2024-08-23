import SimpleITK as sitk
import os
import numpy as np
import shutil
from tqdm import tqdm

label_path = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/raw/labelsTr"
image_path = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/raw/imagesTr"
label_list = os.listdir(label_path)
image_list = os.listdir(image_path)

target_label_path = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/raw_new2/labelsTr"
target_image_path = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/raw_new2/imagesTr"
if not os.path.exists(target_label_path):
    os.makedirs(target_label_path)
    os.makedirs(target_image_path)
for label in tqdm(label_list):
    read_path = os.path.join(label_path, label)
    label_obj = sitk.ReadImage(read_path)
    label_arr = sitk.GetArrayFromImage(label_obj)
    if np.sum(label_arr) > 100:
        shutil.copy(read_path, os.path.join(target_label_path, label))
        shutil.copy(os.path.join(image_path, label[:8] + "_0000.nii.gz"), os.path.join(target_image_path, label[:8] + "_0000.nii.gz"))