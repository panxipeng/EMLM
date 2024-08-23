import SimpleITK as sitk
import numpy as np
import os

d_path = "/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/My_data_enhancement/train_image/labeled_data"
d_list = os.listdir(d_path)
d_list = [item for item in d_list if "lesion" in item]
for d in d_list:
    read_path = os.path.join(d_path, d)
    label = sitk.ReadImage(read_path)
    label_arr = sitk.GetArrayFromImage(label)
    print(np.max(label_arr))
