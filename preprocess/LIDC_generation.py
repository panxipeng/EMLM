import os
import SimpleITK as sitk
from tqdm import tqdm
import shutil
import numpy as np

data_path = "/media/Deepin/003/chen_data/LIDC/new_nrrd"
save_path = "/media/Deepin/003/chen_data/semi-supervision data/LIDC/raw_new"

def check_file_list(file_list):
    nrrd_flag = False
    gz_flag = False
    for file in file_list:
        if file.endswith("nrrd"):
            nrrd_flag = True
        if file.endswith("nii.gz"):
            gz_flag = True
    return nrrd_flag and gz_flag

def check_obs_num(file_list):
    obs_list = set()
    for labelfile in file_list:
        if labelfile.endswith(".nii.gz"):
            obs_list.add(labelfile[6])
        else:
            continue
    if len(obs_list) == 1:
        return 1
    elif len(obs_list) >= 2:
        return 2

def get_new_file_name(file_list):
    for file in file_list:
        if file.endswith(".nrrd"):
            return file[:8]
    else:
        print("No nii.gz file!")
        return None

def get_length(data_path):
    files_list = []
    for root, dirs, files in os.walk(data_path):
        files_list.append(dirs)
    length_list = [len(length_) for length_ in files_list]
    return sum(length_list)


data_num = get_length(data_path)

with tqdm(total=data_num) as pbar:
    for root, dirs, files in os.walk(data_path):
        if len(files) > 0:
            pbar.update(1)
            if not check_file_list(files):
                print("{} lose nii.gz or nrrd!".format(os.path.basename(root)))
                continue
            effective_num = check_obs_num(files)
            if effective_num == 1:
                print("{} only has one obsver!".format(os.path.basename(root)))
                continue
            new_file_name = get_new_file_name(files)

            label_list = []
            for file in files:
                if file.endswith(".nrrd"):
                    image = sitk.ReadImage(os.path.join(root, file))
                if file.endswith("nii.gz"):
                    label = sitk.ReadImage(os.path.join(root, file))
                    label_list.append(sitk.GetArrayFromImage(label))
            final_label = np.zeros_like(label_list[0])
            for label_arr in label_list:
                final_label = final_label + label_arr

            final_label[final_label <= effective_num] = 0
            final_label[final_label > effective_num] = 1
            final_label = sitk.GetImageFromArray(final_label)

            final_label.SetOrigin(image.GetOrigin())
            final_label.SetDirection(image.GetDirection())
            final_label.SetSpacing(image.GetSpacing())
            sitk.WriteImage(image, os.path.join(save_path, "imageTr/{}_0000.nii.gz".format(new_file_name)))
            sitk.WriteImage(final_label, os.path.join(save_path, "labelTr/{}.nii.gz".format(new_file_name)))
            print("{} complete!".format(new_file_name))


