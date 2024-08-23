import SimpleITK as sitk
import os
from tqdm import tqdm

image_root_path = "/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LUAD GD/My_method 3c/argument_TrainValTest/train"
label_root_path = "/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LUAD GD/My_method 3c/argument_TrainValTest"
file_list = os.listdir(image_root_path)
for file in tqdm(file_list):
    read_path = os.path.join(image_root_path, file)
    image = sitk.ReadImage(read_path)
    image_arr = sitk.GetArrayFromImage(image)
    label_read = os.path.join(label_root_path, file[:8] + ".nii.gz")
    label = sitk.ReadImage(label_read)
    label_arr = sitk.GetArrayFromImage(label)
    type(image_arr, label_arr)
    if image_arr.shape[0] < 100:
        print(file)
        os.remove(read_path)
        label_path = os.path.join(label_root_path, file[:8] + ".nii.gz")
        os.remove(label_path)