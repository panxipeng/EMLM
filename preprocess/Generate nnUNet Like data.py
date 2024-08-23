import os
import shutil
from tqdm import tqdm

root_path = "/media/Deepin/My Passport/dataset/Lung Data/LUAD_radiomics/GD_manual_consistency"
target_path = "/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LUAD GD_manual/raw"
image_target_root = target_path + "/imagesTr"
label_target_root = target_path + "/labelsTr"
if not os.path.exists(image_target_root):
    os.makedirs(image_target_root)
    os.makedirs(label_target_root)
pbar = tqdm(total=len(os.listdir(root_path)))
for root, dir, files in os.walk(root_path):
    if len(files)>4:
        folder_name = os.path.basename(root)
        image_path = os.path.join(root, folder_name + ".nii.gz")
        label_path = os.path.join(root, folder_name + "_seg.nii.gz")

        while len(folder_name) < 8:
            folder_name = "0" + folder_name

        image_target_path = os.path.join(image_target_root, folder_name + "_0000.nii.gz")
        label_target_path = os.path.join(label_target_root, folder_name + ".nii.gz")
        shutil.copy(image_path, image_target_path)
        shutil.copy(label_path, label_target_path)
        pbar.update(1)
