import shutil
from tqdm import tqdm
import os

# def save_image(filename, file_path, target_path):
#     pass

root_path = "/media/cmw/My Passport/xiaorongshiyan_sucai/raw"
target_path = "/media/cmw/My Passport/xiaorongshiyan_sucai/processed"

# len_all = 0
# for root, dir, files in os.walk(root_path):
#     if len(files)>10:
for root, dir, files in os.walk(root_path):
    if len(files)>10:
        for nii_file in files:
            if "image" in nii_file or "lesion" in nii_file:
                folder_path = target_path + "/" + nii_file[:8]
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                if not os.path.exists(os.path.join(folder_path, nii_file)):
                    shutil.copy(os.path.join(root, nii_file), os.path.join(folder_path, nii_file))
            else:
                folder_path = target_path + "/" + nii_file[:8]
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                newFileName = nii_file.replace("predict", os.path.basename(root))
                shutil.copy(os.path.join(root, nii_file), os.path.join(folder_path, newFileName))