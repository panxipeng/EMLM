import os
import random
import shutil
from tqdm import tqdm
image_path = "/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/raw/imagesTr"
label_path = "/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/raw/labelsTr"
image_target = "/media/Deepin/Extreme Pro/for feng/image"
label_target = "/media/Deepin/Extreme Pro/for feng/label"
image_list = os.listdir(image_path)
selected_case_list = random.sample(image_list, 60)

for item in tqdm(selected_case_list):
    shutil.copy(os.path.join(image_path, item), os.path.join(image_target, item))
    shutil.copy(os.path.join(label_path, item[:8] + ".nii.gz"), os.path.join(label_target, item[:8] + ".nii.gz"))