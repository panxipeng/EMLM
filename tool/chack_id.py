import os

datapath = '//media/Deepin/003/chen_data/semi-supervision data/shanxi-cui/raw'
image_path = datapath + "/imagesTs"
label_path = datapath + "/labelsTs"
label_list = os.listdir(label_path)
data_list = os.listdir(image_path)
for image in data_list:
    if image[:-12] + ".nii.gz" not in label_list:
        print(image[:-12] + ".nii.gz")