import os

datapath = '//media/Deepin/003/chen_data/semi-supervision data/shanxi-cui/raw'
image_path = datapath + "/imagesTs"
label_path = datapath + "/labelsTs"
label_list = os.listdir(label_path)
data_list = os.listdir(image_path)
for data in data_list:
    if " " in data:
        os.rename(os.path.join(image_path, data), os.path.join(image_path, data.replace(" ", "")))
for label in label_list:
    if " " in label:
        os.rename(os.path.join(label_path, label), os.path.join(label_path, label.replace(" ", "")))
