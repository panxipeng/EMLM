import numpy as np
from tqdm import tqdm
import h5py
import os


def list_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = data[i] + '\n'
        file.write(s)
    file.close()
    print("保存成功")

datapath = '/home/cmw/data/LNDb/LesionMix_km_DBI2/argument_TrainValTest/train'
datalist = os.listdir(datapath)
# train_list = []
for data in tqdm(datalist):
    read = os.path.join(datapath, data)
    h5f = h5py.File(read, "r")
    image = h5f['image'][:]
    label = h5f['label'][:]
    lung = h5f['lung'][:]
    if label.shape[0] != 320:
        print(data)
    # print(image.shape)
    # print(data)
    # train_list.append(name)

# list_save("/media/Deepin/003/chen_data/semi-supervision data/LIDC/data20230213_3d4c_3296/val2.list", train_list)