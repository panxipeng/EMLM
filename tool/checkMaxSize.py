import os
import h5py
from tqdm import tqdm
data_path = "/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LNDb/LesionMix_3/argument_TrainValTest/train"
maxSize = 0
datalist = os.listdir(data_path)

for data in tqdm(datalist):
    if ".h5" not in data:
        continue
    h5f = h5py.File(os.path.join(data_path, data), "r")
    image = h5f['image'][:]
    if image.shape[0] > maxSize:
        maxSize = image.shape[0]
    if image.shape[1] > maxSize:
        maxSize = image.shape[1]

print(maxSize)