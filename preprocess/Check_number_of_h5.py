import h5py
import os
from tqdm import tqdm
import numpy as np
import random
import shutil

root_path = "/data1/data/LIDC/pgp_test"
target_path = "/data1/data/LIDC/label_robutness_test_1_3"
unlabel_rat = 3.0
random.seed(512)
shutil.copytree(root_path, target_path)

root_train_path = root_path + "/argument_TrainValTest/train"
target_train_path = target_path + "/argument_TrainValTest/train"
shutil.rmtree(target_train_path)
os.mkdir(target_train_path)

data_list = os.listdir(root_train_path)
labeled_list = []
unlabeled_list = []

if os.path.exists('./labeled_list.txt'):
    with open('./labeled_list.txt', 'r') as f:
        for line in f:
            labeled_list.append(line.strip())

    with open('./unlabeled_list.txt', 'r') as f:
        for line in f:
            unlabeled_list.append(line.strip())
else:
    for data in tqdm(data_list):
        file_path = os.path.join(root_train_path, data)
        with h5py.File(file_path, "r") as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]
            lung = h5f['lung'][:]
            if np.sum(label) != 0:
                labeled_list.append(data)
            else:
                unlabeled_list.append(data)
    with open('./labeled_list.txt', 'w') as f:
        for item in labeled_list:
            f.write("%s\n" % item)

    with open('./unlabeled_list.txt', 'w') as f:
        for item in unlabeled_list:
            f.write("%s\n" % item)

sample_count = int(len(labeled_list) * unlabel_rat)
sample_unlabeled_files = random.sample(unlabeled_list, sample_count)

all_files = sample_unlabeled_files + labeled_list

for item in tqdm(all_files):
    source_path = os.path.join(root_train_path, item)
    target_path = os.path.join(target_train_path, item)
    shutil.copy(source_path, target_path)


# print("labeled slice num:{}, unlabeled slice num:{} \nrat: {}".format(len(labeled_list), len(unlabeled_list),
#                                                                       float(len(labeled_list) / len(unlabeled_list))))
