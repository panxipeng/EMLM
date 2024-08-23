import os

val_path = "/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LUAD GD/My_method 5class/argument_TrainValTest/val"
train_path = "/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LUAD GD/My_method 5class/argument_TrainValTest/train"
val_list = os.listdir(val_path)
train_list = os.listdir(train_path)

val_list = [item[:8] for item in val_list]
val_list = set(val_list)
train_list = [item[:8] for item in train_list]
train_list = set(train_list)

for data in val_list:
    if data in train_list:
        print(data)