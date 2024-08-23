import argparse
from dataloaders.dataset import BaseDataSets
from dataloaders.dataset import *
from preprocess import Preprocesser
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


# file:///media/Deepin/003/chen_data/semi-supervision data/LIDC/data20230213_3d4c_3296
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data1/data/GD/only_Lesion', help='Name of Experiment')
parser.add_argument('--lesion_path', type=str,
                    default='/home/cmw/data/LUAD GD_manual/LesionMix_km_DBI/lesion_warehouse', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='My_data_enhancement/Fully_Supervised', help='experiment_name')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

num_classes = 2
args = parser.parse_args()
# train_transform = Lesion_and_image_augmentation([0.333, 0.333, 0.334], 3)
# init_augmentation_proportion = [round(1 / (num_classes - 1), 4) for i in range(num_classes - 2)]
# init_augmentation_proportion.append(round(1 - sum(init_augmentation_proportion), 4))
# init_augmentation_proportion = [0.3, 0.3, 0.4]
transform = Lesion_and_image_augmentation('/data1/data/GD/LesionMix_km/lesion_warehouse',
                                                    labeled_prop=1.0, init_lesion_count=3,
                                                    argrate=1.0, racalss_flag=False, enhencement_strength="Weak")
transform2 = None_augmentation(512)
# db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=train_transform)
db = BaseDataSets(base_dir=args.root_path, split="train", transform=transform)
db2 = BaseDataSets(base_dir=args.root_path, split="train", transform=transform2)
for idx, data in tqdm(enumerate(db)):
    Preprocesser.Transform_tensor_to_png(data["image"], data["label"], str(idx), "/data1/data/GD/check_image")
for idx, data in tqdm(enumerate(db2)):
    Preprocesser.Transform_tensor_to_png2(data["image"], data["label"], str(idx), "/data1/data/GD/check_image")
# /media/Deepin/My Passport/dataset/Lung Data/Processed Data/LUAD GD/My_me thod 5class/argument_TrainValTest