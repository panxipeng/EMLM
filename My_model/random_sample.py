import os
import random
import shutil
from tqdm import tqdm

def sample_files(source_folder, target_folder, sample_ratio):
    file_count = len([f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))])
    sample_count = int(file_count * sample_ratio)
    random.seed(1024)
    sample_files = random.sample(os.listdir(source_folder), sample_count)

    for file in tqdm(sample_files):
        source_file = os.path.join(source_folder, file)
        target_file = os.path.join(target_folder, file)
        shutil.copy2(source_file, target_file)



if __name__ == '__main__':
    source_folder = '/data1/data/LIDC/processed/argument_TrainValTest/train'
    target_folder = '/data1/data/LIDC_ft/argument_TrainValTest/train'
    sample_ratio = 0.1

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    sample_files(source_folder, target_folder, sample_ratio)
