import SimpleITK as sitk
import os
from tqdm import tqdm
from medpy import metric
import numpy as np

def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray):
    pred = np.array(pred, dtype=np.int16)
    gt = np.array(gt, dtype=np.int16)
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        dice = 0.0
        hd95 = 999.0
        asd = 999.0
        iou = 0.0
    else:
        dice = metric.dc(pred, gt)
        hd95 = metric.hd95(pred, gt)
        asd = metric.asd(pred, gt)
        # jc = metric.jc(pred, gt)
        tp = np.sum(pred * gt)
        fn = np.sum((np.ones_like(pred) - pred) * gt)
        fp = np.sum(pred * (np.ones_like(gt) - gt))
        iou = tp / (tp + fn + fp)
    return [dice, iou, hd95, asd]

file_path = '/data1/code/model/LIDC/model_baseline_50/Entropy_Minimization_with_LesionMix2/predict'
file_list = os.listdir(file_path)
result = []
pre_files = [file for file in file_list if 'predict' in file]
gt_files = [file for file in file_list if 'lesion' in file]
pre_files.sort()
gt_files.sort()

if len(pre_files) != len(gt_files):
    print("len err!")
    exit()

for pre, gt in tqdm(zip(pre_files, gt_files)):
    pre_obj = sitk.ReadImage(os.path.join(file_path, pre))
    gt_obj = sitk.ReadImage(os.path.join(file_path, gt))

    pre_arr = sitk.GetArrayFromImage(pre_obj)
    gt_arr = sitk.GetArrayFromImage(gt_obj)

    if pre_arr.shape != gt_arr.shape:
        pre_shape = pre_arr.shape
        gt_shape = gt_arr.shape
        if gt_shape[0] > pre_shape[0]:
            pre_arr = np.pad(pre_arr, ((0, gt_shape[0] - pre_shape[0]), (0, gt_shape[1] - pre_shape[1]), (0, gt_shape[2] - pre_shape[2])), mode='constant', constant_values=0)
        else:
            gt_arr = np.pad(gt_arr, (
            (0, pre_shape[0] - gt_shape[0]), (0, gt_shape[1] - pre_shape[1]), (0, gt_shape[2] - pre_shape[2])),
                             mode='constant', constant_values=0)

    result.append(calculate_metric_percase(pre_arr, gt_arr))

dice_list = [item[0] for item in result]
iou_list = [item[1] for item in result]
hd95_list = [item[2] for item in result]
asd_list = [item[3] for item in result]

filenames = ['Dice.list', 'IoU.list', 'HD.list', 'ASD.list']
for i, filename in enumerate(filenames):
    filenames[i] = os.path.join(file_path, filename)

for i, list_i in enumerate(zip(dice_list, iou_list, hd95_list, asd_list)):
    for j, filename in enumerate(filenames):
        with open(filename, 'a') as file:
            file.write(str(list_i[j]) + '\n')
