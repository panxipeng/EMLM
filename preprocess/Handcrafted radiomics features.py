import numpy as np
import six
import collections
import scipy
import multiprocessing as mul
import threading
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
# import text
import os, sys
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from radiomics import featureextractor


# van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. `https://doi.org/10.1158/0008-5472.CAN-17-0339 <https://doi.org/10.1158/0008-5472.CAN-17-0339>`_


def load_correct_roi_Array(path):
    file_list = os.listdir(path)
    for file in file_list:
        if len(file) >= 20:
            result_path = os.path.join(path, file)
            seg = sitk.ReadImage(result_path)
            return seg


# read images in Nifti format
def loadImgArraywithID(fold, iden):
    imgPath = os.path.join(fold, iden + ".nrrd")
    img = sitk.ReadImage(imgPath)
    return img


def loadSegArraywithID(fold, iden):
    # segPath = os.path.join(fold,iden + "_roi.nii.gz")
    segPath = os.path.join(fold, "nodule.nii.gz")
    seg = sitk.ReadImage(segPath)
    seg_array = sitk.GetArrayFromImage(seg)
    return seg


def get_only_maximum_cross_sectional_area_roi(roi_array):
    depth, width, length = np.shape(roi_array)
    maximum_area = 0
    maximum_area_depth = 0
    zero_array = np.zeros((width, length))
    for d in range(depth):
        singal_roi = roi_array[d, :, :]
        if not (singal_roi == zero_array).all():
            # singal_roi = Image.fromarray(singal_roi)
            # singal_roi = cv2.cvtColor(np.unit8(singal_roi), cv2.COLOR_RGB2GRAY)  # 灰度化
            contours, hierarchy = cv2.findContours(singal_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            area = cv2.contourArea(contours[0])
            if area > maximum_area:
                maximum_area = area
                maximum_area_depth = d
    new_roi = np.zeros((depth, width, length))
    new_roi[maximum_area_depth, :, :] = roi_array[maximum_area_depth, :, :]
    return new_roi


if __name__ == '__main__':
    imgDirs = ['/media/zbc/My Passport/dataset/PSELA-3classes/train/iccroi/resampled_ob2']
    for ind, imgDir in enumerate(imgDirs):
        dirlist = os.listdir(imgDir)[:]
        # Feature Extraction
        featureDict = {}
        for ind in tqdm(dirlist):
            # if ind != '10052377':
            #     continue
            #
            path = os.path.join(imgDir, ind)

            path = os.path.join(imgDir, str(ind))

            img = loadImgArraywithID(path, ind)
            # img = sitk.GetArrayFromImage(img)
            # img = sitk.GetImageFromArray(img)
            file_list = os.listdir(path)

            # mask.
            params = './Paramsescc.yaml'

            extractor = featureextractor.RadiomicsFeatureExtractor(params)
            dictkey = []
            feature = []
            file_list.sort()
            for roi in file_list:
                if "nodule" not in roi:
                    continue
                if "20mm" in roi:
                    continue
                mask = sitk.ReadImage(os.path.join(path, roi))
                mask = sitk.GetArrayFromImage(mask)
                # mask = sitk.GetImageFromArray(mask)
                # _, mask = cv2.threshold(mask, 0.1, 1, cv2.THRESH_BINARY)
                mask = sitk.GetImageFromArray(mask)
                result = extractor.execute(img, mask)
                key = list(result.keys())
                if roi != 'nodule.nii.gz':
                    key = [roi[:-13] + item for item in key]
                    key = key[32:]

                    for jind in range(len(key)):
                        # sss = key[jind][key[jind].find('mm_') + 3:]
                        feature.append(result[key[jind][key[jind].find('mm_') + 3:]])
                    dictkey.extend(key)
                else:
                    key = key[32:]
                    for jind in range(len(key)):
                        feature.append(result[key[jind]])
                    dictkey.extend(key)

            featureDict[ind] = feature

            print(ind)

        dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns=dictkey)
        if 'Grade1' in imgDir[-6:]:
            dataframe.to_csv(
                '/media/zbc/My Passport/dataset/PSELA-3classes/train/Features_Radiomics_Grade1.csv')
        elif 'Grade2' in imgDir[-6:]:
            dataframe.to_csv(
                '/media/zbc/My Passport/dataset/PSELA-3classes/train/Features_Radiomics_Grade2.csv')
        elif 'Grade3' in imgDir[-6:]:
            dataframe.to_csv(
                '/media/zbc/My Passport/dataset/PSELA-3classes/train/Features_Radiomics_Grade3.csv')
        else:
            dataframe.to_csv(
                '/media/zbc/My Passport/dataset/PSELA-3classes/train/iccroi/Features_Radiomics_ob2.csv')
        # dataframe.to_csv('/media/zbc/My Passport/dataset/PSELA-3classes/train/iccroi/Features_Radiomics_test.csv')
