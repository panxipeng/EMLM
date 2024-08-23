import os
import os.path as path
import random
import matplotlib.pylab as plt
import sklearn.metrics
from scipy import ndimage
import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp
import joblib
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import pandas as pd
import h5py
import copy
import shutil
from lungmask import mask
from PIL import Image
from radiomics import featureextractor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(1024)


def SplitOneList2N(origin_list, n):
    '''
    将列表平均分成n段
    '''
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1

    for i in range(0, n):
        yield origin_list[i * cnt:(i + 1) * cnt]


def PrintInfo(info):
    def Wrapper(func):
        def infunc(*args, **kwargs):
            print("*" * (len(info) * 3 + 2))
            print("*" * len(info), info, "*" * len(info))
            print("*" * (len(info) * 3 + 2))
            result = func(*args, **kwargs)
            return result
            # print("*" * (len(info + " Finish") * 3 + 2))
            # print("*" * len(info + " Finish"), info + " Finish", "*" * len(info + " Finish"))
            # print("*" * (len(info + " Finish") * 3 + 2))

        return infunc

    return Wrapper


'''
MutiProcessing
该装饰器用于将一个通过循环处理列表类数据的函数方法进行多进程加速处理
参数需要通过指定关键词的形式传入`
被装饰函数不要有返回值
'''


def MutiProcessing(kernel_num):
    def decorator(func):
        def infunc(*args, **kwargs):
            list_keys = []
            for item in list(kwargs.keys()):
                if isinstance(kwargs[item], list):
                    list_keys.append(item)

            len_list = [len(kwargs[item]) for item in list_keys]
            assert len_list[0] == np.mean(len_list)  # ensure the length of all lists are equal

            all_list_ori = [kwargs[listName] for listName in list_keys]
            for s_key in list_keys:
                del kwargs[s_key]
            multi_segment_list = [[single_test for single_test in SplitOneList2N(data_list, kernel_num)] for data_list
                                  in all_list_ori]
            new_list = multi_segment_list[0]
            # new_list = [item for item in multi_segment_list]
            # new_list = [item for item in multi_segment_list]
            # new_list = multi_segment_list
            # check_dict1 = dict(kwargs, **dict.fromkeys(list_keys, new_list[0]))
            # check_dict2 = dict(kwargs, **dict.fromkeys(list_keys, new_list[1]))
            # check_dict3 = dict(kwargs, **dict.fromkeys(list_keys, new_list[2]))
            # check_dict4 = dict(kwargs, **dict.fromkeys(list_keys, new_list[3]))
            # check_dict5 = dict(kwargs, **dict.fromkeys(list_keys, new_list[4]))

            processes = [
                mp.Process(target=func, args=(*args,), kwargs=dict(kwargs, **dict.fromkeys(list_keys, new_list[i]))) for i
                in range(0, kernel_num)]
            for t in processes:
                t.start()  # 开始线程或进程，必须调用
            for t in processes:
                t.join()  # 等待直到该线程或进程结束

        return infunc

    return decorator


class Preprocesser():
    def __init__(self,
                 input_dir="/data1/data/LIDC/raw",
                 output_dir="/data1/data/LIDC/pgp_test",
                 norm_range=(-1000, 200)):
        # path or dir
        self.output_dir = output_dir
        self.argument_output_dir = output_dir + "/argument_TrainValTest"
        self.normal_output_dir = output_dir + "/normal_TrainValTest"
        self.train_dir = self.argument_output_dir + "/train"
        self.train_dir_norm = self.normal_output_dir + "/train"
        self.val_dir = self.argument_output_dir + "/val"
        self.val_dir_norm = self.normal_output_dir + "/val"
        # self.image_save_path = output_dir + "/images"
        # self.lesion_dir = self.train_dir + "/lesion"
        self.labeled_dir = output_dir + "/train_image/labeled_data"
        self.unlabeled_dir = output_dir + "/train_image/unlabeled_data"
        self.val_image_dir = output_dir + "/val_image"
        self.test_image_dir = output_dir + "/test_image"
        self.lesion_warehouse_init = output_dir + "/lesion_warehouse/init"
        self.lesion_warehouse = output_dir + "/lesion_warehouse"
        self.radiomics_classfication_data = output_dir + "/radiomics_classfication_data"
        self.check_dir = output_dir + "/check"

        self.images_dir = os.path.join(input_dir, "imagesTr")
        self.labels_dir = os.path.join(input_dir, "labelsTr")
        self.error_dir = os.path.join(input_dir, "errors")

        # proportion
        self.labeled_proportion = 0.2
        self.unlabeled_proportion = 1 - self.labeled_proportion
        self.train_proportion = 0.6
        self.val_proportion = 0.2
        self.test_proportion = 0.2
        assert self.train_proportion + self.val_proportion + self.test_proportion == 1
        self.norm_range = norm_range
        self.resample_spacing = [0.6, 0.6, 1.0]

        # list
        self.images_list = os.listdir(self.images_dir)
        self.images_list.sort()
        self.train_image_list = random.sample(self.images_list, int(self.train_proportion * len(self.images_list)))
        # self.labeled_image_list = random.sample(self.train_image_list, 50)
        self.labeled_image_list = self.train_image_list
        self.labeled_image_list.sort()
        # self.unlabeled_image_list = [item for item in self.train_image_list if
        #                              item not in self.labeled_image_list]
        self.unlabeled_image_list = []
        self.unlabeled_image_list.sort()
        self.val_image_list = [item for item in self.images_list if item not in self.train_image_list]
        self.val_image_list = random.sample(self.val_image_list, int(0.5 * len(
            self.val_image_list)))  # default the number of val equel to test
        self.test_image_list = [item for item in self.images_list if
                                item not in self.train_image_list and item not in self.val_image_list]

    @PrintInfo("Init Path")
    def init_path(self):
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)
        if not os.path.exists(self.train_dir_norm):
            os.makedirs(self.train_dir_norm)
        if not os.path.exists(self.val_dir_norm):
            os.makedirs(self.val_dir_norm)
        if not os.path.exists(self.labeled_dir):
            os.makedirs(self.labeled_dir)
        if not os.path.exists(self.unlabeled_dir):
            os.makedirs(self.unlabeled_dir)
        if not os.path.exists(self.val_image_dir):
            os.makedirs(self.val_image_dir)
        if not os.path.exists(self.test_image_dir):
            os.makedirs(self.test_image_dir)
        if not os.path.exists(self.lesion_warehouse):
            os.makedirs(self.lesion_warehouse)
        if not os.path.exists(self.normal_output_dir):
            os.makedirs(self.normal_output_dir)
        if not os.path.exists(self.radiomics_classfication_data):
            os.makedirs(self.radiomics_classfication_data)
        if not os.path.exists(self.lesion_warehouse_init):
            os.makedirs(self.lesion_warehouse_init)

        if len(os.listdir(self.labeled_dir)) // 3 != len(self.labeled_image_list):
            print("start Process labeled Data")
            self.ProcessRawData(datalist=self.labeled_image_list, datadir=self.labeled_dir)

        if len(os.listdir(self.unlabeled_dir)) // 3 != len(self.unlabeled_image_list):
            print("start Process unlabeled Data")
            self.ProcessRawData(datalist=self.unlabeled_image_list, datadir=self.unlabeled_dir)

        if len(os.listdir(self.val_image_dir)) // 3 != len(self.val_image_list):
            print("start Process val Data")
            self.ProcessRawData(datalist=self.val_image_list, datadir=self.val_image_dir)

        if len(os.listdir(self.test_image_dir)) // 2 != len(self.test_image_list):
            print("start copy test Data")
            for item in tqdm(self.test_image_list):
                shutil.copy(os.path.join(self.images_dir, item),
                            os.path.join(self.test_image_dir, item[:-12] + "_image.nii.gz"))
                shutil.copy(os.path.join(self.labels_dir, item[:-12] + ".nii.gz"),
                            os.path.join(self.test_image_dir, item[:-12] + "_lesion.nii.gz"))

    @MutiProcessing(3)
    def ProcessRawData(self, datalist, datadir):
        for item in tqdm(datalist):
            image = sitk.ReadImage(os.path.join(self.images_dir, item), sitk.sitkInt16)
            label = sitk.ReadImage(os.path.join(self.labels_dir, item[:-12] + ".nii.gz"), sitk.sitkUInt8)

            image, label, lung = self.Resample_and_Normalization(image, label)

            sitk.WriteImage(image, os.path.join(datadir, item[:-12] + "_image.nii.gz"))
            sitk.WriteImage(label, os.path.join(datadir, item[:-12] + "_lesion.nii.gz"))
            sitk.WriteImage(lung, os.path.join(datadir, item[:-12] + "_lung.nii.gz"))

    @staticmethod
    def Transform_array_to_png(image_arr: np.ndarray, label_arr: np.ndarray, pred_arr: np.ndarray, image_id, save_path):
        image_arr = (image_arr - image_arr.min())/(image_arr.max() - image_arr.min()) * 255
        image = Image.fromarray(np.array(image_arr, dtype=np.uint8))
        label_arr = label_arr * 255
        label = Image.fromarray(np.array(label_arr, dtype=np.uint8))
        pred_arr = pred_arr * 255
        pred = Image.fromarray(np.array(pred_arr, dtype=np.uint8))
        image.save(os.path.join(save_path, str(image_id) + "_image.png"))
        label.save(os.path.join(save_path, str(image_id) + "_label.png"))
        pred.save(os.path.join(save_path, str(image_id) + "_pred.png"))

    @staticmethod
    def Transform_tensor_to_png(image_tensor: torch.Tensor, label_tensor: torch.Tensor, image_id, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_arr = image_tensor.numpy()
        if len(image_arr.shape) == 3 and image_arr.shape[0] == 1:
            image_arr = image_arr.squeeze(0)
        label_arr = label_tensor.numpy()
        if len(label_arr.shape) == 3 and label_arr.shape[0] == 1:
            label_arr = label_arr.squeeze(0)
        # print(np.max(label_arr))
        image_arr = image_arr * 255
        image_arr[image_arr > 255] = 255
        image = Image.fromarray(np.array(image_arr, dtype=np.uint8))
        label_arr = label_arr * 255
        label = Image.fromarray(np.array(label_arr, dtype=np.uint8))
        image.save(os.path.join(save_path, image_id + "_image0.png"))
        label.save(os.path.join(save_path, image_id + "_label0.png"))

    @staticmethod
    def Transform_tensor_to_png2(image_tensor: torch.Tensor, label_tensor: torch.Tensor, image_id, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_arr = image_tensor.numpy()
        if len(image_arr.shape) == 3 and image_arr.shape[0] == 1:
            image_arr = image_arr.squeeze(0)
        label_arr = label_tensor.numpy()
        if len(label_arr.shape) == 3 and label_arr.shape[0] == 1:
            label_arr = label_arr.squeeze(0)
        # print(np.max(label_arr))
        image_arr = image_arr * 255
        image_arr[image_arr > 255] = 255
        image = Image.fromarray(np.array(image_arr, dtype=np.uint8))
        label_arr = label_arr * 255
        label = Image.fromarray(np.array(label_arr, dtype=np.uint8))
        image.save(os.path.join(save_path, image_id + "_image1.png"))
        label.save(os.path.join(save_path, image_id + "_label1.png"))

    def Save_image(self, image_array, image_name, save_path):
        image_array = image_array * 255
        X = Image.fromarray(np.uint8(image_array), "L")
        X.save(save_path + "/{}.jpeg".format(image_name))

    def Check_Filenum(self, image_list: list, label_list: list) -> bool:
        if len(image_list) != len(label_list):
            return False
        return True

    def Check_Size(self, image: np.ndarray, label: np.ndarray) -> bool:
        if image.size != label.size:
            return False
        return True

    def Image_Normalization(self, Image):
        image_arr = sitk.GetArrayFromImage(Image)
        image_arr[image_arr > self.norm_range[1]] = self.norm_range[1]
        image_arr[image_arr < self.norm_range[0]] = self.norm_range[0]

        image_arr = (image_arr - np.min(image_arr)) / (np.max(image_arr) - np.min(image_arr))
        new_Image = sitk.GetImageFromArray(image_arr)
        new_Image.SetSpacing(Image.GetSpacing())
        new_Image.SetDirection(Image.GetDirection())
        new_Image.SetOrigin(Image.GetOrigin())

        return new_Image

    def GetLungMask(self, image):
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        image = sitk.GetArrayFromImage(image)
        lung = mask.apply(image, batch_size=12)
        lung[lung == 2] = 1

        lung = sitk.GetImageFromArray(lung)
        lung.SetSpacing(spacing)
        lung.SetOrigin(origin)
        lung.SetDirection(direction)
        return lung

    def GetBoundary(self, volume):
        for i in range(volume.shape[0]):
            if np.max(volume[i]) != 0:
                volume_zmin = i
                break

        for i in range(volume.shape[0])[::-1]:
            if np.max(volume[i]) != 0:
                volume_zmax = i
                break

        for i in range(volume.shape[1]):
            if np.max(volume[:, i, :]) != 0:
                volume_xmin = i
                break

        for i in range(volume.shape[1])[::-1]:
            if np.max(volume[:, i, :]) != 0:
                volume_xmax = i
                break

        for i in range(volume.shape[2]):
            if np.max(volume[:, :, i]) != 0:
                volume_ymin = i
                break

        for i in range(volume.shape[2])[::-1]:
            if np.max(volume[:, :, i]) != 0:
                volume_ymax = i
                break

        return volume_zmin, volume_zmax, volume_xmin, volume_xmax, volume_ymin, volume_ymax

    def GetBoundarybyLung(self, lung):
        z_list = []
        for i in range(lung.shape[0]):
            if np.sum(lung[i, ...]) >= 1600:
                z_list.append(i)

        return z_list

    def GetBoundarybyLesion(self, label):
        z_list = []
        for i in range(label.shape[0]):
            if np.sum(label[i, ...]) > 0:
                z_list.append(i)

        return z_list

    def Get_Label(self, image_name):
        label_path = os.path.join(self.labels_dir, image_name[:-12] + ".nii.gz")
        return sitk.ReadImage(label_path)

    def Remove_Small_connected_domain(self, image):
        cca = sitk.ConnectedComponentImageFilter()
        cca.SetFullyConnected(True)
        output_ex = cca.Execute(image)

        # get "id num" of everyconnected_domain
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(output_ex)

        num_label = cca.GetObjectCount()  # num of connected_domain
        num_list = [i for i in range(1, num_label + 1)]
        area_list = []
        for l in range(1, num_label + 1):
            area_list.append(stats.GetNumberOfPixels(l))

        sorted_num_list = sorted(num_list, key=lambda x: area_list[num_list.index(x)])
        sorted_area_list = sorted(area_list)

        min_idx = len(sorted_area_list)
        for area in sorted_area_list:
            if area * (self.resample_spacing[0] * self.resample_spacing[1] * self.resample_spacing[2]) < 14:  # 直径为3mm的球体体积约为14
                continue
            else:
                min_idx = sorted_area_list.index(area)
                break

        new_num_list = sorted_num_list[min_idx:]

        output = sitk.GetArrayFromImage(output_ex)

        for one_label in num_list:
            if one_label in new_num_list:
                continue
            x, y, z, w, h, d = stats.GetBoundingBox(one_label)
            one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
            output[z: z + d, y: y + h, x: x + w] *= one_mask
        output = (output > 0).astype(np.uint8)
        output = sitk.GetImageFromArray(output)
        return output

    def SaveLesioninH5(self, image, label, name, source):
        image_arr = sitk.GetArrayFromImage(image)
        label_arr = sitk.GetArrayFromImage(label)

        ccif = sitk.ConnectedComponentImageFilter()
        ccif.SetFullyConnected(True)
        image_cc = ccif.Execute(label)

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(image_cc)

        num_list = [i for i in range(1, ccif.GetObjectCount() + 1)]

        for one_label in num_list:
            x, y, z, w, h, d = stats.GetBoundingBox(one_label)
            w = w if w > h else h
            h = w
            x = int(x + w/2)
            y = int(y + h/2)
            z = int(z + d/2)

            single_lesion = image_arr[z - d//2: z + d//2, y - h//2: y + h//2, x - w//2: x + w//2]
            single_lesion_mask = label_arr[z - d//2: z + d//2, y - h//2: y + h//2, x - w//2: x + w//2]

            if not os.path.exists(self.lesion_warehouse_init):
                os.makedirs(self.lesion_warehouse_init)
            # strdt = h5py.special_dtype(vlen=str)
            with h5py.File(self.lesion_warehouse_init + '/img{}_lesion{}.h5'.format(name, one_label), 'w') as h5f:
                h5f.create_dataset('lesion', data=single_lesion, compression="gzip")
                h5f.create_dataset('lesion_mask', data=single_lesion_mask, compression="gzip")
                h5f.create_dataset('image_id', data=[name])
                h5f.create_dataset('source', data=[source])
                h5f.create_dataset('class', data=[1])
                # h5f.create_dataset('lesion_cord', data=f"({z + d // 2}, {y + h // 2}, {x + w // 2})")
            # f.close()

    def SaveImageLabel_for_Radiomics(self, image, label, name):
        # image_arr = sitk.GetArrayFromImage(image)
        label_arr = sitk.GetArrayFromImage(label)

        ccif = sitk.ConnectedComponentImageFilter()
        ccif.SetFullyConnected(True)
        image_cc = ccif.Execute(label)

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(image_cc)

        num_list = [i for i in range(1, ccif.GetObjectCount() + 1)]
        for one_label in num_list:
            x, y, z, w, h, d = stats.GetBoundingBox(one_label)
            new_label_arr = np.zeros_like(label_arr)
            new_label_arr[z: z + d, y: y + h, x: x + w] = label_arr[z: z + d, y: y + h, x: x + w]
            new_label_arr[new_label_arr > 1] = 1
            new_label = sitk.GetImageFromArray(new_label_arr)
            sitk.WriteImage(image, self.radiomics_classfication_data + "/{}_{}.nii.gz".format(name, one_label))
            sitk.WriteImage(new_label, self.radiomics_classfication_data + "/{}_{}_seg.nii.gz".format(name, one_label))

    def CheckMask(self, mask):
        mask_arr = sitk.GetArrayFromImage(mask)
        return True if np.sum(mask_arr) > 50 else False

    def Resample_and_Normalization(self, image, label):
        lung = self.GetLungMask(image)
        image_resampled, label_resampled, lung_resampled = self.ResapmleImage2(image=image,
                                                                               label=label,
                                                                               lung=lung,
                                                                               targetSpacing=self.resample_spacing)
        image_norm = self.Image_Normalization(image_resampled)  # targetSpacing=(1.0, 1.0, 1.0)
        label_without_minilesion = self.Remove_Small_connected_domain(label_resampled)
        return image_norm, label_without_minilesion, lung_resampled

    @PrintInfo("Get Lesions For Radiomics")
    @MutiProcessing(10)
    def GetLesionForRadiomics(self, datalist):
        for item in tqdm(datalist):
            image = sitk.ReadImage(os.path.join(self.labeled_dir, item[:-12] + "_image.nii.gz"))
            label = sitk.ReadImage(os.path.join(self.labeled_dir, item[:-12] + "_lesion.nii.gz"))
            label = self.Remove_Small_connected_domain(label)
            self.SaveLesioninH5(image, label, item[:-12], "GroundTruth")
            self.SaveImageLabel_for_Radiomics(image, label, item[:-12])

    # @PrintInfo("Get Lesions From labeled Data")
    # def GetLesionFromLabeledData(self):
    #     '''
    #     soure should be one of "GroundTruth" and "psuedo-label"
    #     '''
    #     # self.GetLesionForRadiomics(self.labeled_image_list)
    #     task_num = os.cpu_count() // 2
    #     task_list = [single_test for single_test in split_list_n_list(self.labeled_image_list, task_num)]
    #     processes = [mp.Process(target=self.GetLesionForRadiomics, args=(task_list[i])) for i in range(task_num)]
    #     for t in processes:
    #         t.start()  # 开始线程或进程，必须调用
    #     for t in processes:
    #         t.join()  # 等待直到该线程或进程结束
    #     # source = "GroundTruth"
    #     # for item in tqdm(self.labeled_image_list):
    #     #     image = sitk.ReadImage(os.path.join(self.labeled_dir, item[:-12] + "_image.nii.gz"))
    #     #     label = sitk.ReadImage(os.path.join(self.labeled_dir, item[:-12] + "_lesion.nii.gz"))
    #     #     label = self.Remove_Small_connected_domain(label)
    #     #     self.SaveLesioninH5(image, label, item[:-12], source)
    #     #     self.SaveImageLabel_for_Radiomics(image, label, item[:-12])

    def Crop_size(self, x, image_arr, label_arr):
        if x > 512:
            croped_size = int((x - 512) / 2)
            gap_size = int((x - 512) / 2)
            image_arr = image_arr[:, croped_size:croped_size + 512, croped_size:croped_size + 512]
            label_arr = label_arr[:, croped_size:croped_size + 512, croped_size:croped_size + 512]
        else:
            gap_size = int((512 - x) / 2)
            zero_case = np.zeros((label_arr.shape[0], 512, 512))
            zero_case_img = copy.copy(zero_case)
            zero_case_label = copy.copy(zero_case)
            zero_case_img[:, gap_size:gap_size + x, gap_size:gap_size + x] = image_arr
            zero_case_label[:, gap_size:gap_size + x, gap_size:gap_size + x] = label_arr
            image_arr = zero_case_img
            label_arr = zero_case_label
            gap_size = -gap_size
        return image_arr, label_arr, gap_size

    def ResapmleImage2(self, image, label=None, lung=None, targetSpacing=(1., 1., 1.),
                       resamplemethod=sitk.sitkNearestNeighbor):
        targetsize = [0, 0, 0]

        # read ori info
        ori_size = image.GetSize()
        ori_spacing = image.GetSpacing()
        ori_origin = image.GetOrigin()
        ori_direction = image.GetDirection()

        targetsize[0] = round(ori_size[0] * ori_spacing[0] / targetSpacing[0])
        targetsize[1] = round(ori_size[1] * ori_spacing[1] / targetSpacing[1])
        targetsize[2] = round(ori_size[2] * ori_spacing[2] / targetSpacing[2])

        zoom_ = [ori_spacing[2] / targetSpacing[2], ori_spacing[1] / targetSpacing[1],
                 ori_spacing[0] / targetSpacing[0]]

        image_arr = sitk.GetArrayFromImage(image)
        new_image_arr = ndimage.zoom(image_arr, zoom_, order=0)
        newImage = sitk.GetImageFromArray(new_image_arr)
        newImage.SetSpacing(targetSpacing)
        newImage.SetOrigin(ori_origin)
        newImage.SetDirection(ori_direction)
        # newImage.SetSize(targetsize)
        if label is not None:
            label_arr = sitk.GetArrayFromImage(label)
            new_label_arr = ndimage.zoom(label_arr, zoom_, order=0)
            newLabel = sitk.GetImageFromArray(new_label_arr)
            newLabel.SetSpacing(targetSpacing)
            newLabel.SetOrigin(ori_origin)
            newLabel.SetDirection(ori_direction)
        else:
            newLabel = None
        if label is not None:
            lung_arr = sitk.GetArrayFromImage(lung)
            new_lung_arr = ndimage.zoom(lung_arr, zoom_, order=0)
            newLung = sitk.GetImageFromArray(new_lung_arr)
            newLung.SetSpacing(targetSpacing)
            newLung.SetOrigin(ori_origin)
            newLung.SetDirection(ori_direction)
        else:
            newLung = None

        return newImage, newLabel, newLung

    def SavePatch2d(self, image, label, lung, z, id, data_flag):
        if data_flag == 'train':
            save_path = self.train_dir
        elif data_flag == 'train_norm':
            save_path = self.train_dir_norm
        elif data_flag == 'val_norm':
            save_path = self.val_dir_norm
        else:
            save_path = self.val_dir
        f = h5py.File(save_path + '/{}_slice_{}.h5'.format(id, z), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.create_dataset('lung', data=lung, compression="gzip")
        f.close()

    def CalImageRadiomicsFeatures(self, id_tuple):
        image_id = id_tuple[0]
        label_id = id_tuple[1]
        print(id_tuple)
        # for image_id, label_id in zip(image_list, label_list)
        # lesion_name.append(item[:-7])
        image = sitk.ReadImage(os.path.join(self.radiomics_classfication_data, image_id))
        label = sitk.ReadImage(os.path.join(self.radiomics_classfication_data, label_id))
        spacing = image.GetSpacing()
        Direction = image.GetDirection()
        Origin = image.GetOrigin()
        image = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
        label = sitk.GetImageFromArray(sitk.GetArrayFromImage(label))

        label.SetSpacing(spacing)
        label.SetDirection(Direction)
        label.SetOrigin(Origin)

        image.SetSpacing(spacing)
        image.SetDirection(Direction)
        image.SetOrigin(Origin)


        # label.SetOrigin(image.GetOrigin())
        # label.SetDirection(image.GetDirection())
        # print(label_list[image_list.index(item)])
        params = './Paramsescc.yaml'

        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        dictkey = {}
        feature = []
        result = extractor.execute(image, label)
        key = list(result.keys())
        key = key[47:]
        for jind in range(len(key)):
            feature.append(result[key[jind]])
            dictkey[key[jind]] = result[key[jind]]
        return dictkey

    def CheckClassEleNum(self, labels):
        for i in range(len(np.unique(labels))):
            if np.sum(labels == i) < (len(self.labeled_image_list) // 20):
                return False
        return True

    def compute_sse(self, data, medoids_idx, sample_target):
        """
        计算聚类结果的 SSE 值
        :param data: 数据集
        :param medoids_idx: 簇的质心索引
        :param sample_target: 样本所属簇
        :return: SSE 值
        """
        sse = 0.0
        for i in range(len(medoids_idx)):
            # 获取该簇的所有数据点
            cluster_data = data[sample_target == i]
            # 计算该簇中所有数据点与该簇质心的距离的平方，并将这些平方值相加
            sse += np.sum(np.square(cluster_data - medoids_idx[i]))
        return sse

    # max_k = 51
    # sses = []
    # # KMediod 实现方法可以参考我上一篇文章
    # for k in range(1, max_k):
    #     new_medoids, classify_points, sample_target = KMediod(data, k).run()
    #     sses.append(compute_sse(data, new_medoids, sample_target))
    # best_k = np.argmin(np.diff(sses)) + 2


    def GetBestModel(self, x, model_save_path):
        # DBI_list = []
        model_list = []
        SSE_list = []
        for i in range(2, len(os.listdir(self.lesion_warehouse_init)) // 10):
            model = KMeans(n_clusters=i, random_state=618)
            model.fit(x)
            labels = model.predict(x)
            SSE_list.append(self.compute_sse(x, model.cluster_centers_, labels))
            # if not self.CheckClassEleNum(labels):
            #     continue
            # DBI_list.append(sklearn.metrics.davies_bouldin_score(x, labels))
            model_list.append(model)
        # SSE_list1 = [SSE_list[i] - SSE_list[i-1] for i in range(1, len(SSE_list))]
        # SSE_list2 = [SSE_list1[i] - SSE_list1[i - 1] for i in range(1, len(SSE_list1))]
        max_idx = np.argmin(np.diff(SSE_list)) + 1
        # min = 9999
        # min_idx = 99
        # for idx, i in enumerate(DBI_list):
        #     if i < min:
        #         min = i
        #         min_idx = idx
        joblib.dump(model_list[max_idx], model_save_path)
        return model_list[max_idx]


    @PrintInfo("Calculate Radiomics Features")
    def Cal_Radiomics_MP(self):
        data_list = os.listdir(self.radiomics_classfication_data)
        image_list = [item for item in data_list if "_seg" not in item]
        image_list.sort()
        label_list = [item for item in data_list if "_seg" in item]
        label_list.sort()
        lesion_name = [item[:-7] for item in image_list]
        args = [item for item in zip(image_list, label_list)]
        pool = mp.Pool(processes=3)
        featureDict = pool.map(self.CalImageRadiomicsFeatures, args)

        dataframe = pd.DataFrame(featureDict, index=lesion_name)
        dataframe.to_csv(os.path.join(self.output_dir, 'Labeled_Radiomics.csv'))
        return dataframe

    @PrintInfo("Calculate Radiomics Features")
    def Cal_Radiomics(self):
        data_list = os.listdir(self.radiomics_classfication_data)
        image_list = [item for item in data_list if "_seg" not in item]
        image_list.sort()
        label_list = [item for item in data_list if "_seg" in item]
        label_list.sort()
        featureDict = []
        lesion_name = []
        for item in tqdm(image_list):
            lesion_name.append(item[:-7])
            img = sitk.ReadImage(os.path.join(self.radiomics_classfication_data, item))
            roi = sitk.ReadImage(os.path.join(self.radiomics_classfication_data, label_list[image_list.index(item)]))
            roi.SetOrigin(img.GetOrigin())
            roi.SetDirection(img.GetDirection())
            # print(label_list[image_list.index(item)])
            params = './Paramsescc.yaml'

            extractor = featureextractor.RadiomicsFeatureExtractor(params)
            dictkey = {}
            feature = []
            result = extractor.execute(img, roi)
            key = list(result.keys())
            key = key[47:]
            for jind in range(len(key)):
                feature.append(result[key[jind]])
                dictkey[key[jind]] = result[key[jind]]
            # dictkey[key[jind]] = result[key[jind]]
            featureDict.append(dictkey)

        dataframe = pd.DataFrame(featureDict, index=lesion_name)
        dataframe.to_csv(os.path.join(self.output_dir, 'Labeled_Radiomics.csv'))
        return dataframe

    @PrintInfo("Update Lesion Calss")
    def Update_lesion_calss(self, radiomics_features, n_clusters):
        data_id = radiomics_features.index.tolist()
        data_X = np.array(radiomics_features)
        scaler = MinMaxScaler()
        scaler.fit(data_X)
        data_X = scaler.transform(data_X)

        # y_pred = KMeans(n_clusters=1, random_state=618).fit_predict(data_X)
        kmeans_model = os.path.join(self.output_dir, 'kmeans_model.pkl')
        # dbscan_model = os.path.join(self.output_dir, 'dbscan_model.pkl')
        if os.path.exists(kmeans_model):
            print("Loading KMeans-model...")
            estimator = joblib.load(kmeans_model)
            print("Load Successful!")
        else:
            estimator = self.GetBestModel(data_X, kmeans_model)

        # if os.path.exists(kmeans_model):
        #     print("Loading DBSCAN-model...")
        #     estimator = joblib.load(kmeans_model)
        #     print("Load Successful!")
        # else:
        #     estimator = KMeans(n_clusters=5, random_state=712)
        #     estimator.fit(data_X)
        #     joblib.dump(estimator, kmeans_model)

        y_pred = estimator.predict(data_X)
        # y_pred = estimator.labels_
        y_pred = y_pred + 1  # 病灶类别从1开始计数
        data_dict = []
        for idx, item in enumerate(data_id):
            # imageid, lesion_id = item.split('_')
            imageid, lesion_id = item[:8], item[9:]
            data_id[idx] = "img" + imageid + "_lesion" + lesion_id
            item_dict = dict((("lesion", data_id[idx]), ("class", y_pred[idx])))

            with h5py.File(self.lesion_warehouse_init + "/{}.h5".format(data_id[idx]), 'r+') as h5f:
                # orig_class = h5f["class"][0]
                h5f["class"][0] = y_pred[idx]
                lesion_mask = h5f["lesion_mask"][:]
                lesion_mask[lesion_mask > 0] = 1
                h5f["lesion_mask"][:] = lesion_mask * h5f["class"][0]
                item_dict["lesion_area"] = np.sum(lesion_mask)
                # h5f.create_dataset("lesion_mask", data=lesion_mask * h5f["class"][0])
            data_dict.append(item_dict)
        return data_dict  # data_dict: [{"lesion": str, "class": int, "lesion_area": int}, ...]

    @PrintInfo("Update Lesion Warehouse")
    def Update_lesion_warehouse(self, data_dict):
        for item in data_dict:
            target_path = os.path.join(self.lesion_warehouse, "class{}".format(item["class"]))
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            shutil.copy(
                os.path.join(self.lesion_warehouse_init, item["lesion"] + ".h5"),
                os.path.join(target_path, item["lesion"] + ".h5")
            )

    @PrintInfo("Update train data")
    def Update_traindata(self, data_dict):
        for item in tqdm(self.labeled_image_list):
            label = sitk.ReadImage(os.path.join(self.labeled_dir, item[:-12] + "_lesion.nii.gz"))
            lesion_list = [lesion for lesion in data_dict if lesion["lesion"][3:-8] == item[:-12]]
            new_label = self.updateLabel(label, lesion_list)
            sitk.WriteImage(new_label, os.path.join(self.labeled_dir, item[:-12] + "_lesion.nii.gz"))

    def UpdateOneLabel(self, args):
        label_path = args[0]
        lesion_list = args[1]
        label = sitk.ReadImage(label_path)
        new_label = self.updateLabel(label, lesion_list)
        sitk.WriteImage(new_label, label_path)

    @PrintInfo("Update train data")
    def Update_traindata_MP(self, data_dict):
        all_label = []
        all_lesion = []
        for item in self.labeled_image_list:
            all_label.append(os.path.join(self.labeled_dir, item[:-12] + "_lesion.nii.gz"))
            lesion_list = [lesion for lesion in data_dict if lesion["lesion"][3:-8] == item[:-12]]
            all_lesion.append(lesion_list)
        args = [item for item in zip(all_label, all_lesion)]
        pool = mp.Pool(processes=10)
        res = pool.map(self.UpdateOneLabel, args)

    def updateLabel(self, label, lesion_list):
        label_arr = sitk.GetArrayFromImage(label)

        ccif = sitk.ConnectedComponentImageFilter()
        ccif.SetFullyConnected(True)
        image_cc = ccif.Execute(label)

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(image_cc)

        num_list = [i for i in range(1, ccif.GetObjectCount() + 1)]
        for one_label in num_list:
            area = stats.GetNumberOfPixels(one_label)
            for one_lesion in lesion_list:
                if area == one_lesion['lesion_area']:
                    x, y, z, w, h, d = stats.GetBoundingBox(one_label)
                    single_lesion_mask = label_arr[z: z + d, y: y + h, x: x + w]
                    single_lesion_mask[single_lesion_mask > 0] = one_lesion["class"]
                    # label_arr[z: z + d, y: y + h, x: x + w] = single_lesion_mask
                    lesion_list.remove(one_lesion)
                    break
        label = sitk.GetImageFromArray(label_arr)
        return label

    def Process_and_Save(self, image, label, lung, image_name, data_flag):
        # image, label, lung = self.ResapmleImage1(image, label, lung, [1.0, 1.0, 2.5])
        image_arr = sitk.GetArrayFromImage(image)
        label_arr = sitk.GetArrayFromImage(label)
        label_arr = np.array(label_arr, dtype=np.uint8)
        lung_arr = sitk.GetArrayFromImage(lung)
        if "train" in data_flag:
            target_z = self.GetBoundarybyLung(lung_arr)  # GetBoundary2 针对肺实质获取patch范围
        else:
            target_z = self.GetBoundarybyLesion(label_arr)  # GetBoundary3 针对ROI获取patch范围

        for z in target_z:
            self.SavePatch2d(image_arr[z, ...], label_arr[z, ...], lung_arr[z, ...], z, image_name, data_flag)

    def Check_and_Clean_LesionWarehouse(self):
        lesion_list = os.listdir(self.lesion_warehouse)
        # self.PrintInfo("Clearn Pseudo Lesions")
        for lesion in tqdm(lesion_list):
            h5f = h5py.File(os.path.join(self.lesion_warehouse, lesion), 'r')
            source = h5f["source"][0].decode()
            if source != "GroundTruth":
                os.remove(os.path.join(self.lesion_warehouse, lesion))
        # self.PrintInfo("Finish")

    # 函数未完成
    def Update_lesion_warehouse_with_model(self, model):
        self.Check_and_Clean_LesionWarehouse()

        for item in tqdm(os.listdir(precesser.unlabeled_dir)):
            if "_image.nii.gz" not in item:
                continue
            image = sitk.ReadImage(path.join(precesser.unlabeled_dir, item))
            image_resampleed, _, _ = self.ResapmleImage1(image=image)
            image_arr = sitk.GetArrayFromImage(image_resampleed)
            pseudo_label = np.zeros_like(image_arr)
            for slice in image_arr.shape[0]:
                slice = image_arr[slice, ...]
                slice[slice < self.norm_range[0]] = slice < self.norm_range[0]

    # @PrintInfo("Lesion augmentation")
    # def Lesion_augmentation(self):
    #     lesion_class_list = os.listdir(self.lesion_warehouse)
    #     lesion_class_list.remove("init")
    #     for lesion_class in lesion_class_list:
    #         print("{} :".format(lesion_class))
    #         lesion_path = os.path.join(self.lesion_warehouse, lesion_class)
    #         lesion_list = os.listdir(lesion_path)
    #         for lesion in tqdm(lesion_list):
    #             with h5py.File(os.path.join(lesion_path, lesion), 'r') as h5f:
    #                 self.Generate_lesion_augmentation(h5f, lesion_path, lesion)
    #             os.remove(os.path.join(lesion_path, lesion))
    #
    # def Generate_lesion_augmentation(self, h5f_o, lesion_path, lesion_file_name):
    #     lesion_arr = h5f_o["lesion"][:]
    #     mask_arr = h5f_o["lesion_mask"][:]
    #     source = h5f_o["source"][0].decode()
    #     image_id = h5f_o["image_id"][0].decode()
    #     lesion_class = h5f_o["class"][0]
    #     # rot_augmentation =
    #     for rot_lesion, rot_label, rot_name in self.Lesion_rot(lesion_arr, mask_arr, lesion_file_name):
    #         for flip_lesion, flip_label, new_name in self.Lesion_flip(rot_lesion, rot_label, rot_name):
    #             with h5py.File(os.path.join(lesion_path, new_name), 'w') as h5f:
    #                 h5f.create_dataset('lesion', data=flip_lesion, compression="gzip")
    #                 h5f.create_dataset('lesion_mask', data=flip_label, compression="gzip")
    #                 h5f.create_dataset('image_id', data=[image_id])
    #                 h5f.create_dataset('source', data=[source])
    #                 h5f.create_dataset('class', data=[lesion_class])

    # def Lesion_rot(self, image, label, lesion_name):
    #     rot_range = [rot for rot in range(0, 360, 45)]
    #     for rotation_angle_x in rot_range:
    #         for rotation_angle_y in rot_range:
    #             for rotation_angle_z in rot_range:
    #                 new_image = ndimage.rotate(image, rotation_angle_x, (0, 1))
    #                 new_image = ndimage.rotate(new_image, rotation_angle_y, (0, 2))
    #                 new_image = ndimage.rotate(new_image, rotation_angle_z, (1, 2))
    #
    #                 new_label = ndimage.rotate(label, rotation_angle_x, (0, 1))
    #                 new_label = ndimage.rotate(new_label, rotation_angle_y, (0, 2))
    #                 new_label = ndimage.rotate(new_label, rotation_angle_z, (1, 2))
    #                 new_file_name = lesion_name[:-3] + "_r({}, {}, {}).h5".format(rotation_angle_z, rotation_angle_y,
    #                                                                               rotation_angle_x)
    #
    #                 yield new_image, new_label, new_file_name

    # def Lesion_flip(self, image, label, lesion_name):
    #     flip_range = [0, 1]
    #     for flip_axis_x in flip_range:
    #         for flip_axis_y in flip_range:
    #             for flip_axis_z in flip_range:
    #                 new_image = image
    #                 new_label = label
    #                 if flip_axis_x != 0:
    #                     new_image = np.flip(new_image, axis=1)
    #                     new_label = np.flip(new_label, axis=1)
    #                 if flip_axis_y != 0:
    #                     new_image = np.flip(new_image, axis=2)
    #                     new_label = np.flip(new_label, axis=2)
    #                 if flip_axis_z != 0:
    #                     new_image = np.flip(new_image, axis=0)
    #                     new_label = np.flip(new_label, axis=0)
    #
    #                 new_file_name = lesion_name[:-3] + "_f({}, {}, {}).h5".format(flip_axis_z, flip_axis_y, flip_axis_x)
    #
    #                 yield new_image, new_label, new_file_name

    @PrintInfo("Precessing Train data")
    def Precess_traindata(self):
        print("Generate labeled train data")
        for item in tqdm(os.listdir(self.labeled_dir)):
            if "_image.nii.gz" not in item:
                continue
            image = sitk.ReadImage(path.join(self.labeled_dir, item))
            label = sitk.ReadImage(path.join(self.labeled_dir, item[:-12] + "lesion.nii.gz"))
            lung = sitk.ReadImage(path.join(self.labeled_dir, item[:-12] + "lung.nii.gz"))
            self.Process_and_Save(image, label, lung, item[:8], "train")

    def ProcessOneTrainData(self, args):
        data_path = args[0]
        data_name = args[1]
        data_class = args[2]
        image = sitk.ReadImage(path.join(data_path, data_name))
        label = sitk.ReadImage(path.join(data_path, data_name[:-12] + "lesion.nii.gz"))
        lung = sitk.ReadImage(path.join(data_path, data_name[:-12] + "lung.nii.gz"))
        self.Process_and_Save(image, label, lung, data_name[:8], data_class)

    # @PrintInfo("Precessing train data")
    def Generate_h5_data_MP(self, data_path, data_calss):
        if data_calss == 'train':
            save_path = self.train_dir
        elif data_calss == 'train_norm':
            save_path = self.train_dir_norm
        elif data_calss == 'val_norm':
            save_path = self.val_dir_norm
        else:
            save_path = self.val_dir

        if len(os.listdir(save_path)) > 0:
            return
        # print("Generate labeled train data")
        image_list = os.listdir(data_path)
        image_list = [item for item in image_list if "_image.nii.gz" in item]
        # path_list = [data_path] * len(image_list)
        # class_list = [data_calss] * len(image_list)
        args = [(data_path, image_id, data_calss) for image_id in image_list]
        pool = mp.Pool(processes=3)
        pool.map(self.ProcessOneTrainData, args)

    @PrintInfo("Precessing Val Data")
    def Precess_valdata(self):
        # print("Generate val data")
        for item in tqdm(os.listdir(self.val_image_dir)):
            if "_image.nii.gz" not in item:
                continue
            image = sitk.ReadImage(path.join(self.val_image_dir, item))
            label = sitk.ReadImage(path.join(self.val_image_dir, item[:-12] + "lesion.nii.gz"))
            lung = sitk.ReadImage(path.join(self.val_image_dir, item[:-12] + "lung.nii.gz"))

            self.Process_and_Save(image, label, lung, item[:8], "val")

    @PrintInfo("Precessing Train data")
    def Precess_traindata_norm(self):
        print("Generate labeled train data")
        for item in tqdm(os.listdir(self.labeled_dir)):
            if "_image.nii.gz" not in item:
                continue
            image = sitk.ReadImage(path.join(self.labeled_dir, item))
            label = sitk.ReadImage(path.join(self.labeled_dir, item[:-12] + "lesion.nii.gz"))
            lung = sitk.ReadImage(path.join(self.labeled_dir, item[:-12] + "lung.nii.gz"))
            self.Process_and_Save(image, label, lung, item[:8], "train_norm")

    @PrintInfo("Precessing Val Data")
    def Precess_valdata_norm(self):
        # print("Generate val data")
        for item in tqdm(os.listdir(self.val_image_dir)):
            if "_image.nii.gz" not in item:
                continue
            image = sitk.ReadImage(path.join(self.val_image_dir, item))
            label = sitk.ReadImage(path.join(self.val_image_dir, item[:-12] + "lesion.nii.gz"))
            lung = sitk.ReadImage(path.join(self.val_image_dir, item[:-12] + "lung.nii.gz"))

            self.Process_and_Save(image, label, lung, item[:8], "val_norm")

    def GenerateInitDataForTraing(self):
        # self.PrintInfo("Start init file path")
        self.init_path()

        if not len(os.listdir(self.lesion_warehouse_init)) > 0:
            self.GetLesionForRadiomics(datalist=self.labeled_image_list)

        # self.PrintInfo("Calculate radiomics features")
        # radiomics_path = os.path.join(self.output_dir, 'Labeled_Radiomics.csv')
        # if os.path.exists(radiomics_path):
        #     # print("*" * (len("Read Radiomics Features") * 3 + 2))
        #     print("*" * len("Read Radiomics Features"), "Read Radiomics Features", "*" * len("Read Radiomics Features"))
        #     # print("*" * (len("Read Radiomics Features") * 3 + 2))
        #     radiomics_features = pd.read_csv(radiomics_path, header=0, index_col=0)
        # else:
        #     radiomics_features = self.Cal_Radiomics_MP()
        #
        # data_dict = self.Update_lesion_calss(radiomics_features, n_clusters=3)
        # self.Update_lesion_warehouse(data_dict)

        # self.Update_traindata_MP(data_dict)
        print("Process Train Data...")
        self.Generate_h5_data_MP(self.labeled_dir, "train")
        print("Process val Data...")
        self.Generate_h5_data_MP(self.val_image_dir, "val")

    def GenerateInitDataForTraing_normal(self):
        self.init_path()

        self.Precess_traindata_norm()

        self.Precess_valdata_norm()

    def Preprocess_val(self, image, size=320):
        original_spacing = image.GetSpacing()
        image_array = sitk.GetArrayFromImage(image)
        original_size = image_array.shape
        original_size = list(original_size)
        # original_size.reverse()
        image_re, _, _ = self.ResapmleImage2(image=image, targetSpacing=(1.0, 1.0, 1.0))
        resampled_size = image_re.GetSize()
        resampled_size = list(resampled_size)
        resampled_size.reverse()
        image_norm = self.Image_Normalization(image_re)  # targetSpacing=(1.0, 1.0, 1.0)
        image_arr = sitk.GetArrayFromImage(image_norm)

        shape = image_arr.shape
        volume_ymin = (shape[1] - size) // 2
        volume_ymax = volume_ymin + size

        volume_xmin = (shape[2] - size) // 2
        volume_xmax = volume_xmin + size
        image_arr = image_arr[:, volume_ymin:volume_ymax, volume_xmin:volume_xmax]
        cut_coordinate = [volume_xmin, volume_xmax]
        return image_arr, original_spacing, original_size, resampled_size, cut_coordinate

    def Restore_val(self, label: np.ndarray, target_spacing, target_size, resampled_size, cut_coordinate):
        new_label = np.zeros(resampled_size)
        new_label[:, cut_coordinate[0]:cut_coordinate[1], cut_coordinate[0]:cut_coordinate[1]] = label
        # new_label = sitk.GetImageFromArray(new_label)
        # new_label.SetSpacing((1, 1, 1))
        zoom_ = [1 / target_spacing[2], 1 / target_spacing[1], 1 / target_spacing[0]]
        new_label = ndimage.zoom(new_label, zoom_, order=0)
        new_label = np.array(new_label, dtype=np.uint8)
        assert target_size == list(new_label.shape)
        return new_label

    def Combine_external_lesion_warehouse(self, external_warehouse_path, external_radiomics_feature):
        final_external_path = self.output_dir + "/final_lesion_warehouse"
        external_lesions = os.listdir(external_warehouse_path)
        internal_lesions = os.listdir(self.lesion_warehouse_init)
        for lesion in external_lesions:
            shutil.copy(os.path.join(external_warehouse_path, lesion), os.path.join(final_external_path, lesion))

        for lesion in internal_lesions:
            shutil.copy(os.path.join(self.lesion_warehouse_init, lesion), os.path.join(final_external_path, lesion))

    def preprocess_test(self, test_path, output_path):
        test_image = os.listdir(test_path)
        test_image = [image for image in test_image if "image" in image]
        for item in test_image:
            image = sitk.ReadImage(os.path.join(test_path, item), sitk.sitkInt16)
            label = sitk.ReadImage(os.path.join(test_path, item.replace("image", "lesion")), sitk.sitkUInt8)
            image, label, lung = self.Resample_and_Normalization(image, label)
            image_arr = sitk.GetArrayFromImage(image)
            label_arr = sitk.GetArrayFromImage(label)
            label_arr = np.array(label_arr, dtype=np.uint8)
            lung_arr = sitk.GetArrayFromImage(lung)
            data_flag = "test"
            if "train" in data_flag:
                target_z = self.GetBoundarybyLung(lung_arr)  # GetBoundary2 针对肺实质获取patch范围
            else:
                target_z = self.GetBoundarybyLesion(label_arr)  # GetBoundary3 针对ROI获取patch范围

            for z in target_z:
                # self.SavePatch2d(image_arr[z, ...], label_arr[z, ...], lung_arr[z, ...], z, image_name, data_flag)
                image_ = image_arr[z, ...]
                label_ = label_arr[z, ...]
                lung_ = lung_arr[z, ...]
                save_path = output_path
                id_ = item[:8] + "_{}".format(z)
                f = h5py.File(save_path + '/{}_slice_{}.h5'.format(id_, z), 'w')
                f.create_dataset('image', data=image_, compression="gzip")
                f.create_dataset('label', data=label_, compression="gzip")
                f.create_dataset('lung', data=lung_, compression="gzip")
                f.close()


if __name__ == '__main__':
    # Run 1
    # Lesion Decoupling and Radiomics Unsupervised Clustering re-enhancement Method for Lung Cancer Medical Image Segmentation
    random.seed(618)
    precesser = Preprocesser("/data1/data/LIDC/raw",
                             "/data1/data/LIDC/pgp_test")
    # LDRUC
    precesser.GenerateInitDataForTraing()
    # precesser.preprocess_test("/data1/data/LIDC/only_Lesion/test_image", "/data1/data/LIDC/only_Lesion/test_image_slice")
    # precesser.GenerateInitDataForTraing_normal()
    # Run 2
    # precesser.Standard_pipeline()

    # precesser.test_connected_domain_2(
    #     '/media/Deepin/003/chen_data/semi-supervision data/shanxi-cui/raw/imagesTr/P0006260_0000.nii.gz',
    #     '/media/Deepin/003/chen_data/semi-supervision data/shanxi-cui/raw/labelsTr/P0006260.nii.gz')

    # precesser.Draw_single_slice_img_example()
    # precesser.Normal_pipeline()
    # precesser = Preprocesser("/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/raw",
    #                          "/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/data3(4d)")
    # precesser.Draw_single_slice_img_example("10033962", 192)

    # Test 1
