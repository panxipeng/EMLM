import copy
import os
import cv2
import scipy.ndimage
import torch
import random
# import numpy as np
import numpy as np
import SimpleITK as sitk
from glob import glob
from torch.utils.data import Dataset
import h5py
# from scipy.ndimage.interpolation import zoom
from scipy.ndimage import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom


class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        # scipy.ndimage.zoom()
        if self.split == "train":
            self.sample_list = os.listdir(self._base_dir + "/train")
            self.sample_list = [case[:-3] for case in self.sample_list]
            # self.sample_list = self.sample_list[:32]
        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/val")
            self.sample_list = [case[:-3] for case in self.sample_list]
            # self.sample_list = self.sample_list[:16]
        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        # print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/train/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/val/{}.h5".format(case), "r")

        image = h5f["image"][:]
        image = np.array(image, dtype=np.float64)
        label = h5f["label"][:]
        label = np.array(label, dtype=np.uint8)
        # lung_area = h5f["lung_area"][:]
        lung = h5f["lung"][:]
        lung = np.array(lung, dtype=np.uint8)
        # if self.split == "train":
        #     lung_area = h5f["lung_area"]
        sample = {"image": image, "label": label, "lung": lung}
        if self.transform is not None:
            sample = self.transform(sample)
        else:
            image = torch.from_numpy(image).unsqueeze(0)
            label = torch.from_numpy(label).unsqueeze(0)
            # image = torch.from_numpy(image.astype(np.float32))
            # label = torch.from_numpy(label.astype(np.float32))
            sample = {"image": image, "label": label}
        sample["idx"] = idx
        # sample["label"] = sample["label"] / 1.0

        return sample


class Lesion_and_image_augmentation:
    def __init__(self, lesion_path, augmentation_proportion: list, lesion_count: int):
        self.augmentation_proportion = augmentation_proportion
        self.lesion_count = lesion_count
        self.lesion_path = lesion_path
        self.LesionClassList = os.listdir(self.lesion_path)
        self.LesionClassList.remove("init")
        self.CC = transforms.CenterCrop(512)
        self.trs = transforms.Normalize(mean=[0.5], std=[0.5])
        self.config = {
            "max_rotation": 45,  #
            "max_brightness": 0.05,  #
            "max_contrast": 0.1,  ##
            "gaussian_noise_variance": 0.01,  #
            "gaussian_blur_sigma_range": [0.5, 1],  # #
            "gamma_range": [0.7, 1.5],  #

            "p_rotate": 0.5,
            "p_gaussian_noise": 0.5,
            "p_gaussian_blur": 0.5,
            "p_brightness": 0.5,
            "p_contras": 0.5,
            "p_gamma": 0.5,
            "p_mirror": 0.5
        }

    def __call__(self, sample):
        # self.config['img_fill_value'] = image.min()
        # assert self.lesion_count > 0
        # if np.sum(sample["label"][:]) != 0:
        #     print()
        new_sample = self.LesionAssemble(sample)
        image = new_sample["image"]
        label = new_sample["label"]

        # image = np.asarray(image, dtype=np.fl)



        image = self.add_gaussian_noise(image)
        image = self.add_gaussian_blur(image)
        image = self.adjust_brightness(image)
        image = self.adjust_contrast(image)
        # image = self.augment_gamma(image)
        # image = self.totensor(image.copy())
        # label = self.totensor(label.copy())
        image, label = self.random_rotate(image.astype(np.float32), label.astype(np.float32))
        image, label = self.random_flip(image, label)
        label = self.Remove_Small_connected_domain(label)
        # label = self.Remove_Small_connected_domain(label)
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        label = torch.from_numpy(label.copy()).unsqueeze(0)
        # image = self.trs(image)
        image = self.CC(image)
        label = self.CC(label)
        # image = self.trs(image)
        # img, label = self.mirror(img, label)
        # img, label = self.check_shape(img, label)
        # assert img.shape == tuple(self.config['crop_shape']), 'The shape of img is wrong.'
        # img_aug.append(img)
        # label_aug.append(label)
        # self.Transform_array_to_png(image, label, "imageAug", './')
        new_sample = {'image': image, 'label': label}
        return new_sample

    def LesionAssemble(self, sample):
        image = sample["image"]
        label = sample["label"]
        lung = sample["lung"]
        # if np.sum(lung) < 15000:
        #     return {"image": image, "label": label}
        # self.Transform_array_to_png(image, label, "ori", './')  # 图片不正确，仍需调整。
        ChoicedClassList = np.random.choice(self.LesionClassList, size=self.lesion_count, p=self.augmentation_proportion)
        for CurrentClass in ChoicedClassList:
            lesion_path = os.path.join(self.lesion_path, CurrentClass)
            LesionList = os.listdir(lesion_path)
            lesion = random.sample(LesionList, 1)[0]

            h5f = h5py.File(os.path.join(lesion_path, lesion), 'r')

            lesion_arr = h5f["lesion"][:]
            lesion_mask = h5f["lesion_mask"][:]

            # lesion_mask = self.Remove_Small_connected_domain(lesion_mask)

            lesion_arr, lesion_mask = self.Generate_lesion_augmentation(lesion_arr, lesion_mask)
            # 随机选择一层
            try:
                random_index = np.random.randint(lesion_arr.shape[0])
            except ValueError as e:
                print(lesion_arr.shape)
                print(lesion_arr.shape[0])
                raise e
            lesion_arr = lesion_arr[random_index, :, :]
            lesion_mask = lesion_mask[random_index, :, :]
            count = 0
            new_lung = copy.deepcopy(lung)
            while True:
                count += 1
                if count > 50:
                    # print("lung area: {};".format(np.sum(lung)))
                    # raise RuntimeError("LesionAssemble error")
                    break
                lung_area = list(np.argwhere(new_lung != 0))
                if len(lung_area) <= 100:
                    break
                assemble_cord = list(random.sample(lung_area, 1)[0])
                assemble_cord = [int(assemble_cord[0] - lesion_arr.shape[0] / 2),
                                 int(assemble_cord[1] - lesion_arr.shape[1] / 2)]
                if assemble_cord[0] < 0:
                    assemble_cord[0] = 0
                if assemble_cord[1] < 0:
                    assemble_cord[1] = 0
                if assemble_cord[0] + lesion_arr.shape[0] > label.shape[0]:
                    assemble_cord[0] = label.shape[0] - lesion_arr.shape[0]
                if assemble_cord[1] + lesion_arr.shape[1] > label.shape[1]:
                    assemble_cord[1] = label.shape[1] - lesion_arr.shape[1]

                if not self.overlap_check(assemble_cord, lesion_mask, label):
                    continue

                local_image = image[assemble_cord[0]: assemble_cord[0] + lesion_mask.shape[0],
                              assemble_cord[1]: assemble_cord[1] + lesion_mask.shape[1]]

                local_lung = lung[assemble_cord[0]: assemble_cord[0] + lesion_mask.shape[0],
                              assemble_cord[1]: assemble_cord[1] + lesion_mask.shape[1]]
                local_label = local_lung * lesion_mask

                new_local_image = np.where(local_label > 0, lesion_arr, local_image)

                image[assemble_cord[0]: assemble_cord[0] + lesion_mask.shape[0], assemble_cord[1]: assemble_cord[1] + lesion_mask.shape[1]] = new_local_image
                label[assemble_cord[0]: assemble_cord[0] + lesion_mask.shape[0], assemble_cord[1]: assemble_cord[1] + lesion_mask.shape[1]] = local_label
                label_l = copy.deepcopy(label)
                label_l[label_l>0] = 1
                new_lung = lung - label_l
                break
            one_label = copy.deepcopy(label)
            one_label[one_label > 0] = 1
            lung = lung - one_label
            lung[lung != 1] = 0
            # lung_area_list = np.argwhere(lung == 1)
            # lung_area = np.argwhere(lung == 1)
        # self.Transform_array_to_png(image, label, "lesionAug", './')
        new_sample = {"image": image, "label": label}
        return new_sample

    def overlap_check(self, assemble_cord, lesion_mask, label):
        new_label = copy.deepcopy(label)
        new_label[assemble_cord[0]: assemble_cord[0] + lesion_mask.shape[0], assemble_cord[1]: assemble_cord[1] + lesion_mask.shape[1]] = lesion_mask

        if np.sum(new_label) == np.sum(label) + np.sum(lesion_mask):
            return True
        else:
            return False

    def Generate_lesion_augmentation(self, lesion, lesion_mask):
        lesion, lesion_mask = self.Lesion_rot(lesion, lesion_mask)

        lesion, lesion_mask = self.Lesion_flip(lesion, lesion_mask)

        lesion, lesion_mask = self.Lesion_resize(lesion, lesion_mask)

        return lesion, lesion_mask

    def Remove_Small_connected_domain(self, image):
        min_area_thr = 40
        # img1, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # contours返回的是轮廓像素点列表，一张图像有几个目标区域就有几个列表值
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.array(image, dtype=np.uint8), connectivity=8)
        # 筛选超过平均面积的连通域
        image_filtered = image
        for (i, label) in enumerate(np.unique(labels)):
            # 如果是背景，忽略
            if label == 0:
                continue
            if stats[i][-1] < min_area_thr:
                # print(image[int(centroids[i, 0]), int(centroids[i, 1])])
                # print(centroids[i, 0], centroids[i, 1])
                image_filtered[labels == i] = 0
        # for i in range(len(contours)):
        #     area = cv2.contourArea(contours[i])
        #     if area < threshold:
        #         cv2.drawContours(image, [contours[i]], 0, 0, -1)  ##去除小面积连通域

        return image_filtered

    def Lesion_rot(self, image, label):
        # rot_range = [rot for rot in range(0, 360, 45)]
        rotation_angle_x = random.randint(-15, 15)
        rotation_angle_y = random.randint(-15, 15)
        rotation_angle_z = random.randint(-180, 180)

        new_image = rotate(image, rotation_angle_x, axes=(0, 1), reshape=False, order=3)
        new_image = rotate(new_image, rotation_angle_y, axes=(0, 2), reshape=False, order=3)
        new_image = rotate(new_image, rotation_angle_z, axes=(1, 2), reshape=False, order=3)

        new_label = rotate(label, rotation_angle_x, axes=(0, 1), reshape=False, order=0)
        new_label = rotate(new_label, rotation_angle_y, axes=(0, 2), reshape=False, order=0)
        new_label = rotate(new_label, rotation_angle_z, axes=(1, 2), reshape=False, order=0)

        return new_image, new_label

    def Lesion_flip(self, image, label):
        flip_range = [0, 1]
        flip_axis_x = random.sample(flip_range, 1)[0]
        flip_axis_y = random.sample(flip_range, 1)[0]
        flip_axis_z = random.sample(flip_range, 1)[0]
        new_image = image
        new_label = label
        if flip_axis_x != 0:
            new_image = np.flip(new_image, axis=1)
            new_label = np.flip(new_label, axis=1)
        if flip_axis_y != 0:
            new_image = np.flip(new_image, axis=2)
            new_label = np.flip(new_label, axis=2)
        if flip_axis_z != 0:
            new_image = np.flip(new_image, axis=0)
            new_label = np.flip(new_label, axis=0)

        return new_image, new_label

    def add_gaussian_noise(self, data):
        """随机高斯噪声"""
        if np.random.uniform() <= self.config['p_gaussian_noise']:
            variance = np.random.uniform(0, self.config['gaussian_noise_variance'])
            noise = np.random.normal(0, variance, size=data.shape)
            data = data + noise
        return data

    def add_gaussian_blur(self, data):
        """随机高斯模糊"""
        if np.random.uniform() <= self.config['p_gaussian_blur']:
            lower, upper = self.config['gaussian_blur_sigma_range'][0], self.config['gaussian_blur_sigma_range'][1]
            sigma = np.random.uniform(lower, upper)
            data = gaussian_filter(data, sigma=sigma, order=0)
        return data

    def adjust_brightness(self, data):
        """随机亮度变换"""
        if np.random.uniform() <= self.config['p_brightness']:
            alpha = np.random.uniform(1 - self.config['max_brightness'], 1 + self.config['max_brightness'])
            data = alpha * data
        return data

    def adjust_contrast(self, data):
        """随机对比度变换"""
        if np.random.uniform() <= self.config['p_contras']:
            factor = np.random.uniform(1 - self.config['max_contrast'], 1 + self.config['max_contrast'])
            mn, minm, maxm = data.mean(), data.min(), data.max()

            data = (data - mn) * factor + mn
            data = np.clip(data, a_min=minm, a_max=maxm)
        return data

    def augment_gamma(self, data, epsilon=1e-7):
        if np.random.uniform() <= self.config['p_gamma']:
            data_sample = - data
            mn, sd, minm = data_sample.mean(), data_sample.std(), data_sample.min()
            rnge = data_sample.max() - minm

            gamma = np.random.uniform(self.config['gamma_range'][0], self.config['gamma_range'][1])
            data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn

            data = - data_sample
        return data

    def Lesion_resize(self, image, label):
        # import random
        # resize_range = [0.8, 0.9, 1.0, 1.1, 1.2]
        # resize_factor = random.sample(resize_range, 1)[0]
        resize_factor = random.uniform(0.8, 1.2)

        new_image = image
        new_label = label

        new_image = zoom(new_image, resize_factor, order=3)
        new_label = zoom(new_label, resize_factor, order=0)

        return new_image, new_label

    def random_rotate(self, img, label):
        if np.random.uniform() <= self.config['p_rotate']:
            angle = np.random.uniform(-self.config['max_rotation'], self.config['max_rotation'])

            img = rotate(img, angle, axes=(0, 1), reshape=True, order=3)

            label = rotate(label, angle, axes=(0, 1), reshape=True, order=0)

        return img, label

    def random_flip(self, image, label):
        flip_range = [0, 1]
        flip_axis = random.sample(flip_range, 1)

        new_image = np.flip(image, axis=flip_axis)
        new_label = np.flip(label, axis=flip_axis)

        return new_image, new_label

    def UpdateAugmentation_proportion(self, new_Augmentation_proportion):
        self.augmentation_proportion = new_Augmentation_proportion

    @staticmethod
    def Transform_array_to_png(image_arr: np.ndarray, label_arr: np.ndarray, image_flag, save_path):
        image_arr = image_arr * 255
        image = Image.fromarray(np.array(image_arr, dtype=np.uint8))
        label_arr = label_arr * 255
        label = Image.fromarray(np.array(label_arr, dtype=np.uint8))
        image.save(os.path.join(save_path, "test_image_{}.png".format(image_flag)))
        label.save(os.path.join(save_path, "test_label_{}.png".format(image_flag)))

    @staticmethod
    def Transform_tensor_to_png(image_tensor: torch.Tensor, label_tensor: torch.Tensor, image_id, save_path):
        image_arr = image_tensor.numpy()
        label_arr = label_tensor.numpy()
        image_arr = image_arr * 255
        image = Image.fromarray(np.array(image_arr, dtype=np.uint8))
        label_arr = label_arr * 85
        label = Image.fromarray(np.array(label_arr.squeeze(0), dtype=np.uint8))
        image.save(os.path.join(save_path, str(image_id) + "_image.png"))
        label.save(os.path.join(save_path, str(image_id) + "_label.png"))

class Image_augmentation():
    def __init__(self, patchsize):
        # super().__init__()
        # self.lesion_path = "/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/My_data_enhancement/lesion_warehouse"
        # self.LesionClassList = os.listdir(self.lesion_path)
        # self.LesionClassList.remove("init")
        self.CC = transforms.CenterCrop(patchsize)
        self.trs = transforms.Normalize(mean=[0.5], std=[0.5])
        self.config = {
            "max_rotation": 45,  #
            "max_brightness": 0.1,  #
            "max_contrast": 0.25,  ##
            "gaussian_noise_variance": 0.01,  #
            "gaussian_blur_sigma_range": [0.5, 1],  # #
            "gamma_range": [0.7, 1.5],  #

            "p_rotate": 0.5,
            "p_gaussian_noise": 0.5,
            "p_gaussian_blur": 0.5,
            "p_brightness": 0.5,
            "p_contras": 0.5,
            "p_gamma": 0.5,
            "p_mirror": 0.5
        }

    def __call__(self, sample):
        new_sample = sample
        image = new_sample["image"]
        label = new_sample["label"]
        label[label > 0] = 1

        # image = np.asarray(image, dtype=np.fl)

        image = self.add_gaussian_noise(image)
        image = self.add_gaussian_blur(image)
        image = self.adjust_brightness(image)
        image = self.adjust_contrast(image)
        # image = self.augment_gamma(image)
        # image = self.totensor(image.copy())
        # label = self.totensor(label.copy())
        image, label = self.random_rotate(image.astype(np.float32), label.astype(np.float32))
        image, label = self.random_flip(image, label)
        label = self.Remove_Small_connected_domain(label)
        # label = self.Remove_Small_connected_domain(label)
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        label = torch.from_numpy(label.copy()).unsqueeze(0)
        # image = self.trs(image)
        image = self.CC(image)
        label = self.CC(label)
        # image = self.trs(image)
        new_sample = {'image': image, 'label': label}
        return new_sample

    def add_gaussian_noise(self, data):
        """随机高斯噪声"""
        if np.random.uniform() <= self.config['p_gaussian_noise']:
            variance = np.random.uniform(0, self.config['gaussian_noise_variance'])
            noise = np.random.normal(0, variance, size=data.shape)
            data = data + noise
        return data

    def add_gaussian_blur(self, data):
        """随机高斯模糊"""
        if np.random.uniform() <= self.config['p_gaussian_blur']:
            lower, upper = self.config['gaussian_blur_sigma_range'][0], self.config['gaussian_blur_sigma_range'][1]
            sigma = np.random.uniform(lower, upper)
            data = gaussian_filter(data, sigma=sigma, order=0)
        return data

    def adjust_brightness(self, data):
        """随机亮度变换"""
        if np.random.uniform() <= self.config['p_brightness']:
            alpha = np.random.uniform(1 - self.config['max_brightness'], 1 + self.config['max_brightness'])
            data = alpha * data
        return data

    def adjust_contrast(self, data):
        """随机对比度变换"""
        if np.random.uniform() <= self.config['p_contras']:
            factor = np.random.uniform(1 - self.config['max_contrast'], 1 + self.config['max_contrast'])
            mn, minm, maxm = data.mean(), data.min(), data.max()

            data = (data - mn) * factor + mn
            data = np.clip(data, a_min=minm, a_max=maxm)
        return data

    def augment_gamma(self, data, epsilon=1e-7):
        if np.random.uniform() <= self.config['p_gamma']:
            data_sample = - data
            mn, sd, minm = data_sample.mean(), data_sample.std(), data_sample.min()
            rnge = data_sample.max() - minm

            gamma = np.random.uniform(self.config['gamma_range'][0], self.config['gamma_range'][1])
            data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn

            data = - data_sample
        return data

    def Remove_Small_connected_domain(self, image):
        min_area_thr = 40
        # img1, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # contours返回的是轮廓像素点列表，一张图像有几个目标区域就有几个列表值
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.array(image, dtype=np.uint8), connectivity=8)
        # 筛选超过平均面积的连通域
        image_filtered = image
        for (i, label) in enumerate(np.unique(labels)):
            # 如果是背景，忽略
            if label == 0:
                continue
            if stats[i][-1] < min_area_thr:
                # print(image[int(centroids[i, 0]), int(centroids[i, 1])])
                # print(centroids[i, 0], centroids[i, 1])
                image_filtered[labels == i] = 0
        # for i in range(len(contours)):
        #     area = cv2.contourArea(contours[i])
        #     if area < threshold:
        #         cv2.drawContours(image, [contours[i]], 0, 0, -1)  ##去除小面积连通域

        return image_filtered

    def random_rotate(self, img, label):
        if np.random.uniform() <= self.config['p_rotate']:
            angle = np.random.uniform(-self.config['max_rotation'], self.config['max_rotation'])

            img = rotate(img, angle, axes=(0, 1), reshape=True, order=3)

            label = rotate(label, angle, axes=(0, 1), reshape=True, order=0)

        return img, label

    def random_flip(self, image, label):
        flip_range = [0, 1]
        flip_axis = random.sample(flip_range, 1)

        new_image = np.flip(image, axis=flip_axis)
        new_label = np.flip(label, axis=flip_axis)

        return new_image, new_label

class None_augmentation():
    def __init__(self, patchsize):
        # self.LesionClassList = os.listdir(self.lesion_path)
        # self.LesionClassList.remove("init")
        self.CC = transforms.CenterCrop(patchsize)
        self.trs = transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, sample):
        # self.config['img_fill_value'] = image.min()
        # assert self.lesion_count > 0
        new_sample = sample
        image = new_sample["image"]
        label = new_sample["label"]
        label[label > 0] = 1
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        label = torch.from_numpy(label.copy())
        # image = self.trs(image)
        image = self.CC(image)
        label = self.CC(label)


        new_sample = {'image': image, 'label': label}
        return new_sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)

    shape = len(image.shape)
    if shape == 3:  # 3d noe-channels
        plane = [0, 1, 2]
        axis = np.random.randint(0, 3)
        plane = tuple(plane.remove(axis))
        image = np.rot90(image, k, axes=plane)
        image = np.flip(image, axis=axis).copy()
        if label is not None:
            label = np.rot90(label, k, axes=plane)
            label = np.flip(label, axis=axis).copy()
        else:
            label = None
    else:  # 2d one channel
        axis = np.random.randint(0, 2)
        image = np.rot90(image, k)
        image = np.flip(image, axis=axis).copy()
        if label is not None:
            label = np.rot90(label, k)
            label = np.flip(label, axis=axis).copy()
        else:
            label = None
        # plane = tuple(plane.remove(axis))
    # axis = np.random.randint(2, len(image.shape))
    # ax1 = np.random.randint(2, len(image.shape))

    # image = np.rot90(image, k, axes=plane)
    # image = np.flip(image, axis=axis).copy()
    # if label is not None and label.shape != image.shape:  # HU值分层归一化后会导致标签与图像尺寸不同，图像比标签多channel维度
    #     label = np.rot90(label, k, axes=(ax1 - 1, ax2 - 1))
    #     label = np.flip(label, axis=axis - 1).copy()
    #     return image, label
    # elif label is not None and label.shape == image.shape:
    #     label = np.rot90(label, k, axes=(ax1, ax2))
    #     label = np.flip(label, axis=axis).copy()
    #     return image, label
    # if label is not None:  # HU值分层归一化后会导致标签与图像尺寸不同，图像比标签多channel维度
    #     label = np.rot90(label, k, axes=(ax1 - 1, ax2 - 1))
    #     label = np.flip(label, axis=axis - 1).copy()
    #     return image, label
    # else:
    #     return image
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    ax1 = np.random.randint(0, len(image.shape))
    if len(image.shape) == 3:
        plane = [0, 1, 2]
        image = ndimage.rotate(image, angle, axes=tuple(plane.remove(ax1)), order=0, reshape=False)
        label = ndimage.rotate(label, angle, axes=tuple(plane.remove(ax1)), order=0, reshape=False)
    else:
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    # if label.shape != image.shape:
    #     image = ndimage.rotate(image, angle, axes=(ax1, ax2), order=0, reshape=False)
    #     label = ndimage.rotate(label, angle, axes=(ax1 - 1, ax2 - 1), order=0, reshape=False)
    # else:
    #     image = ndimage.rotate(image, angle, axes=(ax1, ax2), order=0, reshape=False)
    #     label = ndimage.rotate(label, angle, axes=(ax1, ax2), order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.totensor = transforms.ToTensor()
        self.CC = transforms.CenterCrop(384)
        self.trs = transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if len(image.shape) == 2:
            image = self.totensor(image.astype(np.float32))
            label = torch.from_numpy(label)
            # label = self.totensor(label.astype(np.uint8))
            # label[label > 0] = 1
            # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:
            image = self.totensor(image.astype(np.float32))
            label = torch.from_numpy(label)
            # label = self.totensor(label.astype(np.uint8))
            # label[label > 0] = 1
            # image = torch.from_numpy(image.astype(np.float32))
        image = self.trs(image)
        image = self.CC(image)
        label = self.CC(label)
        sample = {"image": image, "label": label}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.totensor = transforms.ToTensor()
        self.CC = transforms.CenterCrop(384)
        self.trs = transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))

        if len(image.shape) == 2:
            image = self.totensor(image.astype(np.float32))
            label = torch.from_numpy(label)
            # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:
            image = self.totensor(image.astype(np.float32))
            label = torch.from_numpy(label)
            # image = torch.from_numpy(image.astype(np.float32))
        image = self.trs(image)
        image = self.CC(image)
        label = self.CC(label)
        sample = {"image": image, "label": label}
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        for c in range(image.shape[0]):
            image[c, ...] = (image[c, ...] - self.mean[c]) / self.std[c]
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == '__main__':
    # h5f = h5py.File(
    #     '/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/My_data_enhancement/argument_TrainValTest/train/10013476_slice_168.h5',
    #     'r')
    # savepath = "./"
    # sample = {'image': h5f['image'][:], 'label': h5f['label'][:], 'lung': h5f['lung'][:]}
    # lung_area = h5f['lung_area'][:]
    # lung_area_list = h5f['lung_area'][:].tolist()
    # augment = Lesion_and_image_augmentation([0.333, 0.333, 0.334], 3)
    # # augment.Transform_array_to_png(sample['image'], sample['label'], savepath)
    # # augment.Transform_array_to_png(sample['image'], h5f["lung"][:], savepath)
    # new_sample = augment(sample, lung_area_list)

    root_path = '/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/My_data_enhancement/argument_TrainValTest'
    train_transform = Lesion_and_image_augmentation([0.333, 0.333, 0.334], 3)
    db_train = BaseDataSets(base_dir=root_path, split="train", transform=train_transform)

    for idx, data in enumerate(db_train):
        image = data['image'].squeeze(0)
        label = data['label']
        Lesion_and_image_augmentation.Transform_tensor_to_png(image, label, idx, '/media/Deepin/003/chen_data/semi-supervision data/shengyi-linhuan/My_data_enhancement/check_image')

