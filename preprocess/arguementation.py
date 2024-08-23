import SimpleITK as sitk
import os
import numpy as np
import h5py
import random
import cv2
from PIL import Image
import torch
from torchvision import transforms
'''input_image, lesion_mask, lung_mask是通过sitk.ReadImage()
函数读取的CT影像图像，病灶mask和肺实质的mask，以上图像都是3维的。
lesion_list: 是一个列表，其中包含多个numpy.dnarray类型的3维数组，每一个数组代表一个独立
的病灶。病灶都是三维的

return：new_image, new_lesion_mask  # 新的sitk.Image对象，new_image,是原始图像（可
能存在原始病灶）与lesion_list结合后的新CT图像，new_lesion_mask是与new_image匹配的
包含所有病灶区域的新mask。
~~~~~~~~~~~~~~~~~~~~~~~
方法描述：
将lesion_list: 中的所有病灶插入到input_image图像中，且要根据lung_mask限制病灶的位置，
任何病灶不得超出肺实质的范围，任何病灶不得相互重合。lesion_mask可能有也可能没有，若
lesion_mask存在的话就要注意来自lesion_list的新病灶也不要与lesion_mask中的病灶重合。
'''

class LessionAugmentation:
    def __init__(self, input_image, lesion_mask, lung_mask, argument_rate, argument_proportion):
        self.input_image = input_image
        self.lesion_mask = lesion_mask
        self.lung_mask = lung_mask
        self.argument_rate = argument_rate
        self.argument_proportion = argument_proportion
        # 病灶路径
        self.lesion_warehouse = r"F:\test\lesion_warehouse\class"

    # 根据argument_proportion比例选择病灶类别
    # 思路：若argument_proportion=[0.2,0.3,0.5],则随机从1-10中选择一个数字，若为1-2，判定类别1；若为2-3，判定类别2；若为5-10，判定类别3.
    def GetLesion(self):
        randomNum = np.random.rand()
        compareNum = lambda x, y: 1 if x < y else 0
        lesion_class = 1
        for i in range(len(self.argument_proportion)):
            if compareNum(randomNum, sum(self.argument_proportion[:i])) == 1:
                lesion_class = i + 1
                break
        lesion_path = os.path.join(self.lesion_warehouse, "class{}".format(lesion_class))
        lesion_list = os.listdir(lesion_path)
        h5f = h5py.File(os.path.join(lesion_path, lesion_list[np.random.randint(0, len(lesion_list))]), 'r')
        lesion = h5f["lesion"][:]
        mask = h5f["label"][:] * h5f["class"][0]
        return lesion, mask

    def Lesion_arguement(self, lesion, lesion_mask):
        n = random.random()
        if n > 0.5:
            random

    # 对比增强
    def Contrast(self, img):
        # 图像方差
        std = np.sqrt(np.var(img))
        if std <= 3:
            p = 3.0
        elif std <= 10:
            p = (27 - 2 * std) / 7
        else:
            p = 1.0

        In = img / 255.0
        G = cv2.GaussianBlur(img, (5, 5), 0)

        E = np.power(((G + 0.1) / (img + 0.1)), p)
        res = np.power(In, E)

        dst = np.uint8(res * 255.0)
        return dst

    # 对slice进行随机增强
    def SliceAumentation(self, imageSlice, labelSlice, n):
        # 放缩倍数
        scaleNum = random.randint(2, 5)
        # 旋转角度
        rotNum = random.choice([-3, -2, -1, 1, 2, 3])
        if n == 0:
            # 水平翻转
            Img_Aumentation = cv2.flip(imageSlice, 1)
            label_Augmentation = cv2.flip(imageSlice, 1)
        elif n == 1:
            # 纵向翻转
            Img_Aumentation = cv2.flip(imageSlice, 0)
            label_Augmentation = cv2.flip(imageSlice, 0)
        elif n == 2:
            # 旋转
            Img_Aumentation = np.rot90(imageSlice, rotNum)
            label_Augmentation = np.rot90(imageSlice, rotNum)
        elif n == 3:
            # 扩大
            Img_Aumentation = cv2.resize(imageSlice, (imageSlice.shape[0] * scaleNum, imageSlice.shape[1] * scaleNum),
                                      interpolation=cv2.INTER_CUBIC)
            label_Augmentation = cv2.resize(imageSlice, (imageSlice.shape[0] * scaleNum, imageSlice.shape[1] * scaleNum),
                                      interpolation=cv2.INTER_CUBIC)
        elif n==4:
            # 缩小
            Img_Aumentation = cv2.resize(imageSlice, (imageSlice.shape[0] // scaleNum, imageSlice.shape[1] // scaleNum),
                                      interpolation=cv2.INTER_CUBIC)
            label_Augmentation = cv2.resize(imageSlice, (imageSlice.shape[0] // scaleNum, imageSlice.shape[1] // scaleNum),
                                      interpolation=cv2.INTER_CUBIC)
        # 对比度增强
        elif n==5:
            Img_Aumentation = self.Contrast(imageSlice)
            label_Augmentation = labelSlice
        # 高斯模糊
        else:
            Img_Aumentation = cv2.GaussianBlur(imageSlice, (9, 9), 0)
            label_Augmentation = labelSlice
        return Img_Aumentation, label_Augmentation

    # 随机寻找病灶包含肺实质的位置，返回坐标
    def findLocation(self,label_Aug,lg_mask,ls_mask):
        x = []
        for i in range(lg_mask.shape[0]):
            if np.max(lg_mask[i, :]) == 1:
                x.append(i)
        y = []
        for i in range(lg_mask.shape[1]):
            for j in range(lg_mask.shape[0]):
                if lg_mask[j][i] == 1:
                    y.append(i)

        MaxMin_X = max(x)-min(x)
        MaxMin_y = max(y)-min(y)
        s = []
        counter = 0
        for i in range(min(x),max(x)):
            for j in range(min(y),max(y)):
                if lg_mask[i][j]:
                    s.append([i,j])
                    counter+= 1

        gg = 1
        origin_x = 0
        origin_y = 0
        while gg:
            randompoint = np.random.randint(counter)
            origin_x = s[randompoint][0]
            origin_y = s[randompoint][1]
            allvalue = []
            for i in range(label_Aug.shape[0]):
                if MaxMin_y-origin_y > label_Aug.shape[1] and MaxMin_X - origin_x > label_Aug.shape[0]:
                    # 判断是否在肺实质内
                    if np.min(lg_mask[origin_x+i, origin_y:origin_y + label_Aug.shape[1]])==1:
                        allvalue.append(np.max(ls_mask[origin_x+i, origin_y:origin_y + label_Aug.shape[1]]))
                    else:
                        allvalue.append(0)
                else:
                    allvalue.append(0)
            if min(allvalue) == 0:
                gg=0

        return origin_x,origin_y

    # 将病灶加在原图像上
    def ADDlession(self,image,Img_Aug,origin_x, origin_y,flag):
        if flag==1:
            randomN = random.random()
            for i in range(Img_Aug.shape[0]):
                for j in range(Img_Aug.shape[1]):
                    if origin_x + i < image.shape[0] and origin_y + j < image.shape[1]:
                        image[origin_x + i][origin_y + j] = randomN * (Img_Aug[i][j] - image[origin_x + i][origin_y + j])
        else:
            for i in range(Img_Aug.shape[0]):
                for j in range(Img_Aug.shape[1]):
                    if origin_x + i < image.shape[0] and origin_y + j < image.shape[1]:
                        image[origin_x + i][origin_y + j] = Img_Aug[i][j]
        return image

    def pad(self,imageArray):
        image = Image.fromarray(imageArray)
        imgSize = image.size  # 大小/尺寸
        w = image.width  # 图片的宽
        h = image.height

        left = int((384 - w) / 2)
        right = left
        if w + left + right != 384:
            left = 384 - w - right

            top = int((384 - h) / 2)
            bottom = top
        if h + top + bottom != 384:
            top = 384 - h - bottom

        new_image = cv2.copyMakeBorder(imageArray, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                         value=[255, 255, 255])
        return new_image

    def Lesionassemble(self):
        classDict = self.classifyLession()
        image = self.input_image
        ls_mask = self.lesion_mask
        lg_mask = self.lung_mask
        # 根据比例获取病灶类别
        classNum = self.Chooselesion()
        print(classNum)
        # 获取所有符合病灶类别的文件名
        findFilename = lambda d, value: [k for k, v in d.items() if v == value]
        classFilenames = findFilename(classDict,classNum)
        # 根据增强倍数随机抽取结节
        lessions = random.sample(classFilenames,self.argument_rate)
        s=0
        for lession in lessions:
            s=s+1
            print(s)
            h5 = h5py.File(self.lesion_warehouse + '/' + lession)
            lession_img = h5["lesion"][:]
            lession_label = h5['lesion_mask'][:]
            # 随机抽取病灶的slice
            randomSlice = np.random.randint(0,lession_img.shape[0])
            imageSlice = lession_img[randomSlice]
            labelSlice = lession_label[randomSlice]
            # 随机增强
            Img_Aug, label_Aug = self.SliceAumentation(imageSlice,labelSlice,random.randint(0,4))
            origin_x, origin_y = self.findLocation(label_Aug,lg_mask,ls_mask)
            print("origin_x,origin_y",origin_x,origin_y)
            image = self.ADDlession(image,Img_Aug,origin_x, origin_y,1)
            ls_mask = self.ADDlession(ls_mask,label_Aug,origin_x, origin_y,0)

        new_image = self.pad(image)
        new_ls_mask = self.pad(ls_mask)

        ne = Image.fromarray(new_image)
        ne.show()

        # 图片级增强
        new_image, new_ls_mask = self.SliceAumentation(new_image,new_ls_mask,random.choice([0, 1, 2, 5, 6]))
        # 图像归一化
        new_image = (new_image - np.min(new_image)) / (np.max(new_image) - np.min(new_image))
        # 转为torch
        new_image = torch.tensor(new_image, dtype=torch.float32)

        return new_image, new_ls_mask

if __name__ == "__main__":
    path = r'F:\test\train\10013476_slice_24.h5'
    h5 = h5py.File(path)
    img = h5['image'][:]
    label = h5['label'][:]
    lung = h5["lung"][:]
    aug = LessionAugmentation(img, label, lung, 4, [0.6, 0.3, 0.1])
    aug.Lesionassemble()



