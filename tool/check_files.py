import os
import SimpleITK as sitk
from tqdm import tqdm
path = "/data1/data/LIDC/raw"
# image_list = os.listdir(path)
# image_list = [image for image in image_list if "image"]
image_path = os.path.join(path, "imagesTr")
label_path = os.path.join(path, "labelsTr")
image_list = os.listdir(image_path)
label_list = os.listdir(label_path)
for image in tqdm(image_list):
    image_obj = sitk.ReadImage(os.path.join(image_path, image))
    label_obj = sitk.ReadImage(os.path.join(label_path, image.replace("_0000", "")))
    image_arr = sitk.GetArrayFromImage(image_obj)
    label_arr = sitk.GetArrayFromImage(label_obj)
    image_shape = image_arr.shape
    label_shape = label_arr.shape
    if len(image_shape) < 3 or len(label_shape) < 3:
        print(image, image_shape, label_shape)

    if image_shape[0]<50 or label_shape[0]<50:
        print(image, image_shape)

