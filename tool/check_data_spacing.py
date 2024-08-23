import os
import SimpleITK as sitk
from tqdm import tqdm

root = "/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LUAD GD/raw/imagesTr"
file_list = os.listdir(root)
spacing_max0 = 0
spacing_max1 = 0
spacing_max2 = 0
spacing_min0 = 10
spacing_min1 = 10
spacing_min2 = 10
for file in tqdm(file_list):
    image = sitk.ReadImage(os.path.join(root, file))
    spacing = image.GetSpacing()
    if spacing[0] > spacing_max0:
        spacing_max0 = spacing[0]
    if spacing[0] < spacing_min0:
        spacing_min0 = spacing[0]

    if spacing[1] > spacing_max1:
        spacing_max1 = spacing[1]
    if spacing[1] < spacing_min1:
        spacing_min1 = spacing[1]

    if spacing[2] > spacing_max2:
        spacing_max2 = spacing[2]
    if spacing[2] < spacing_min2:
        spacing_min2 = spacing[2]
print(spacing_max0, spacing_max1, spacing_max2, spacing_min0, spacing_min1, spacing_min2)
