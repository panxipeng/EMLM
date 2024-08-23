import os
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from preprocess import Preprocesser
from networks.net_factory import net_factory

def Cal_Dice(pred:np.ndarray, gt:np.ndarray):
    smooth = 1
    pred_f = pred.flatten()
    gt_f = gt.flatten()
    intersection = np.sum(pred_f * gt_f)
    return (2 * intersection) / (np.sum(pred_f) + np.sum(gt_f) + smooth)

TestData_path = "/media/Deepin/My Passport/dataset/Lung Data/Processed Data/LUAD GD/My_method 3class/test_image"
model_path = "/home/Deepin/PycharmProjects/My-Semi-Supervision/model/Fully_Supervised_My_method_7class_50/unet/unet_best_model.pth"
data_list = os.listdir(TestData_path)
data_list = [item for item in data_list if "image" in item]
net = net_factory(net_type='unet', in_chns=1, class_num=8)
# save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
net.load_state_dict(torch.load(model_path))
# model = torch.load(model_path)
net.eval()
processor = Preprocesser()
dice = []
with torch.no_grad():
    for data in tqdm(data_list):
        image = sitk.ReadImage(os.path.join(TestData_path, data))
        label = sitk.ReadImage(os.path.join(TestData_path, data.replace("image", "lesion")))
        label_ = sitk.GetArrayFromImage(label)
        image_, original_spacing, original_size, resampled_size, cut_coordinate = processor.Preprocess_val(image)
        predict = []
        for i in tqdm(range(image_.shape[0])):
            slice = image_[i, ...]
            slice = torch.from_numpy(slice.copy()).unsqueeze(0).unsqueeze(0).type(torch.float32).cuda()
            pre = net(slice)
            pre = torch.argmax(pre.squeeze(), dim=0)
            pre = pre.detach().cpu().numpy()
            predict.append(pre)
        predict = np.array(predict, dtype=np.uint8)
        predict[predict > 0] = 1
        predict = processor.Restore_val(predict, original_spacing, original_size, resampled_size, cut_coordinate)
        predict_save = sitk.GetImageFromArray(predict)
        predict_save.SetSpacing(original_spacing)
        sitk.WriteImage(predict_save, os.path.join(TestData_path, data.replace("image", "predict")))
        dice_cof = Cal_Dice(predict, label_)
        print(data, ":", dice_cof)
        dice.append(dice_cof)
dice = np.mean(dice)
print("mean DICE: {}".format(dice))