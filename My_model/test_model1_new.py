import os
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from torchvision import transforms
from medpy import metric
# from dataloaders.dataset import Lesion_and_image_augmentation, Image_augmentation, TwoStreamBatchSampler, \
#     None_augmentation
from preprocess.preprocess import Preprocesser
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# file:///media/Deepin/003/chen_data/semi-supervision data/LIDC/data20230213_3d4c_3296
parser = argparse.ArgumentParser()
#
parser.add_argument('--dataset', type=str,
                    default='GD', help='Name of Experiment')
parser.add_argument('--model_path', type=str,
                    default="/data1/code/model/LUAD GD/model_baseline_50/Entropy_Minimization_with_LesionMix2/unet/unet_best_model.pth",
                    help='Name of Experiment')
parser.add_argument('--target_path', type=str,
                    default='/data1/code/model/LUAD GD/model_baseline_50/Entropy_Minimization_with_LesionMix2//predict', help='Name of Experiment')
parser.add_argument('--root_path', type=str,
                    default='/data1/data/GD/only_Lesion', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--patch_size', type=list, default=512,
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
args = parser.parse_args()


def Cal_Dice(pred: np.ndarray, gt: np.ndarray):
    smooth = 1
    pred_f = pred.flatten()
    gt_f = gt.flatten()
    intersection = np.sum(pred_f * gt_f)
    return (2 * intersection) / (np.sum(pred_f) + np.sum(gt_f) + smooth)


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray):
    if np.sum(pred) == 0 or np.sum(pred) == 0:
        return [0.0, 0.0, 512, 512, 0.0]
    pred = np.array(pred, dtype=np.int16)
    gt = np.array(gt, dtype=np.int16)
    dice = metric.dc(pred, gt)
    hd95 = metric.hd95(pred, gt)
    asd = metric.asd(pred, gt)
    jc = metric.jc(pred, gt)
    tp = np.sum(pred * gt)
    fn = np.sum((np.ones_like(pred) - pred) * gt)
    fp = np.sum(pred * (np.ones_like(gt) - gt))
    iou = tp / (tp + fn + fp)
    return [dice, iou, hd95, asd, jc]


def test_single_volume3D(image, label, net, classes):
    image = image.type(torch.float32).cuda()
    # label = label.type(torch.float32).cuda()
    # image = image.float().cuda()
    with torch.no_grad():
        output = torch.argmax(torch.softmax(
            net(image), dim=1), dim=1).squeeze(0)
        output = output.cpu().detach().numpy()
    prediction = np.asarray(output)
    prediction[prediction > 0] = 1
    label = np.asarray(label.squeeze(1))
    metric_list = []
    # for i in range(0, 2):
    #     metric_list.append(calculate_metric_percase(prediction == i, label == i))
    metric_list.append(calculate_metric_percase(prediction == 1, label == 1))
    return metric_list

if __name__ == '__main__':
    preprocess = Preprocesser("/data1/data/GD/raw", "/data1/data/GD/data")
    model = torch.load(args.model_path)
    model.cuda()
    image_list = os.listdir(os.path.join(args.root_path, "test_image"))
    image_list = [image for image in image_list if "image" in image]
    target_path = args.target_path
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    transform = transforms.CenterCrop(512)
    model.eval()
    metric_list = 0
    preformence = []

    for image_name in tqdm(image_list):
        image_ori = sitk.ReadImage(os.path.join(args.root_path, "test_image", image_name))
        label_ori = sitk.ReadImage(os.path.join(args.root_path, "test_image", image_name.replace("image", "lesion")))
        image_obj, label_obj, lung_obj = preprocess.Resample_and_Normalization(image_ori, label_ori)
        image = sitk.GetArrayFromImage(image_obj)
        label = sitk.GetArrayFromImage(label_obj)
        lung = sitk.GetArrayFromImage(lung_obj)
        if np.sum(label) == 0:
            continue
        transform_ = transforms.CenterCrop(label.shape[1])
        # result = np.zeros_like(label)
        result = []
        with torch.no_grad():
            for slice in image[:]:
                slice = torch.from_numpy(slice.copy()).unsqueeze(0)
                # lung_slice = torch.from_numpy(lung_slice.copy()).unsqueeze(0)
                slice = transform(slice)
                # lung_slice = transform(lung_slice)
                predict = torch.argmax(torch.softmax(
                    model(slice.unsqueeze(0).float().cuda()), dim=1), dim=1).squeeze(0)
                predict = transform_(predict).cpu().detach().numpy()
                # lung_slice = transform_(predict).cpu().detach().numpy()
                # predict[lung_slice == 0] = 0
                result.append(predict)
        result = np.asarray(result)
        result[lung == 0] = 0
        preformence.append(calculate_metric_percase(result, label))
        result_obj = sitk.GetImageFromArray(result)
        _, result, _ = preprocess.ResapmleImage2(image=image_obj, label=result_obj, lung=lung_obj, targetSpacing=image_ori.GetSpacing())
        sitk.WriteImage(result, os.path.join(target_path, image_name.replace("image", "predict")))
        # sitk.WriteImage(image_ori, os.path.join(target_path, image_name))
        # sitk.WriteImage(label_ori, os.path.join(target_path, image_name.replace("image", "lesion")))

    with open('RD_GD_50_preformence.txt', 'w') as f:
        for item in preformence:
            f.write("%s\n" % item)
    preformence = np.mean(preformence, 0)

    print("dice:{},  iou:{}, hd95:{}, asd:{}, jc:{}". format(preformence[0], preformence[1], preformence[2], preformence[3], preformence[4]))