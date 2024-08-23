import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from preprocess.preprocess import Preprocesser
import SimpleITK as sitk


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    # if pred.sum() > 0:
    dice = metric.binary.dc(pred, gt)
    return dice



def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    patch_size = image.shape
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, ...]
        x, y = slice.shape[0], slice.shape[1]

        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume2(image, label, net, classes):
    val_batchsize = 64
    # z, c, x, y = image.shape
    # patch_size = image.shape
    # image, label = image.squeeze(0).cpu().detach(
    # ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(0, image.shape[0], val_batchsize):
        if ind + val_batchsize<image.shape[0]:
            slice = image[ind:ind + val_batchsize, ...]
        else:
            slice = image[ind:, ...]
        if len(slice.shape)<4:
            input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
        else:
            input = torch.from_numpy(slice).float().cuda()
        # input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            # pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            if ind + val_batchsize < image.shape[0]:
                prediction[ind:ind + val_batchsize, ...] = out
            else:
                prediction[ind:, ...] = out
    prediction = np.asarray(prediction)
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume3(image, label, net, classes):
    image = image.cuda()
    net.eval()
    with torch.no_grad():
        output = torch.argmax(torch.softmax(
            net(image), dim=1), dim=1).squeeze(0)
        output = output.cpu().detach().numpy()
        # pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
    prediction = np.asarray(output)
    label = np.asarray(label)
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list



def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, ...]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_ds2(image, label, net, classes):
    val_batchsize = 64
    # image, label = image.squeeze(0).cpu().detach(
    # ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        if ind + val_batchsize<image.shape[0]:
            slice = image[ind:ind + val_batchsize, ...]
        else:
            slice = image[ind:, ...]
        # slice = image[ind, ...]
        # x, y = slice.shape[0], slice.shape[1]
        # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        if len(slice.shape)<4:
            input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
        else:
            input = torch.from_numpy(slice).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            # pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            # prediction[ind] = out
            if ind + val_batchsize < image.shape[0]:
                prediction[ind:ind + val_batchsize, ...] = out
            else:
                prediction[ind:, ...] = out
    prediction = np.asarray(prediction)
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
