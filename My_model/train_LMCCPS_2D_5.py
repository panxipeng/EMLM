import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from contrastive_loss import ConLoss, contrastive_loss_sup
from utils.losses import ConLoss
from networks.projector import projectors, classifier
# from networks.pretrained_unet import preUnet
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders.dataset import newBaseDataSets,BaseDataSets, RandomGenerator, newLesion_and_image_augmentation, None_augmentation, Image_augmentation
from dataloaders.brats2019 import TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, metrics, ramps

# from val_2D import test_single_volume, test_single_volume_ds, test_single_volume2, test_single_volume_ds2, test_single_volume3

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data1/data/GD/only_Lesion',
                    help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='GD', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='model_baseline_10/LesionMix_Cross_Contrastive_Supervision12', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
# parser.add_argument('--max_iterations', type=int,
#                     default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epoch', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=512,
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
# parser.add_argument('--data_num', type=int, default=5200,
#                     help='labeled data')
# parser.add_argument('--labeled_num', type=int, default=5200 // 2,
#                     help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency1', type=float,
                    default=0.20, help='consistency')
parser.add_argument('--consistency2', type=float,
                    default=0.5, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=100.0, help='consistency_rampup')
args = parser.parse_args()


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


# def patients_to_slices(dataset, patiens_num):
#     ref_dict = None
#     if "ACDC" in dataset:
#         ref_dict = {"3": 68, "7": 136,
#                     "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
#     elif "Prostate":
#         ref_dict = {"2": 27, "4": 53, "8": 120,
#                     "12": 179, "16": 256, "21": 312, "42": 623}
#     else:
#         print("Error")
#     return ref_dict[str(patiens_num)]
def patients_to_slices(args):
    return int(len(os.listdir(args.root_path + "/train")) * (args.labeled_bs / args.batch_size))


def get_current_consistency_weight1(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency1 * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_current_consistency_weight2(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency2 * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def Cal_Dice(pred: np.ndarray, gt: np.ndarray):
    smooth = 1
    pred_f = pred.flatten()
    gt_f = gt.flatten()
    intersection = np.sum(pred_f * gt_f)
    return (2 * intersection) / (np.sum(pred_f) + np.sum(gt_f) + smooth)


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray):
    pred = np.array(pred, dtype=np.int16)
    gt = np.array(gt, dtype=np.int16)
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    if gt.sum() > 0:
        dice = Cal_Dice(pred, gt)
        return dice
    elif pred.sum() == 0:
        return 1
    else:
        return 0


def test_single_volume_ds(image, label, net, classes):
    image = image.type(torch.float32).cuda()
    # label = label.type(torch.float32).cuda()
    # image = image.float().cuda()
    with torch.no_grad():
        output = torch.argmax(torch.softmax(
            net(image), dim=1), dim=1).squeeze(0)
        output = output.cpu().detach().numpy()
    prediction = np.asarray(output)
    # prediction[prediction > 0] = 1
    label = np.asarray(label.squeeze(1))
    metric_list = []
    # for i in range(classes):
    #     metric_list.append(calculate_metric_percase(prediction == i, label == i))
    metric_list.append(calculate_metric_percase(prediction == 1, label == 1))
    return metric_list


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    # max_iterations = args.max_iterations

    loss_list1 = []
    dice_list1 = []

    loss_list2 = []
    dice_list2 = []

    projector_1 = projectors().cuda()
    projector_2 = projectors().cuda()

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()
    model1 = kaiming_normal_init_weight(model1)
    model2 = xavier_normal_init_weight(model2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # train_transform = Image_augmentation(args.patch_size)
    train_transform1 = newLesion_and_image_augmentation('/data1/data/GD/LesionMix_km/lesion_warehouse',
                                                        labeled_prop=args.labeled_bs / args.batch_size,
                                                        init_lesion_count=3,
                                                        argrate=0.3, racalss_flag=False)
    train_transform11 = newLesion_and_image_augmentation('/data1/data/GD/LesionMix_km/lesion_warehouse',
                                                        labeled_prop=args.labeled_bs / args.batch_size,
                                                        init_lesion_count=1,
                                                        argrate=0.9, racalss_flag=False)
    train_transform2 = Image_augmentation(args.patch_size)
    val_transform = None_augmentation(args.patch_size)

    db_train1 = newBaseDataSets(base_dir=args.root_path, split="train", transform=[train_transform2, train_transform1])
    db_train2 = newBaseDataSets(base_dir=args.root_path, split="train", transform=[train_transform2, train_transform11])
    db_val = BaseDataSets(base_dir=args.root_path, split="val", transform=val_transform)

    model1.train()
    model2.train()
    total_slices = len(db_train1)
    labeled_slice = int(total_slices * (args.labeled_bs / args.batch_size))
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    total_slices = len(db_train1)
    labeled_slice = patients_to_slices(args)
    # print("Total silices is: {}, labeled slices is: {}".format(
    #     total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader1 = DataLoader(db_train1, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader2 = DataLoader(db_train2, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=8, shuffle=False, num_workers=8)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    # optimizer1 = optim.Adam(model1.parameters(),lr=base_lr)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    # optimizer2 = optim.Adam(model2.parameters(), lr=base_lr)

    scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.9999)
    scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.9999)
    # scheduler2 = optim.lr_scheduler.LinearLR(optimizer2,start_factor=0.01, end_factor=0.00001,total_iters=100000)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    conloss = ConLoss()


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader1)))

    iter_num = 0
    max_epoch = args.max_epoch
    best_performance1 = 0.0
    best_performance2 = 0.0
    # iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in range(args.max_epoch):
        # with tqdm(total=len(trainloader), desc=f"epoch {epoch_num + 1}") as pbar:
        for i_batch, [sampled_batch1, sampled_batch2] in enumerate(zip(trainloader1, trainloader2)):
            if epoch_num + 1 < 30:
                volume_batch1, label_batch1 = sampled_batch1['image'].float(), sampled_batch1['label'].float()
                volume_batch1, label_batch1 = volume_batch1.cuda(), label_batch1.cuda()

                volume_batch2, label_batch2 = sampled_batch2['image'].float(), sampled_batch2['label'].float()
                volume_batch2, label_batch2 = volume_batch2.cuda(), label_batch2.cuda()
            else:
                volume_batch1, label_batch1 = sampled_batch1['ori_image'].float(), sampled_batch1['ori_label'].float()
                volume_batch1, label_batch1 = volume_batch1.cuda(), label_batch1.cuda()

                volume_batch2, label_batch2 = sampled_batch2['ori_image'].float(), sampled_batch2['ori_label'].float()
                volume_batch2, label_batch2 = volume_batch2.cuda(), label_batch2.cuda()

            outputs1_1 = model1(volume_batch1)
            outputs_soft1_1 = torch.softmax(outputs1_1, dim=1)
            feat1_1 = projector_1(outputs1_1)

            outputs2_2 = model2(volume_batch2)
            outputs_soft2_2 = torch.softmax(outputs2_2, dim=1)
            feat2_2 = projector_2(outputs2_2)

            with torch.no_grad():
                outputs2_1 = model2(volume_batch1)
                outputs_soft2_1 = torch.softmax(outputs2_1, dim=1)
                feat2_1 = projector_2(outputs2_1)

                outputs1_2 = model1(volume_batch2)
                outputs_soft1_2 = torch.softmax(outputs1_2, dim=1)
                feat1_2 = projector_1(outputs1_2)

            pseudo_outputs1 = torch.argmax(outputs_soft2_1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft1_2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            consistency_weight1 = get_current_consistency_weight1(iter_num // 150)
            consistency_weight2 = get_current_consistency_weight2(iter_num // 150)

            if args.labeled_bs != 1:
                loss1 = 0.5 * (ce_loss(outputs1_1[:args.labeled_bs],
                                       label_batch1[:args.labeled_bs].long().squeeze()) + dice_loss(
                    outputs_soft1_1[:args.labeled_bs], label_batch1[:args.labeled_bs]))
                loss2 = 0.5 * (ce_loss(outputs2_2[:args.labeled_bs],
                                       label_batch2[:args.labeled_bs].long().squeeze()) + dice_loss(
                    outputs_soft2_2[:args.labeled_bs], label_batch2[:args.labeled_bs]))
            else:
                loss1 = 0.5 * (ce_loss(outputs1_1[:args.labeled_bs],
                                       label_batch1[:args.labeled_bs].long().squeeze(1)) + dice_loss(
                    outputs_soft1_1[:args.labeled_bs], label_batch1[:args.labeled_bs]))
                loss2 = 0.5 * (ce_loss(outputs2_2[:args.labeled_bs],
                                       label_batch2[:args.labeled_bs].long().squeeze(1)) + dice_loss(
                    outputs_soft2_2[:args.labeled_bs], label_batch2[:args.labeled_bs]))

            pseudo_supervision1 = ce_loss(outputs1_1[args.labeled_bs:], pseudo_outputs1)
            pseudo_supervision2 = ce_loss(outputs2_2[args.labeled_bs:], pseudo_outputs2)

            # Loss_contrast1 = conloss(feat1_1, feat2_1.detach())
            # Loss_contrast2 = conloss(feat1_1.detach(), feat2_1)

            Loss_contrast1 = conloss(feat1_1, feat2_1)
            Loss_contrast2 = conloss(feat2_2, feat1_2)

            # Ent_loss1 = losses.entropy_loss(feat1_1)
            # Ent_loss2 = losses.entropy_loss(outputs2_2)

            model1_loss = loss1 + consistency_weight1 * pseudo_supervision1 + consistency_weight1 * Loss_contrast1
            model2_loss = loss2 + consistency_weight1 * pseudo_supervision2 + consistency_weight1 * Loss_contrast2

            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            loss_list1.append(float(model1_loss.cpu().detach()))
            loss_list2.append(float(model2_loss.cpu().detach()))

            # writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight1, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model1_pseudo_loss',
                              pseudo_supervision1, iter_num)
            writer.add_scalar('loss/model1_contrast_loss',
                              Loss_contrast1, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('loss/model2_pseudo_loss',
                              pseudo_supervision2, iter_num)
            writer.add_scalar('loss/model2_contrast_loss',
                              Loss_contrast2, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f, model2 loss : %f' %
                (iter_num, model1_loss.item(), model2_loss.item()))

            scheduler1.step()
            scheduler2.step()

        model1.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(tqdm(valloader)):
            metric_i = test_single_volume_ds(
                sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(valloader)

        writer.add_scalar('info/model1_val_dice', metric_list[0], epoch_num)
        performance1 = np.mean(metric_list, axis=0)
        dice_list1.append(performance1)

        if performance1 > best_performance1:
            best_performance1 = performance1
            save_mode_path = os.path.join(snapshot_path,
                                          'model1_iter_{}_dice_{}.pth'.format(
                                              epoch_num, round(best_performance1, 4)))
            save_best = os.path.join(snapshot_path,
                                     '{}_best_model1.pth'.format(args.model))
            torch.save(model1, save_mode_path)
            torch.save(model1, save_best)

        logging.info(
            'epoch %d : model1_mean_dice : %f model1_best_dice : %f' % (epoch_num, performance1, best_performance1))
        model1.train()

        model2.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(tqdm(valloader)):
            metric_i = test_single_volume_ds(
                sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(valloader)
        writer.add_scalar('info/model2_val_dice', metric_list[0], epoch_num)
        performance2 = np.mean(metric_list, axis=0)
        dice_list2.append(performance2)
        if performance2 > best_performance2:
            best_performance2 = performance2
            save_mode_path = os.path.join(snapshot_path,
                                          'model2_iter_{}_dice_{}.pth'.format(
                                              iter_num, round(best_performance2)))
            save_best = os.path.join(snapshot_path,
                                     '{}_best_model2.pth'.format(args.model))
            torch.save(model2, save_mode_path)
            torch.save(model2, save_best)

        logging.info(
            'epoch %d : model2_mean_dice : %f model2_best_dice : %f' % (epoch_num, performance2, best_performance2))
        model2.train()

        save_mode_path = os.path.join(
            snapshot_path, 'model1_epoch_' + str(epoch_num) + '.pth')
        torch.save(model1, save_mode_path)
        logging.info("save model1 to {}".format(save_mode_path))

        save_mode_path = os.path.join(
            snapshot_path, 'model2_epoch_' + str(epoch_num) + '.pth')
        torch.save(model2, save_mode_path)
        logging.info("save model2 to {}".format(save_mode_path))
    writer.close()

    with open('./MMS_1_3_loss1.txt', 'w') as f:
        for item in loss_list1:
            f.write("%s\n" % item)

    with open('./MMS_1_3_dice1.txt', 'w') as f:
        for item in dice_list1:
            f.write("%s\n" % item)

    with open('./MMS_1_3_loss2.txt', 'w') as f:
        for item in loss_list2:
            f.write("%s\n" % item)

    with open('./MMS_1_3_dice2.txt', 'w') as f:
        for item in dice_list2:
            f.write("%s\n" % item)
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../../model/{}/{}/{}".format(
        args.dataset, args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
