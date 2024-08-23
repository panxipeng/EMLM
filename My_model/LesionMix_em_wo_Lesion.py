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
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, Lesion_and_image_augmentation, None_augmentation, Image_augmentation
from dataloaders.brats2019 import TwoStreamBatchSampler
from networks.discriminator import FCDiscriminator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data1/data/LIDC/only_Lesion', help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='LIDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='model_baseline_10/LesionMix_em_wo_Lesionmix', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_epoch', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=512,
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def patients_to_slices(args):
    return int(len(os.listdir(args.root_path + "/train")) * (args.labeled_bs / args.batch_size))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

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

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    # train_transform = Image_augmentation(args.patch_size)
    # train_transform = Lesion_and_image_augmentation('/data1/data/LIDC/processed/lesion_warehouse',
    #                                                 labeled_prop=args.labeled_bs/args.batch_size, init_lesion_count=3,
    #                                                 argrate=1.0, racalss_flag=False, enhencement_strength="Weak")
    train_transform = Image_augmentation(args.patch_size)
    val_transform = None_augmentation(args.patch_size)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=train_transform)
    db_val = BaseDataSets(base_dir=args.root_path, split="val", transform=val_transform)

    total_slices = len(db_train)
    labeled_slice = int(total_slices * (args.labeled_bs / args.batch_size))
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args)
    # print("Total silices is: {}, labeled slices is: {}".format(
    #     total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=8, shuffle=False, num_workers=8)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = args.max_epoch
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # if (epoch_num + 1) == 10:
        #     trainloader.dataset.transform.Switch_aug(False, "Mix")
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # loss_ce = ce_loss(outputs[:args.labeled_bs],
            #                   label_batch[:][:args.labeled_bs].long().squeeze())
            # loss_dice = dice_loss(
            #     outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs])
            if args.labeled_bs != 1:
                loss_ce = ce_loss(outputs[:args.labeled_bs],
                                  label_batch[:args.labeled_bs][:].long().squeeze())
                loss_dice = dice_loss(
                    outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs])
            else:
                loss_ce = ce_loss(outputs[:args.labeled_bs],
                                  label_batch[:args.labeled_bs][:].long().squeeze(1))
                loss_dice = dice_loss(
                    outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs])
            supervised_loss = 0.5 * (loss_dice + loss_ce)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_loss = losses.entropy_loss(outputs_soft)
            if (epoch_num + 1) <= 10:
                loss = supervised_loss
            else:
                loss = supervised_loss + consistency_weight * consistency_loss
            # loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', scheduler.get_last_lr(), iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            scheduler.step()
            # if iter_num % 20 == 0:
            #     image = volume_batch[1, 0:1, :, :]
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(
            #         outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction',
            #                      outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

            # if iter_num > 0 and iter_num % 200 == 0:
        model.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(valloader):
            metric_i = test_single_volume_ds(
                sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(valloader)
        # for class_i in range(num_classes-1):
        #     writer.add_scalar('info/val_{}_dice'.format(class_i+1),
        #                       metric_list[class_i, 0], iter_num)
        #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
        #                       metric_list[class_i, 1], iter_num)
        writer.add_scalar('info/val_dice', metric_list[0], iter_num)

        performance = metric_list[0]

        # mean_hd95 = np.mean(metric_list, axis=0)[1]
        # writer.add_scalar('info/val_mean_dice', performance, iter_num)
        # writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

        if performance > best_performance:
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path,
                                          'epoch_{}_dice_{}.pth'.format(
                                              epoch_num, round(best_performance, 4)))
            save_best = os.path.join(snapshot_path,
                                     '{}_best_model.pth'.format(args.model))
            torch.save(model, save_mode_path)
            torch.save(model, save_best)

        logging.info(
            'epoch %d : mean_dice : %f' % (epoch_num, performance))
        model.train()

        # if iter_num % 3000 == 0:
        save_mode_path = os.path.join(
            snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(model, save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

        #     if iter_num >= max_iterations:
        #         break
        # if iter_num >= max_iterations:
        #     iterator.close()
        #     break
    writer.close()
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
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
