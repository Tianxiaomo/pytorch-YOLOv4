# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import logging
import os, sys
from tqdm import tqdm
from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
import argparse
from easydict import EasyDict as edict

import numpy as np


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3):
        super(Yolo_loss, self).__init__()
        strides = [32, 16, 8]
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        # ANCH_MASK: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anch_mask = [6, 7, 8]
        self.ignore_thre = 0.5
        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        self.stride = strides[0]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output in xin:
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes
            dtype = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
                output[..., np.r_[:2, 4:n_ch]])

            # calculate pred - xywh obj cls

            x_shift = dtype(np.broadcast_to(
                np.arange(fsize, dtype=np.float32), output.shape[:4]))
            y_shift = dtype(np.broadcast_to(
                np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

            masked_anchors = np.array(self.masked_anchors)

            w_anchors = dtype(np.broadcast_to(np.reshape(
                masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
            h_anchors = dtype(np.broadcast_to(np.reshape(
                masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

            pred = output.clone()
            pred[..., 0] += x_shift
            pred[..., 1] += y_shift
            pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
            pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

            if labels is None:  # not training
                pred[..., :4] *= self.stride
                return pred.view(batchsize, -1, n_ch).data

            pred = pred[..., :4].data

            # target assignment

            tgt_mask = torch.zeros(batchsize, self.n_anchors,
                                   fsize, fsize, 4 + self.n_classes).type(dtype)
            obj_mask = torch.ones(batchsize, self.n_anchors,
                                  fsize, fsize).type(dtype)
            tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                    fsize, fsize, 2).type(dtype)

            target = torch.zeros(batchsize, self.n_anchors,
                                 fsize, fsize, n_ch).type(dtype)

            labels = labels.cpu().data
            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

            truth_x_all = labels[:, :, 0] // self.stride
            truth_y_all = labels[:, :, 1] // self.stride
            truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) // self.stride
            truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) // self.stride
            truth_i_all = truth_x_all.to(torch.int16).numpy()
            truth_j_all = truth_y_all.to(torch.int16).numpy()

            for b in range(batchsize):
                n = int(nlabel[b])
                if n == 0:
                    continue
                truth_box = dtype(np.zeros((n, 4)))
                truth_box[:n, 2] = truth_w_all[b, :n]
                truth_box[:n, 3] = truth_h_all[b, :n]
                truth_i = truth_i_all[b, :n]
                truth_j = truth_j_all[b, :n]

                # calculate iou between truth and reference anchors
                anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
                best_n_all = np.argmax(anchor_ious_all, axis=1)
                best_n = best_n_all % 3
                best_n_mask = ((best_n_all == self.anch_mask[0]) | (
                        best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))

                truth_box[:n, 0] = truth_x_all[b, :n]
                truth_box[:n, 1] = truth_y_all[b, :n]

                pred_ious = bboxes_iou(
                    pred[b].view(-1, 4), truth_box, xyxy=False)
                pred_best_iou, _ = pred_ious.max(dim=1)
                pred_best_iou = (pred_best_iou > self.ignore_thre)
                pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
                # set mask to zero (ignore) if pred matches truth
                obj_mask[b] = ~ pred_best_iou

                if sum(best_n_mask) == 0:
                    continue

                for ti in range(best_n.shape[0]):
                    if best_n_mask[ti] == 1:
                        i, j = truth_i[ti], truth_j[ti]
                        a = best_n[ti]
                        obj_mask[b, a, j, i] = 1
                        tgt_mask[b, a, j, i, :] = 1
                        target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                                                truth_x_all[b, ti].to(torch.int16).to(torch.float)
                        target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                                                truth_y_all[b, ti].to(torch.int16).to(torch.float)
                        target[b, a, j, i, 2] = torch.log(
                            truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                        target[b, a, j, i, 3] = torch.log(
                            truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                        target[b, a, j, i, 4] = 1
                        target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).numpy()] = 1
                        tgt_scale[b, a, j, i, :] = torch.sqrt(
                            2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

            # loss calculation

            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            bceloss = nn.BCELoss(weight=tgt_scale * tgt_scale,
                                 size_average=False)  # weighted BCEloss
            loss_xy += bceloss(output[..., :2], target[..., :2])
            loss_wh += self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
            loss_obj += self.bce_loss(output[..., 4], target[..., 4])
            loss_cls += self.bce_loss(output[..., 5:], target[..., 5:])
            loss_l2 += self.l2_loss(output, target)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=2, img_scale=0.5):
    train_dataset = Yolo_dataset(config.train_label, config)
    val_dataset = Yolo_dataset(config.val_label, config)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8,
                            pin_memory=True, drop_last=True)

    # writer = SummaryWriter(log_dir=cfg.TRAIN_TENSORBOARD_DIR,
    #                        comment=f'OPT_{cfg.TRAIN_OPTIMIZER}_LR_{cfg.lr}_BS_{batch_size}_SCALE_{img_scale}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = cfg.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {cfg.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Optimizer:       {cfg.TRAIN_OPTIMIZER}
    ''')

    # optimizer = optim.Adam([{'params': model.conv1.parameters(), 'lr': 0.2},
    #                         {'params': model.conv2.parameters(), 'lr': 0.2},
    #                         {'params': model, 'lr': 0.02},
    #                         {'params': model, 'lr': 0.3}
    #                         ])

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-08)

    criterion = Yolo_loss()
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc_every = 0
        epoch_acc_last = 0
        epoch_acc = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:

            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if i % config.subdivisions == 0:
                    optimizer.step()
                    model.zero_grad()

                if epoch_step % log_step == 0:
                    #     writer.add_scalar('Loss/train', loss.item(), global_step)
                    #     writer.add_scalar('acc_every/train', epoch_acc_every / (epoch_step * batch_size), global_step)
                    #     writer.add_scalar('acc/train', epoch_acc / (epoch_step * batch_size), global_step)
                    #     writer.add_scalar('acc_last/train', epoch_acc_last / (epoch_step * batch_size), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item()
                                        })

                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                pbar.update(images.shape[0])

            # logging.info('Validation Dice Coeff: {},{},{}'.format(val_acc, val_acc_every, val_acc_last))
            # lr = scheduler.step(val_acc_last)
            # # writer.add_scalar('lr', lr, global_step)
            # writer.add_scalar('Acc/test', val_acc, global_step)
            # writer.add_scalar('Acc_every/test', val_acc_every, global_step)
            # writer.add_scalar('Acc_last/test', val_acc_last, global_step)

            if save_cp:
                try:
                    os.mkdir(cfg.checkpoints)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(model.state_dict(),
                           cfg.checkpoints + f'wide_resnet-50-2_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    # writer.close()


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='pretrained/mobilenetv3-large-657e7b3d.pth',
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Yolov4()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # model = model.cuda()
    model.to(device=device)

    try:
        train(model=model,
              config=Cfg,
              epochs=cfg.TRAIN_EPOCHS,
              batch_size=cfg.batchsize,
              device=device, )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
