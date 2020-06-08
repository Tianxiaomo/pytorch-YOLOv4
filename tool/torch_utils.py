import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from tool import utils 


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    # print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def yolo_forward(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 1, 1, 1, 1, 1, num_classes, 1, 1, 1, 1, 1, num_classes, 1, 1, 1, 1, 1, num_classes ]
    #
    list_of_slices = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        for j in range(5):
            list_of_slices.append(output[:, begin + j : begin + j + 1])

        list_of_slices.append(output[:, begin + 5 : end])

    # Apply sigmoid(), exp() and softmax() to slices
    # [ 1, 1,     1, 1,     1,      num_classes, 1, 1, 1, 1, 1, num_classes, 1, 1, 1, 1, 1, num_classes ]
    #   sigmid()  exp()  sigmoid()  softmax()
    for i in range(num_anchors):
        begin = i * (5 + 1)

        # print(list_of_slices[begin].size())
        
        list_of_slices[begin] = torch.sigmoid(list_of_slices[begin])
        list_of_slices[begin + 1] = torch.sigmoid(list_of_slices[begin + 1])
        
        list_of_slices[begin + 2] = torch.exp(list_of_slices[begin + 2])
        list_of_slices[begin + 3] = torch.exp(list_of_slices[begin + 3])

        list_of_slices[begin + 4] = torch.sigmoid(list_of_slices[begin + 4])

        list_of_slices[begin + 5] = torch.nn.Softmax(dim=1)(list_of_slices[begin + 5])

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    batch = output.size(0)
    H = output.size(2)
    W = output.size(3)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0), axis=0)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    # Apply C-x, C-y, P-w, P-h to slices
    for i in range(num_anchors):
        begin = i * (5 + 1)
        
        list_of_slices[begin] += torch.tensor(grid_x, device=device)
        list_of_slices[begin + 1] += torch.tensor(grid_y, device=device)
        
        list_of_slices[begin + 2] *= anchor_w[i]
        list_of_slices[begin + 3] *= anchor_h[i]


    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + 1)

        xmin_list.append(list_of_slices[begin])
        ymin_list.append(list_of_slices[begin + 1])
        xmax_list.append(list_of_slices[begin] + list_of_slices[begin + 2])
        ymax_list.append(list_of_slices[begin + 1] + list_of_slices[begin + 3])

        # Shape: [batch, 1, H, W]
        det_confs = list_of_slices[begin + 4]

        # Shape: [batch, num_classes, H, W]
        cls_confs = list_of_slices[begin + 5]

        det_confs_list.append(det_confs)
        cls_confs_list.append(cls_confs)
    
    # Shape: [batch, num_anchors, H, W]
    xmin = torch.cat(xmin_list, dim=1)
    ymin = torch.cat(ymin_list, dim=1)
    xmax = torch.cat(xmax_list, dim=1)
    ymax = torch.cat(ymax_list, dim=1)

    # normalize coordinates to [0, 1]
    xmin = xmin / W
    ymin = ymin / H
    xmax = xmax / W
    ymax = ymax / H

    # Shape: [batch, num_anchors * H * W] 
    det_confs = torch.cat(det_confs_list, dim=1).view(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors, num_classes, H * W] 
    cls_confs = torch.cat(cls_confs_list, dim=1).view(batch, num_anchors, num_classes, H * W)
    # Shape: [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)

    # Shape: [batch, num_anchors * H * W, 1]
    xmin = xmin.view(batch, num_anchors * H * W, 1)
    ymin = ymin.view(batch, num_anchors * H * W, 1)
    xmax = xmax.view(batch, num_anchors * H * W, 1)
    ymax = ymax.view(batch, num_anchors * H * W, 1)

    # Shape: [batch, num_anchors * h * w, 4]
    boxes = torch.cat((xmin, ymin, xmax, ymax), dim=2).clamp(-10.0, 10.0)

    # Shape: [batch, num_anchors * h * w, num_classes, 4]
    # boxes = boxes.view(N, num_anchors * H * W, 1, 4).expand(N, num_anchors * H * W, num_classes, 4)
    

    return  boxes, cls_confs, det_confs



def do_detect(model, img, conf_thresh, n_classes, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    
    t1 = time.time()

    boxes_and_confs = model(img)

    # print(boxes_and_confs)
    output = []
    
    for i in range(len(boxes_and_confs)):
        output.append([])
        output[-1].append(boxes_and_confs[i][0].cpu().detach().numpy())
        output[-1].append(boxes_and_confs[i][1].cpu().detach().numpy())
        output[-1].append(boxes_and_confs[i][2].cpu().detach().numpy())

    t2 = time.time()

    print('-----------------------------------')
    print('          Preprocess : %f' % (t2 - t1))
    print('     Model Inference : %f' % (t1 - t0))
    print('-----------------------------------')

    '''
    for i in range(len(boxes_and_confs)):
        output.append(boxes_and_confs[i].cpu().detach().numpy())
    '''

    return utils.post_processing(img, conf_thresh, n_classes, nms_thresh, output)

