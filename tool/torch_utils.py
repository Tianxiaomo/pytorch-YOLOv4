import sys
import os
import time
import math
import torch
import numpy as np
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


def yolo_forward_alternative(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.size(0)
    H = output.size(2)
    W = output.size(3)

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()


    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0).reshape(1, 1, H * W).repeat(batch, 0).repeat(num_anchors, 1)
    grid_y = np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1).reshape(1, 1, H * W).repeat(batch, 0).repeat(num_anchors, 1)
    # Shape: [batch, num_anchors, H * W]
    grid_x_tensor = torch.tensor(grid_x, device=device, dtype=torch.float32)
    grid_y_tensor = torch.tensor(grid_y, device=device, dtype=torch.float32)

    anchor_array = np.array(anchors).reshape(1, num_anchors, 2)
    anchor_array = anchor_array.repeat(batch, 0)
    anchor_array = np.expand_dims(anchor_array, axis=3).repeat(H * W, 3)
    # Shape: [batch, num_anchors, 2, H * W]
    anchor_tensor = torch.tensor(anchor_array, device=device, dtype=torch.float32)

    # normalize coordinates to [0, 1]
    normal_array = np.array([1.0 / W, 1.0 / H, 1.0 / W, 1.0 / H], dtype=np.float32).reshape(1, 1, 4)
    normal_array = normal_array.repeat(batch, 0)
    normal_array = normal_array.repeat(num_anchors * H * W, 1)
    # Shape: [batch, num_anchors * H * W, 4]
    normal_tensor = torch.tensor(normal_array, device=device, dtype=torch.float32)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(batch, num_anchors, num_classes, H * W)
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.nn.Softmax(dim=2)(cls_confs)

    # Shape: [batch, num_anchors, 2, H * W]
    bxy = bxy.view(batch, num_anchors, 2, H * W)
    # Shape: [batch, num_anchors, 2, H * W]
    bwh = bwh.view(batch, num_anchors, 2, H * W)

    # Apply C-x, C-y, P-w, P-h
    bxy[:, :, 0] += grid_x_tensor
    bxy[:, :, 1] += grid_y_tensor

    print(anchor_tensor.size())
    bwh *= anchor_tensor

    # Shape: [batch, num_anchors, 4, H * W] --> [batch, num_anchors * H * W, 4]
    boxes = torch.cat((bxy, bwh), dim=2).permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, 4)

    print(normal_tensor.size())
    boxes *= normal_tensor


    return  boxes, cls_confs, det_confs



def yolo_forward(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.size(0)
    H = output.size(2)
    W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(batch, num_anchors, num_classes, H * W)
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.nn.Softmax(dim=2)(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0), axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii] + torch.tensor(grid_x, device=device, dtype=torch.float32) # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1] + torch.tensor(grid_y, device=device, dtype=torch.float32) # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)


    ########################################
    #   Figure out bboxes from slices     #
    ########################################
    
    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # normalize coordinates to [0, 1]
    bx = bx / W
    by = by / H
    bw = bw / W
    bh = bh / H

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx.view(batch, num_anchors * H * W, 1)
    by = by.view(batch, num_anchors * H * W, 1)
    bw = bw.view(batch, num_anchors * H * W, 1)
    bh = bh.view(batch, num_anchors * H * W, 1)

    # Shape: [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx, by, bw, bh), dim=2).view(batch, num_anchors * H * W, 1, 4)
    # Shape: [batch, num_anchors * h * w, num_classes, 4]
    boxes = boxes.repeat([1, 1, num_classes, 1])

    # boxes:     [batch, num_anchors * H * W, num_classes, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(batch, num_anchors * H * W, 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, num_classes, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return  boxes, confs




def do_detect(model, img, conf_thresh, n_classes, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
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

    t2 = time.time()

    print('-----------------------------------')
    print('          Preprocess : %f' % (t1 - t0))
    print('     Model Inference : %f' % (t2 - t1))
    print('-----------------------------------')

    '''
    for i in range(len(boxes_and_confs)):
        output.append(boxes_and_confs[i].cpu().detach().numpy())
    '''

    return utils.post_processing(img, conf_thresh, n_classes, nms_thresh, output)

