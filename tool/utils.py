import sys
import os
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import itertools
import struct  # get_image_size
import imghdr  # get_image_size


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    
    # print('iou box1:', box1)
    # print('iou box2:', box2)

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes_back(boxes, cls_confs, det_confs, conf_thresh):
    
    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    # boxes = np.mean(boxes, axis=2, keepdims=False)

    t1 = time.time()
    all_boxes = []
    for b in range(boxes.shape[0]):
        l_boxes = []
        for i in range(boxes.shape[1]):
 
            det_conf = det_confs[b, i]
            max_cls_conf = cls_confs[b, i].max(axis=0)
            max_cls_id= cls_confs[b, i].argmax(axis=0)

            if det_conf > conf_thresh:
                bcx = boxes[b, i, 0]
                bcy = boxes[b, i, 1]
                bw = boxes[b, i, 2] - bcx
                bh = boxes[b, i, 3] - bcy

                l_box = [bcx, bcy, bw, bh, det_conf, max_cls_conf, max_cls_id]

                l_boxes.append(l_box)
        all_boxes.append(l_boxes)
    t2 = time.time()

    if False:
        print('---------------------------------')
        print('      boxes: %f' % (t2 - t1))
        print('---------------------------------')

    return all_boxes


def get_region_boxes(boxes, cls_confs, det_confs, conf_thresh):
    
    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    t1 = time.time()
    all_boxes = []
    for b in range(boxes.shape[0]):
        l_boxes = []
        # Shape: [batch, num_anchors * H * W] -> [num_anchors * H * W]
        # print(det_confs.shape)
        det_conf = det_confs[b, :]
        # print(det_conf.shape)
        argwhere = np.argwhere(det_conf > conf_thresh)
 
        max_cls_conf = cls_confs[b, argwhere].max(axis=2).flatten()
        max_cls_id = cls_confs[b, argwhere].argmax(axis=2).flatten()

        bcx = boxes[b, argwhere, 0]
        bcy = boxes[b, argwhere, 1]
        bw = boxes[b, argwhere, 2] - bcx
        bh = boxes[b, argwhere, 3] - bcy

        for i in range(bcx.shape[0]):
            # print(max_cls_conf[i])
            l_box = [bcx[i], bcy[i], bw[i], bh[i], det_conf[i], max_cls_conf[i], max_cls_id[i]]
            l_boxes.append(l_box)

        all_boxes.append(l_boxes)
    t2 = time.time()

    if False:
        print('---------------------------------')
        print('      boxes: %f' % (t2 - t1))
        print('---------------------------------')
    
    
    return all_boxes


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = np.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    sortIds = np.argsort(det_confs)
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



def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def post_processing(img, conf_thresh, n_classes, nms_thresh, output):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    boxes = []  
    t1 = time.time()
    for i in range(len(output)):
        boxes.append(get_region_boxes(output[i][0], output[i][1], output[i][2], conf_thresh))
    t2 = time.time()
    '''
    for i in range(3):
        masked_anchors = []
        for m in anchor_masks[i]:
            masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]
        masked_anchors = [anchor / strides[i] for anchor in masked_anchors]
        boxes.append(get_region_boxes_out_model(output[i], conf_thresh, 80, masked_anchors, len(anchor_masks[i])))
    '''
    if img.shape[0] > 1:
        bboxs_for_imgs = [
            boxes[0][index] + boxes[1][index] + boxes[2][index]
            for index in range(img.shape[0])]
        # 分别对每一张图片的结果进行nms
        t3 = time.time()
        boxes = [nms(bboxs, nms_thresh) for bboxs in bboxs_for_imgs]
    else:
        boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
        t3 = time.time()
        boxes = nms(boxes, nms_thresh)
    t4 = time.time()

    print('-----------------------------------')
    print('     get_region_boxes : %f' % (t2 - t1))
    print('                  nms : %f' % (t4 - t3))
    print('   post process total : %f' % (t4 - t1))
    print('-----------------------------------')
    return boxes
