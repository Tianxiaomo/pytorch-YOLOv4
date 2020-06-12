import sys
import os
import time
import math
import numpy as np

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
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
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



def get_region_boxes(boxes, confs, conf_thresh):

    print('Getting boxes from boxes and confs ...')
    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    # boxes: [batch, num_anchors * H * W, num_classes, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    # [batch, num_anchors * H * W, num_classes, 4] --> [batch, num_anchors * H * W, 4]
    boxes = boxes[:, :, 0, :]

    t1 = time.time()
    all_boxes = []
    for b in range(boxes.shape[0]):
        l_boxes = []

        # [num_anchors * H * W, num_classes] --> [num_anchors * H * W]
        max_conf = confs[b, :, :].max(axis=1)
        # [num_anchors * H * W, num_classes] --> [num_anchors * H * W]
        max_id = confs[b, :, :].argmax(axis=1)

        argwhere = np.argwhere(max_conf > conf_thresh)

        max_conf = max_conf[argwhere].flatten()
        max_id = max_id[argwhere].flatten()

        # print(max_conf)
        # print(max_id)

        bcx = boxes[b, argwhere, 0]
        bcy = boxes[b, argwhere, 1]
        bw = boxes[b, argwhere, 2]
        bh = boxes[b, argwhere, 3]

        # print(bcy)

        for i in range(bcx.shape[0]):
            # print(max_cls_conf[i])
            # print('bbox: [ {},\t{},\t{},\t{} ]'.format(bcx[i], bcy[i], bw[i], bh[i]))

            l_box = [bcx[i], bcy[i], bw[i], bh[i], max_conf[i], max_conf[i], max_id[i]]
            l_boxes.append(l_box)

        all_boxes.append(l_boxes)
    t2 = time.time()

    if False:
        print('---------------------------------')
        print('              boxes: %f' % (t2 - t1))
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
        boxes.append(get_region_boxes(output[i][0], output[i][1], conf_thresh))
    t2 = time.time()

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
