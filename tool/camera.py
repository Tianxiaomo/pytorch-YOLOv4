# -*- coding: utf-8 -*-
'''
@Time          : 2020/04/26 15:48
@Author        : Tianxiaomo
@File          : camera.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
from __future__ import division
import cv2
from tool.darknet2pytorch import Darknet
import argparse
from tool.utils import *
from tool.torch_utils import *


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    cfgfile = "cfg/yolov4.cfg"
    weightsfile = "weight/yolov4.weights"

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    CUDA = torch.cuda.is_available()
    num_classes = 80
    bbox_attrs = 5 + num_classes
    class_names = load_class_names("data/coco.names")

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    if CUDA:
        model.cuda()

    model.eval()
    cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            sized = cv2.resize(frame, (model.width, model.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            boxes = do_detect(model, sized, 0.5, 0.4, CUDA)

            orig_im = plot_boxes_cv2(frame, boxes, class_names=class_names)

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            break
