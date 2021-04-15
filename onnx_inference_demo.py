# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : onnx_inference_demo.py
@Noice         : Infer with onnx weights and network structure.
@Modificattion :
    @Author    : Huxwell
    @Time      : 21/04/15 9:58
    @Detail    : Instead of darknet weights and cfg, use the onnx ones (i.e to test for fidelity)
'''
# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import onnxruntime
import argparse

"""hyper parameters"""
use_cuda = True

def detect_cv2(onnx, imgfile, namesfile):
    import cv2
    class_names = load_class_names(namesfile)

    session = onnxruntime.InferenceSession(onnx, None)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(input_name, input_shape)
    input_h = input_shape[2]
    input_w = input_shape[3]
    network_width, network_height = input_shape[3], input_shape[2]

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (network_width, network_height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    sized = sized.astype('float32') / 255.
    sized = sized.transpose(2, 0, 1)
    sized = sized.reshape(*input_shape)

    for i in range(2):
        start = time.time()
        boxes = do_detect_onnx(session, sized, 0.15, 0.2, use_cuda, input_name)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions_onnx_weights.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-onnx', type=str,
                        default='yolov4_1_3_416_416_static.onnx',
                        help='path to onnx file', dest='onnx')
    parser.add_argument('-imgfile', type=str,
                        default='/data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-class_names', type=str,
                        help='path to file containing class names')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    detect_cv2(args.onnx, args.imgfile, args.class_names)