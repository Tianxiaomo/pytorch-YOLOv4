# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         : Save weights in PyTorch format.
@Modificattion :
    @Author    : Huxwell
    @Time      : 21/04/14 18:15
    @Detail    : 
'''
# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
use_cuda = True

def save_model_pytorch(cfgfile, weightfile, imgfile, output, namesfile=None):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    if namesfile is None:
        num_classes = m.num_classes
        if num_classes == 20:
            namesfile = 'data/voc.names'
        elif num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.15, 0.2, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions_while_saving_model.jpg', class_names=class_names)
    torch.save(m.state_dict(), output)
    print("Saved weights to ", output)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-output', type=str,
                        default='converted_from_darknet.pth',
                        help='path to save weights.', dest='output')
    parser.add_argument('-class_names', type=str, default=None,
                        help='path to file containing class names')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.imgfile:
        save_model_pytorch(args.cfgfile, args.weightfile, args.imgfile, args.output, args.class_names)