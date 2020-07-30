import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from models import *

import sys
import cv2
from tool.utils import load_class_names, plot_boxes_cv2


n_classes = 80
cocoImageListFileName = "/home/erics/MS_COCO/val2017.txt"
cocoClassIDFileName = "/home/erics/yolo_cpp_standalone/data/categories.txt"
cocoClassNamesFileName = "/home/erics/yolo_cpp_standalone/data/coco.names"

model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
model.load_state_dict(pretrained_dict)


boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
