# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''

import torch
import argparse
from models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov4.cfg", help="path to model definition file")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Darknet(opt.model_def).to(device)
