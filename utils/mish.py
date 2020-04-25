# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 14:07
@Author        : huguanghao
@File          : mish.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import torch

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = x*(torch.tanh(torch.nn.functional.softplus(x)))
        return x