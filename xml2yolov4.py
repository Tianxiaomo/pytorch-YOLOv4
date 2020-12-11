# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:05:40 2020

@author: Liu qi
"""


from xml.dom.minidom import parse
import xml.dom.minidom
import os
import shutil

def get_file_path(file_path):
    '''
    :param filename:修改后路径
    :param file_path: 图片所在label文件夹目录
    :param h,w: 图片的长宽
    :return:图片的路径列表
    '''
    img_paths = []
    img_names = os.listdir(file_path)
    for img_name in img_names:
        img_path = os.path.join(file_path, img_name)
        img_paths.append(img_path)
    return img_paths
def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))
    return os.path.join(*path)
path = r'D:\bonc\DPcase\yolov4_yolov5\nonmotor\new_data_1_label'
mkdir_if_not_exist(['./images/train/'])
xml_paths = get_file_path(path)

label_dict  = {"bicycle":0,
               "express_bicycle":1,
               "express_tricycle":2,
               "tricycle":3,
               "electric_bicycle":4,}
f = open('train_1.txt','w+')

for xml_path in xml_paths:
    domTree = parse(xml_path)
    rootNode = domTree.documentElement
    #img_name
    filename = rootNode.getElementsByTagName("filename")
    img_name = filename[0].childNodes[0].data
    #move image
    #shutil.copy('D:/bonc/DPcase/non-motor vehicle/new_data_1/'+img_name,'../data/'+img_name)
    
    rm = True
    xmlobjects = rootNode.getElementsByTagName("object")
    for xmlobject in xmlobjects:
        obname = xmlobject.getElementsByTagName("name")
        label_name = obname[0].childNodes[0].data
        if label_name in label_dict:
            if rm:
                f.write('\n')
                f.write('../nonmotor/images/train/'+img_name)
            rm =False
            label = label_dict[label_name]
            
            obbox = xmlobject.getElementsByTagName("bndbox")
            
            obxmin = obbox[0].getElementsByTagName("xmin")
            xmin = obxmin[0].childNodes[0].data
            
            obymin = obbox[0].getElementsByTagName("ymin")
            ymin = obymin[0].childNodes[0].data
            
            obxmax = obbox[0].getElementsByTagName("xmax")
            xmax = obxmax[0].childNodes[0].data
            
            obymax = obbox[0].getElementsByTagName("ymax")
            ymax = obymax[0].childNodes[0].data
            
            
            bbox_info=" %d,%d,%d,%d,%d" % (int(xmin), int(ymin), int(xmax), int(ymax), label)
            f.write(bbox_info)
f.close()