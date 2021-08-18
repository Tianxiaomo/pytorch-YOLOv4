# -*- coding: utf-8 -*-
# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
# from _typeshed import Self

from cv2 import data
from numpy.core.fromnumeric import resize, shape, sort
from torch import tensor
from vizer.draw import draw_boxes
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
from my_vehicle import Vehicle, sorts
import matplotlib.pyplot as plt
import pandas as pd
from sort import *
import copy
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from deep_sort_pytorch.deep_sort import build_tracker
from deep_sort_pytorch.utils import draw

"""hyper parameters"""
use_cuda = True

def cal_dist(x1,y1,x2,y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'cfg/obj.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
            
    # print(np.shape(boxes))
    # print(boxes[0][1])
    
    result = plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)
    cv2.imshow("result",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("data/0714test5.mp4")
    # cap = cv2.VideoCapture("data/Highway.mp4")

    frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # cap.set(3, 1080)
    # cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'cfg/obj.names'
    class_names = load_class_names(namesfile)
    
    downward_counter = [0,0,0,0,0]
    upward_counter = [0,0,0,0,0]
    memory = []
    
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")
    
    while True:
        process_all_start = time.time()
        ret, img = cap.read() #grab and retrieve the frame
        sized1 = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized1, cv2.COLOR_BGR2RGB) #get video frame
        height, width = sized.shape[:2]
        start = time.time()
        boxes = do_detect(m, sized, 0.7, 0.5, use_cuda)
        
        finish = time.time()
        print("Video_FPS : ",int(fps))
        print('Predicted in %f seconds.' % (finish - start))
        print("Predict_FPS : " ,int(1/(finish-start)))
        print("====================")
        # print(frames_count,fps,width,height)
        
        #depent on data
        up_linepos = 317
        line = [(190, 317), (900, 317)]
        result_img2 = cv2.line(img,(190,up_linepos),(900,up_linepos),(255,0,0),3,cv2.LINE_AA)
        # up_linepos = 500
        # line = [(160, up_linepos), (1600, up_linepos)]
        # result_img2 = cv2.line(img,(160,up_linepos),(1600,up_linepos),(255,0,0),3,cv2.LINE_AA)
        
        wh = np.flip(img.shape[0:2])
        result_img2 = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)
        
        # sort_test = []
        bbx_cxywh = []
        conf = []
        cls = []
        
        for class_index in range(len(boxes[0])):
            x1y1 = tuple((np.array(boxes[0][class_index][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[0][class_index][2:4]) * wh).astype(np.int32))
            x1, y1 = x1y1
            x2, y2 = x2y2
            w = abs(x2-x1)
            h = abs(y2-y1)
            classes = int(boxes[0][class_index][6])
            if x1 < 0:
                continue
            cx =int( (x2 + x1) / 2)
            cy = int((y2 + y1) / 2)
            
            result_img2 = cv2.circle(result_img2,((cx),(cy)),2,(0,0,255),cv2.FILLED)
            
            # sort_test.append((int(x1),int(y1),int(x2),int(y2),boxes[0][class_index][4],classes))
            
            bbx_cxywh.append([cx,cy,w,h])
            conf.append([boxes[0][class_index][4]])
            cls.append([classes])
            
            print("car class :",classes)
        print("====================")

        #     if len(past_vehicles):
        #         best_dist = np.inf
        #         for vehicle in past_vehicles:
        #             dist = vehicle.cal_dist(x1, y1)
        #             if dist < best_dist:
        #                 best_dist = dist
        #                 best_vehicles = vehicle
        #         if best_dist < 50:
        #             # vehicles.append(best_vehicles.update(cx, cy, x1y1, x2y2))
        #             vehicles.append(best_vehicles.update(x1,y1,x2,y2,boxes[0][class_index][4]))
        #             # print("pass1")
        #         else:
        #             vehicles.append(sorts(x1,y1,x2,y2,boxes[0][class_index][4]))
        #             # print("pass2")
        #     else:
        #         vehicles.append(sorts(x1,y1,x2,y2,boxes[0][class_index][4]))
        #         # print("pass3")
        
        # past_vehicles = vehicles.copy()
        
        #SORT
        # mot_tracker = Sort()
        # sort_test = np.array(sort_test)
        # track_bbx_ids = mot_tracker.update(sort_test)
        
        # # select person class
        # mask = classes == 0
        # bbox_xywh = bbox_xywh[mask]
        # bbx_cxywh[:, 3:] *= 1.2
        # # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
        # cls_conf = cls_conf[mask]
        
        #DEEP SORT
        mot_tracker = DeepSort('deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7')
        # mot_tracker = build_tracker("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",use_cuda = use_cuda)
        cls = np.array(cls)
        bbx_cxywh = np.asarray(bbx_cxywh)
        conf = np.array(conf)
        ###############################
        ##########DeepSort Start#########
        ###############################
        # track_bbx_ids = mot_tracker.update(bbx_cxywh,conf,img,cls)
        track_bbx_ids = mot_tracker.update(bbx_cxywh,conf,img,cls)

        # print(current_box)
        track_bbx_ids =np.array(track_bbx_ids)
        
        print("track_bbx_ids\n", track_bbx_ids)
        print("====================")
        
        current_box = []
        previous_box = copy.deepcopy(memory)
        indexIDs = []
        memory = []
        bbx = []
        
        #extract data from sort
        for track in track_bbx_ids:
            current_box.append([track[0], track[1],track[2], track[3],track[5]])
            indexIDs.append(int(track[4]))
            bbx.append([track[0], track[1],track[2], track[3]])
            memory.append([track[0], track[1],track[2], track[3],track[5]])
        current_box = np.array(current_box,int)
        bbx = np.array(bbx,int)
        print("current_box\n",current_box)
        print("====================")
        previous_box = np.array(previous_box,int)
        print("previous\n",previous_box)
        print("====================")
        
        result_img2 = draw.draw_boxes(img,bbx,indexIDs)
        
        #data processing # class and  counting 
        if len(current_box) > 0:
            i = int(0)
            for box in current_box:
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                for k in range(len(previous_box)):
                    if cal_dist(x,y,(previous_box[k][0]),(previous_box[k][1]))  < 50:
                        # print("same vehcile")
                        (x2, y2) = (int(previous_box[k][0]), int(previous_box[k][1]))
                        (w2, h2) = (int(previous_box[k][2]), int(previous_box[k][3]))
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                        cv2.line(result_img2, p0, p1, color, 3)
                        if (y2 + (h2-y2)/2) < up_linepos:
                            if intersect(p0, p1,line[0] ,line[1] ):
                                if box[4] == 0:
                                    downward_counter[0] +=1
                                elif box[4] == 1:
                                    downward_counter[1] += 1
                                elif box[4] == 2:
                                    downward_counter[2] += 1
                                elif box[4] == 3:
                                    downward_counter[3] += 1
                                elif box[4] == 4:
                                    downward_counter[4] += 1
                        else:
                            if intersect(p0, p1,line[0] ,line[1] ):
                                if box[4] == 0:
                                    upward_counter[0] +=1
                                elif box[4] == 1:
                                    upward_counter[1] += 1
                                elif box[4] == 2:
                                    upward_counter[2] += 1
                                elif box[4] == 3:
                                    upward_counter[3] += 1
                                elif box[4] == 4:
                                    upward_counter[4] += 1
        
        process_all_end = time.time()
        print("Process Time : ", (process_all_end-process_all_start))
        print("Process_all FPS : ", 1/(process_all_end-process_all_start))
                            
                # text = "{}".format(indexIDs[i])
                # cv2.putText(result_img2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # i += 1
        
    #出現車輛id        
        # for i in range(len(track_bbx_ids)):
        #     cv2.putText(result_img2,"id : "+str(track_bbx_ids[i,4]),(track_bbx_ids[i,2],track_bbx_ids[i,1]),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
        
        # 左邊文字背景框
        cv2.rectangle(result_img2, (0, 0), (160, 80), (85, 0, 0), -1)
        cv2.putText(result_img2, "Car : " + str(downward_counter[1]), (0, 30), cv2.FONT_HERSHEY_COMPLEX, .6, (170, 255, 50),1)
        cv2.putText(result_img2, "motorcycle : " + str(downward_counter[2]), (0, 50), cv2.FONT_HERSHEY_COMPLEX, .6, (50, 255, 50),1)
        cv2.putText(result_img2, "truck : " + str(downward_counter[4]), (0, 70), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 250, 150),1)
        # 右邊文字背景框
        cv2.rectangle(result_img2, (740, 0), (900, 80), (85, 0, 0), -1)
        cv2.putText(result_img2, "Car : " + str(upward_counter[1]), (740, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 255, 50),1)
        cv2.putText(result_img2, "motorcycle : " + str(upward_counter[2]), (740, 50), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 255, 50),1)
        cv2.putText(result_img2, "truck : " + str(upward_counter[4]), (740, 70), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 255, 50),1)
        
        cv2.imshow('Yolo demo', result_img2)
        print("=====The End of a frame=====")
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default=False,
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    if args.imgfile:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
