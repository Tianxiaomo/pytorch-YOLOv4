"""
A script to evaluate the model's performance using pre-trained weights using COCO API.
Example usage: python evaluate_on_coco.py -dir D:\cocoDataset\val2017\val2017 -gta D:\cocoDataset\annotatio
ns_trainval2017\annotations\instances_val2017.json -c cfg/yolov4-smaller-input.cfg -g 0
Explanation: set where your images can be found using -dir, then use -gta to point to the ground truth annotations file
and finally -c to point to the config file you want to use to load the network using.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from easydict import EasyDict as edict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cfg import Cfg
from tool.darknet2pytorch import Darknet
from tool.utils import load_class_names
from tool.torch_utils import do_detect


def get_class_name(cat):
    class_names = load_class_names("./data/coco.names")
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return class_names[cat]

def convert_cat_id_and_reorientate_bbox(single_annotation):
    cat = single_annotation['category_id']
    bbox = single_annotation['bbox']
    x, y, w, h = bbox
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    if 0 <= cat <= 10:
        cat = cat + 1
    elif 11 <= cat <= 23:
        cat = cat + 2
    elif 24 <= cat <= 25:
        cat = cat + 3
    elif 26 <= cat <= 39:
        cat = cat + 5
    elif 40 <= cat <= 59:
        cat = cat + 6
    elif cat == 60:
        cat = cat + 7
    elif cat == 61:
        cat = cat + 9
    elif 62 <= cat <= 72:
        cat = cat + 10
    elif 73 <= cat <= 79:
        cat = cat + 11
    single_annotation['category_id'] = cat
    single_annotation['bbox'] = [x1, y1, w, h]
    return single_annotation



def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()
    else:
        return obj

def evaluate_on_coco(cfg, resFile):
    annType = "bbox"  # specify type here
    with open(resFile, 'r') as f:
        unsorted_annotations = json.load(f)
    sorted_annotations = list(sorted(unsorted_annotations, key=lambda single_annotation: single_annotation["image_id"]))
    sorted_annotations = list(map(convert_cat_id_and_reorientate_bbox, sorted_annotations))
    reshaped_annotations = defaultdict(list)
    for annotation in sorted_annotations:
        reshaped_annotations[annotation['image_id']].append(annotation)

    with open('temp.json', 'w') as f:
        json.dump(sorted_annotations, f)

    cocoGt = COCO(cfg.gt_annotations_path)
    cocoDt = cocoGt.loadRes('temp.json')

    with open(cfg.gt_annotations_path, 'r') as f:
        gt_annotation_raw = json.load(f)
        gt_annotation_raw_images = gt_annotation_raw["images"]
        gt_annotation_raw_labels = gt_annotation_raw["annotations"]

    rgb_label = (255, 0, 0)
    rgb_pred = (0, 255, 0)

    for i, image_id in enumerate(reshaped_annotations):
        image_annotations = reshaped_annotations[image_id]
        gt_annotation_image_raw = list(filter(
            lambda image_json: image_json['id'] == image_id, gt_annotation_raw_images
        ))
        gt_annotation_labels_raw = list(filter(
            lambda label_json: label_json['image_id'] == image_id, gt_annotation_raw_labels
        ))
        if len(gt_annotation_image_raw) == 1:
            image_path = os.path.join(cfg.dataset_dir, gt_annotation_image_raw[0]["file_name"])
            actual_image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(actual_image)

            for annotation in image_annotations:
                x1_pred, y1_pred, w, h = annotation['bbox']
                x2_pred, y2_pred = x1_pred + w, y1_pred + h
                cls_id = annotation['category_id']
                label = get_class_name(cls_id)
                draw.text((x1_pred, y1_pred), label, fill=rgb_pred)
                draw.rectangle([x1_pred, y1_pred, x2_pred, y2_pred], outline=rgb_pred)
            for annotation in gt_annotation_labels_raw:
                x1_truth, y1_truth, w, h = annotation['bbox']
                x2_truth, y2_truth = x1_truth + w, y1_truth + h
                cls_id = annotation['category_id']
                label = get_class_name(cls_id)
                draw.text((x1_truth, y1_truth), label, fill=rgb_label)
                draw.rectangle([x1_truth, y1_truth, x2_truth, y2_truth], outline=rgb_label)
            actual_image.save("./data/outcome/predictions_{}".format(gt_annotation_image_raw[0]["file_name"]))
        else:
            print('please check')
            break
        if (i + 1) % 100 == 0: # just see first 100
            break

    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def test(model, annotations, cfg):
    if not annotations["images"]:
        print("Annotations do not have 'images' key")
        return
    images = annotations["images"]
    # images = images[:10]
    resFile = 'data/coco_val_outputs.json'

    if torch.cuda.is_available():
        use_cuda = 1
    else:
        use_cuda = 0

    # do one forward pass first to circumvent cold start
    throwaway_image = Image.open('data/dog.jpg').convert('RGB').resize((model.width, model.height))
    do_detect(model, throwaway_image, 0.5, 80, 0.4, use_cuda)
    boxes_json = []

    for i, image_annotation in enumerate(images):
        logging.info("currently on image: {}/{}".format(i + 1, len(images)))
        image_file_name = image_annotation["file_name"]
        image_id = image_annotation["id"]
        image_height = image_annotation["height"]
        image_width = image_annotation["width"]

        # open and resize each image first
        img = Image.open(os.path.join(cfg.dataset_dir, image_file_name)).convert('RGB')
        sized = img.resize((model.width, model.height))

        if use_cuda:
            model.cuda()

        start = time.time()
        boxes = do_detect(model, sized, 0.0, 80, 0.4, use_cuda)
        finish = time.time()
        if type(boxes) == list:
            for box in boxes:
                box_json = {}
                category_id = box[-1]
                score = box[-2]
                bbox_normalized = box[:4]
                box_json["category_id"] = int(category_id)
                box_json["image_id"] = int(image_id)
                bbox = []
                for i, bbox_coord in enumerate(bbox_normalized):
                    modified_bbox_coord = float(bbox_coord)
                    if i % 2:
                        modified_bbox_coord *= image_height
                    else:
                        modified_bbox_coord *= image_width
                    modified_bbox_coord = round(modified_bbox_coord, 2)
                    bbox.append(modified_bbox_coord)
                box_json["bbox_normalized"] = list(map(lambda x: round(float(x), 2), bbox_normalized))
                box_json["bbox"] = bbox
                box_json["score"] = round(float(score), 2)
                box_json["timing"] = float(finish - start)
                boxes_json.append(box_json)
                # print("see box_json: ", box_json)
                with open(resFile, 'w') as outfile:
                    json.dump(boxes_json, outfile, default=myconverter)
        else:
            print("warning: output from model after postprocessing is not a list, ignoring")
            return

        # namesfile = 'data/coco.names'
        # class_names = load_class_names(namesfile)
        # plot_boxes(img, boxes, 'data/outcome/predictions_{}.jpg'.format(image_id), class_names)

    with open(resFile, 'w') as outfile:
        json.dump(boxes_json, outfile, default=myconverter)

    evaluate_on_coco(cfg, resFile)


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Test model on test dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-gta', '--ground_truth_annotations', type=str, default='instances_val2017.json',
                        help='ground truth annotations file', dest='gt_annotations_path')
    parser.add_argument('-w', '--weights_file', type=str, default='weights/yolov4.weights',
                        help='weights file to load', dest='weights_file')
    parser.add_argument('-c', '--model_config', type=str, default='cfg/yolov4.cfg',
                        help='model config file to load', dest='model_config')
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Darknet(cfg.model_config)

    model.print_network()
    model.load_weights(cfg.weights_file)
    model.eval()  # set model away from training

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device=device)

    annotations_file_path = cfg.gt_annotations_path
    with open(annotations_file_path) as annotations_file:
        try:
            annotations = json.load(annotations_file)
        except:
            print("annotations file not a json")
            exit()
    test(model=model,
         annotations=annotations,
         cfg=cfg, )
