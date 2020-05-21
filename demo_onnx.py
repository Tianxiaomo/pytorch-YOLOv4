import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
from tool.utils import *


def main(onnx_path, image_path):
    session = onnxruntime.InferenceSession(onnx_path)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread(image_path)
    detect(session, image_src)



def detect(session, image_src):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name
    # output, output_exist = session.run(['decoder.output_conv', 'lane_exist.linear2'], {"input.1": image_np})

    # print(img_in)

    outputs = session.run(None, {input_name: img_in})

    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)

    # print(outputs[2])

    boxes = post_processing(img_in, 0.4, outputs)

    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src, boxes, savename='predictions_onnx.jpg', class_names=class_names)



if __name__ == '__main__':
    print("Warning: This demo only supports onnx model whose batchSize == 1")
    if len(sys.argv) == 3:
        onnx_path = sys.argv[1]
        image_path = sys.argv[2]
        main(onnx_path, image_path)
    else:
        print('Please execute this demo this way:\n')
        print('  python demo_onnx.py <onnxFile> <imageFile>')
