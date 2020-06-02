import sys
import os
import argparse
import numpy as np
import cv2
from PIL import Image
import common
import tensorrt as trt

from tool.utils import *

TRT_LOGGER = trt.Logger()

def main(engine_path, image_path, image_size):
    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = common.allocate_buffers(engine)
        image_src = cv2.imread(image_path)

        detect(engine, context, buffers, image_src, image_size)


def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



def detect(engine, context, buffers, image_src, image_size):
    IN_IMAGE_H, IN_IMAGE_W = image_size

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    print("Shape of the network input: ", img_in.shape)
    # print(img_in)

    inputs, outputs, bindings, stream = buffers
    print('Length of inputs: ', len(inputs))
    inputs[0].host = img_in

    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print('Len of outputs: ', len(trt_outputs))
    # print(trt_outputs)
    '''
    print('Shape of outputs: ')
    print(trt_outputs[0].shape)
    print(trt_outputs[1].shape)
    print(trt_outputs[2].shape)

    trt_outputs[0] = trt_outputs[0].reshape(-1, 255, IN_IMAGE_H // 8, IN_IMAGE_W // 8)
    trt_outputs[1] = trt_outputs[1].reshape(-1, 255, IN_IMAGE_H // 16, IN_IMAGE_W // 16)
    trt_outputs[2] = trt_outputs[2].reshape(-1, 255, IN_IMAGE_H // 32, IN_IMAGE_W // 32)
    '''

    print('Shapes supposed to be: ')
    print(trt_outputs[0].shape)
    print(trt_outputs[1].shape)
    print(trt_outputs[2].shape)

    print(trt_outputs[3].shape)
    print(trt_outputs[4].shape)
    print(trt_outputs[5].shape)

    print(trt_outputs[6].shape)
    print(trt_outputs[7].shape)
    print(trt_outputs[8].shape)

    h1 = IN_IMAGE_H // 8
    w1 = IN_IMAGE_W // 8
    h2 = IN_IMAGE_H // 16
    w2 = IN_IMAGE_W // 16
    h3 = IN_IMAGE_H // 32
    w3 = IN_IMAGE_W // 32


    trt_outputs = [
        [
            trt_outputs[2].reshape(-1, 3 * h1 * w1, 4),
            trt_outputs[1].reshape(-1, 3 * h1 * w1, 80),
            trt_outputs[0].reshape(-1, 3 * h1 * w1)
        ],
        [
            trt_outputs[5].reshape(-1, 3 * h2 * w2, 4),
            trt_outputs[4].reshape(-1, 3 * h2 * w2, 80),
            trt_outputs[3].reshape(-1, 3 * h2 * w2)
        ],
        [
            trt_outputs[8].reshape(-1, 3 * h3 * w3, 4),
            trt_outputs[7].reshape(-1, 3 * h3 * w3, 80),
            trt_outputs[6].reshape(-1, 3 * h3 * w3)
        ]
    ]


    # print(outputs[2])
    num_classes = 80

    boxes = post_processing(img_in, 0.4, num_classes, 0.5, trt_outputs)

    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src, boxes, savename='predictions_trt.jpg', class_names=class_names)



if __name__ == '__main__':
    engine_path = sys.argv[1]
    image_path = sys.argv[2]
    
    if len(sys.argv) < 4:
        image_size = (416, 416)
    elif len(sys.argv) < 5:
        image_size = (int(sys.argv[3]), int(sys.argv[3]))
    else:
        image_size = (int(sys.argv[3]), int(sys.argv[4]))
    
    main(engine_path, image_path, image_size)
