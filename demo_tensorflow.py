import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import cv2
from tool.utils import post_processing, load_class_names, plot_boxes_cv2


def demo_tensorflow(tfpb_file="./weight/yolov4.pb", image_path=None, print_sensor_name=False):
    graph_name = 'yolov4'
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as persisted_sess:
        print("loading graph...")
        with gfile.FastGFile(tfpb_file, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def, name=graph_name)

        # print all sensor_name
        if print_sensor_name:
            tensor_name_list = [tensor.name for tensor in tf.compat.v1.get_default_graph().as_graph_def().node]
            for tensor_name in tensor_name_list:
                print(tensor_name)

        inp = persisted_sess.graph.get_tensor_by_name(graph_name + '/' + 'input:0')
        print(inp.shape)
        out1 = persisted_sess.graph.get_tensor_by_name(graph_name + '/' + 'output_1:0')
        out2 = persisted_sess.graph.get_tensor_by_name(graph_name + '/' + 'output_2:0')
        out3 = persisted_sess.graph.get_tensor_by_name(graph_name + '/' + 'output_3:0')
        print(out1.shape, out2.shape, out3.shape)

        # image_src = np.random.rand(1, 3, 608, 608).astype(np.float32)  # input image
        # Input
        image_src = cv2.imread(image_path)
        resized = cv2.resize(image_src, (inp.shape[2], inp.shape[3]), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        print("Shape of the network input: ", img_in.shape)

        feed_dict = {inp: img_in}

        outputs = persisted_sess.run([out1, out2, out3], feed_dict)
        print(outputs[0].shape)
        print(outputs[1].shape)
        print(outputs[2].shape)

        boxes = post_processing(img_in, 0.4, outputs)

        num_classes = 80
        if num_classes == 20:
            namesfile = 'data/voc.names'
        elif num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'

        class_names = load_class_names(namesfile)
        result = plot_boxes_cv2(image_src, boxes, savename=None, class_names=class_names)
        cv2.imshow("tensorflow predicted", result)
        cv2.waitKey()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('weight/yolov4.pb')
        sys.argv.append('data/dog.jpg')
    if len(sys.argv) == 3:
        tfpbfile = sys.argv[1]
        image_path = sys.argv[2]
        demo_tensorflow(tfpbfile, image_path)
    else:
        print('Please execute this script this way:\n')
        print('  python demo_tensorflow.py <tfpbfile> <imageFile>')
