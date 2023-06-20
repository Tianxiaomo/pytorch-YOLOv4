import sys
import onnx
from onnx_tf.backend import prepare


# tensorflow >=2.0
# 1: Thanks:github:https://github.com/onnx/onnx-tensorflow
# 2: Run git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
#    Run pip install -e .
# Note:
#    Errors will occur when using "pip install onnx-tf", at least for me,
#    it is recommended to use source code installation
def transform_to_tensorflow(onnx_input_path, pb_output_path):
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('../weight/yolov4_1_3_608_608.onnx')  # use:darknet2onnx.py
        sys.argv.append('../weight/yolov4.pb')  # use:onnx2tensorflow.py
    if len(sys.argv) == 3:
        onnxfile = sys.argv[1]
        tfpb_outfile = sys.argv[2]
        transform_to_tensorflow(onnxfile, tfpb_outfile)
    else:
        print('Please execute this script this way:\n')
        print('  python onnx2tensorflow.py <onnxfile> <tfpboutfile>')
