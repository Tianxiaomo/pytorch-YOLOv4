# This should be run in JetPack 4.4 / JetPack 4.4 G.A. with DeepStream 5.0 / DeepStream 5.0 GA .

1. Compile the custom plugin for Yolo
2. Convert the ONNX file to TRT with TRTEXEC / TensorRT
3. Change the model-engine-file in config_infer_primary_yoloV4.txt
4. In the deepstream_app_config_yoloV4.txt, change 
          a) source0 : uri=file:<your file> directory. 
          b) primary-gie : model-engine-file=<your_onnx_engine>
# Note that for multi-batch, overhead is large owing to NMS is not used.
