# Pytorch-YOLOv4

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

A minimal PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/


- [x] Inference
- [x] Train
    - [x] Mosaic

```
├── README.md
├── dataset.py            dataset
├── demo.py               demo to run pytorch --> tool/darknet2pytorch
├── demo_darknet2onnx.py  tool to convert into onnx --> tool/darknet2pytorch
├── demo_pytorch2onnx.py  tool to convert into onnx
├── models.py             model for pytorch
├── train.py              train models.py
├── cfg.py                cfg.py for train
├── cfg                   cfg --> darknet2pytorch
├── data            
├── weight                --> darknet2pytorch
├── tool
│   ├── camera.py           a demo camera
│   ├── coco_annotation.py       coco dataset generator
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```

![image](https://user-gold-cdn.xitu.io/2020/4/26/171b5a6c8b3bd513?w=768&h=576&f=jpeg&s=78882)

# 0. Weights Download

## 0.1 darknet
- baidu(https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA     Extraction code:dm5b)
- google(https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

## 0.2 pytorch
you can use darknet2pytorch to convert it yourself, or download my converted model.

- baidu
    - yolov4.pth(https://pan.baidu.com/s/1ZroDvoGScDgtE1ja_QqJVw Extraction code:xrq9) 
    - yolov4.conv.137.pth(https://pan.baidu.com/s/1ovBie4YyVQQoUrC3AY0joA Extraction code:kcel)
- google
    - yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
    - yolov4.conv.137.pth(https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)

# 1. Train

[use yolov4 to train your own data](Use_yolov4_to_train_your_own_data.md)

1. Download weight
2. Transform data

    For coco dataset,you can use tool/coco_annotation.py.
    ```
    # train.txt
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    ...
    ```
3. Train

    you can set parameters in cfg.py.
    ```
     python train.py -g [GPU_ID] -dir [Dataset direction] ...
    ```

# 2. Inference

## 2.1 Performance on MS COCO dataset (using pretrained DarknetWeights from <https://github.com/AlexeyAB/darknet>)

**ONNX and TensorRT models are converted from Pytorch (TianXiaomo): Pytorch->ONNX->TensorRT.**
See following sections for more details of conversions.

- val2017 dataset (input size: 416x416)

| Model type          | AP          | AP50        | AP75        |  APS        | APM         | APL         |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| DarkNet (YOLOv4 paper)|     0.471 |       0.710 |       0.510 |       0.278 |       0.525 |       0.636 |
| Pytorch (TianXiaomo)|       0.466 |       0.704 |       0.505 |       0.267 |       0.524 |       0.629 |
| TensorRT FP32 + BatchedNMSPlugin | 0.472| 0.708 |       0.511 |       0.273 |       0.530 |       0.637 |
| TensorRT FP16 + BatchedNMSPlugin | 0.472| 0.708 |       0.511 |       0.273 |       0.530 |       0.636 |

- testdev2017 dataset (input size: 416x416)

| Model type          | AP          | AP50        | AP75        |  APS        | APM         | APL         |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| DarkNet (YOLOv4 paper)|     0.412 |       0.628 |       0.443 |       0.204 |       0.444 |       0.560 |
| Pytorch (TianXiaomo)|       0.404 |       0.615 |       0.436 |       0.196 |       0.438 |       0.552 |
| TensorRT FP32 + BatchedNMSPlugin | 0.412| 0.625 |       0.445 |       0.200 |       0.446 |       0.564 |
| TensorRT FP16 + BatchedNMSPlugin | 0.412| 0.625 |       0.445 |       0.200 |       0.446 |       0.563 |


## 2.2 Image input size for inference

Image input size is NOT restricted in `320 * 320`, `416 * 416`, `512 * 512` and `608 * 608`.
You can adjust your input sizes for a different input ratio, for example: `320 * 608`.
Larger input size could help detect smaller targets, but may be slower and GPU memory exhausting.

```py
height = 320 + 96 * n, n in {0, 1, 2, 3, ...}
width  = 320 + 96 * m, m in {0, 1, 2, 3, ...}
```

## 2.3 **Different inference options**

- Load the pretrained darknet model and darknet weights to do the inference (image size is configured in cfg file already)

    ```sh
    python demo.py -cfgfile <cfgFile> -weightfile <weightFile> -imgfile <imgFile>
    ```

- Load pytorch weights (pth file) to do the inference

    ```sh
    python models.py <num_classes> <weightfile> <imgfile> <IN_IMAGE_H> <IN_IMAGE_W> <namefile(optional)>
    ```
    
- Load converted ONNX file to do inference (See section 3 and 4)

- Load converted TensorRT engine file to do inference (See section 5)

## 2.4 Inference output

There are 2 inference outputs.
- One is locations of bounding boxes, its shape is  `[batch, num_boxes, 1, 4]` which represents x1, y1, x2, y2 of each bounding box.
- The other one is scores of bounding boxes which is of shape `[batch, num_boxes, num_classes]` indicating scores of all classes for each bounding box.

Until now, still a small piece of post-processing including NMS is required. We are trying to minimize time and complexity of post-processing.


# 3. Darknet2ONNX

- **This script is to convert the official pretrained darknet model into ONNX**

- **Pytorch version Recommended:**

    - Pytorch 1.4.0 for TensorRT 7.0 and higher
    - Pytorch 1.5.0 and 1.6.0 for TensorRT 7.1.2 and higher

- **Install onnxruntime**

    ```sh
    pip install onnxruntime
    ```

- **Run python script to generate ONNX model and run the demo**

    ```sh
    python demo_darknet2onnx.py <cfgFile> <namesFile> <weightFile> <imageFile> <batchSize>
    ```

## 3.1 Dynamic or static batch size

- **Positive batch size will generate ONNX model of static batch size, otherwise, batch size will be dynamic**
    - Dynamic batch size will generate only one ONNX model
    - Static batch size will generate 2 ONNX models, one is for running the demo (batch_size=1)

# 4. Pytorch2ONNX

- **You can convert your trained pytorch model into ONNX using this script**

- **Pytorch version Recommended:**

    - Pytorch 1.4.0 for TensorRT 7.0 and higher
    - Pytorch 1.5.0 and 1.6.0 for TensorRT 7.1.2 and higher

- **Install onnxruntime**

    ```sh
    pip install onnxruntime
    ```

- **Run python script to generate ONNX model and run the demo**

    ```sh
    python demo_pytorch2onnx.py <weight_file> <image_path> <batch_size> <n_classes> <IN_IMAGE_H> <IN_IMAGE_W>
    ```

    For example:

    ```sh
    python demo_pytorch2onnx.py yolov4.pth dog.jpg 8 80 416 416
    ```

## 4.1 Dynamic or static batch size

- **Positive batch size will generate ONNX model of static batch size, otherwise, batch size will be dynamic**
    - Dynamic batch size will generate only one ONNX model
    - Static batch size will generate 2 ONNX models, one is for running the demo (batch_size=1)


# 5. ONNX2TensorRT

- **TensorRT version Recommended: 7.0, 7.1**

## 5.1 Convert from ONNX of static Batch size

- **Run the following command to convert YOLOv4 ONNX model into TensorRT engine**

    ```sh
    trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
    ```
    - Note: If you want to use int8 mode in conversion, extra int8 calibration is needed.

## 5.2 Convert from ONNX of dynamic Batch size

- **Run the following command to convert YOLOv4 ONNX model into TensorRT engine**

    ```sh
    trtexec --onnx=<onnx_file> \
    --minShapes=input:<shape_of_min_batch> --optShapes=input:<shape_of_opt_batch> --maxShapes=input:<shape_of_max_batch> \
    --workspace=<size_in_megabytes> --saveEngine=<engine_file> --fp16
    ```
- For example:

    ```sh
    trtexec --onnx=yolov4_-1_3_320_512_dynamic.onnx \
    --minShapes=input:1x3x320x512 --optShapes=input:4x3x320x512 --maxShapes=input:8x3x320x512 \
    --workspace=2048 --saveEngine=yolov4_-1_3_320_512_dynamic.engine --fp16
    ```

## 5.3 Run the demo

```sh
python demo_trt.py <tensorRT_engine_file> <input_image> <input_H> <input_W>
```

- This demo here only works when batchSize is dynamic (1 should be within dynamic range) or batchSize=1, but you can update this demo a little for other dynamic or static batch sizes.
    
- Note1: input_H and input_W should agree with the input size in the original ONNX file.
    
- Note2: extra NMS operations are needed for the tensorRT output. This demo uses python NMS code from `tool/utils.py`.


# 6. ONNX2Tensorflow

- **First:Conversion to ONNX**

    tensorflow >=2.0
    
    1: Thanks:github:https://github.com/onnx/onnx-tensorflow
    
    2: Run git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
    Run pip install -e .
    
    Note:Errors will occur when using "pip install onnx-tf", at least for me,it is recommended to use source code installation

# 7. ONNX2TensorRT and DeepStream Inference
  
  1. Compile the DeepStream Nvinfer Plugin 
  
  ```
      cd DeepStream
      make 
  ```
  2. Build a TRT Engine.
  
   For single batch, 
   ```
   trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
   ```
   
   For multi-batch, 
  ```
  trtexec --onnx=<onnx_file> --explicitBatch --shapes=input:Xx3xHxW --optShapes=input:Xx3xHxW --maxShapes=input:Xx3xHxW --minShape=input:1x3xHxW --saveEngine=<tensorRT_engine_file> --fp16
  ```
  
  Note :The maxShapes could not be larger than model original shape.
  
  3. Write the deepstream config file for the TRT Engine.
  
  
   
Reference:
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3

```
@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```
