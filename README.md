# Pytorch-YOLOv4

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=>=0.4&color=<COLOR>)

A minimal PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/


- [x] Inference
- [ ] Train
    - [x] Mocaic

# Weight

- baidu(https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA     Extraction code:dm5b)
- google(https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

![image](https://user-gold-cdn.xitu.io/2020/4/26/171b5a6c8b3bd513?w=768&h=576&f=jpeg&s=78882)

# Inference
- download model weight https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
```
python demo.py cfgfile weightfile imgfile
```

- [x] Convolution weight reshape
- [x] Mish activation
- [x] route number > 2
- [x] Maxpooling
- [ ] yololayer

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
