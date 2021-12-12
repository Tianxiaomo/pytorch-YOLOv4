The release of yolov4 has attracted a lot of attention, but because darknet is written in big brother c language, there are many unchanged reading of the code, so the weekend wrote a pytorch version (to rub a wave of heat). Although pytorch - yolov4 write good has been a while, but for a variety of reasons have not been validated (mainly lazy), people raised many questions to help fix many bugs, there are big brothers together to add new features, thank you for your help. These days the highest call is how to how to use their own data for training, and yesterday was the weekend, so the thing that has dragged on for a long time to do. It is not like using a lot of data, so I made a simple dataset myself


# 1. Code Preparation

github Cloning Code
```
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
```
# 2. Data Preparation

Prepare train.txt, which contains the image name and box in the following format

```
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
```
- image_path : Image Name
- x1,y1 : Coordinates of the upper left corner
- x2,y2 : Coordinates of the lower right corner
- id : Object Class

I use their own data is their own production of a small data set to detect a variety of coins (also 1 yuan, 50 cents, 10 cents three), why not use other things to produce data sets, no ah, only these coins on hand feel more appropriate, relatively simple compared to other things。

![UTOOLS1590383513325.png](https://user-gold-cdn.xitu.io/2020/5/25/1724a3e953909b1b?w=1649&h=791&f=png&s=1290382)

A total of a few prepared。

# 3. Parameter Setting

When I started training, I directly used the original parameters, batch size set to 64, ran a few epochs found that it is not right, my data is only a total of more than 20. After modifying the network update strategy, not in accordance with the step of each epoch update, using the total steps update, observe the loss seems to be able to train, so sleep, tomorrow to see how the training (the ghost knows what I changed)

Today, I opened my computer and saw that what xx,loss converged to 2.e+4, which must be strange again, so I killed it. So I set the batch size to 4 directly, and can train normally。

```
Cfg.batch = 4
Cfg.subdivisions = 1
```

# 4. Start training

```
 python train.py -l 0.001 -g 4 -pretrained ./yolov4.conv.137.pth -classes 3 -dir /home/OCR/coins

-l learning rate
-g gpu id
-pretrained Pre-trained backbone network, converted from yolov4.conv.137 of darknet given by AlexeyAB
-classes NO. of classes
-dir Training image dir
```


Look at the loss curve
```
tensorboard --logdir log --host 192.168.212.75 --port 6008
```
![UTOOLS1590386319240.png](https://user-gold-cdn.xitu.io/2020/5/25/1724a696148d13f3?w=1357&h=795&f=png&s=151465)

# 5. Inference

```
python model.py 3 weight/Yolov4_epoch166_coins.pth data/coin2.jpg data/coins.names

python model.py num_classes weightfile imagepath namefile
```
coins.names
```
1yuan
5jiao
1jiao

```

![UTOOLS1590386705468.png](https://user-gold-cdn.xitu.io/2020/5/25/1724a6f46e826bb8?w=774&h=1377&f=png&s=1191048)

The results were poor (only 3 types of coins were available for the training data).

# Attachment

- coins dataset (link: https://pan.baidu.com/s/1y701NRKSdpj6UKDIH-GpqA) 
(Extraction code: j09s)
- yolov4.conv.137.pth (Link: https://pan.baidu.com/s/1ovBie4YyVQQoUrC3AY0joA Extraction code: kcel)
