yolov4的发布引起了不少的关注，但由于darknet是大佬c语言写的，对代码的阅读有诸多不变，所以周末的时候写了个pytorch版的(蹭一波热度)。虽然pytorch——yolov4写好已经有一段时间了，但是由于种种原因一直没有进行验证(主要就是懒)，大家提出了诸多问题帮助修复很多bug，还有大佬一起增加新的功能，感谢大家的帮助。这些天呼声最高的就是如何如何使用自己的数据进行训练，昨天又是周末，就把这件拖了很久的事做了。并不像使用很多数据，于是自己制作了一个简单的数据集。

# 1. 代码准备

github 克隆代码
```
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
```
# 2. 数据准备

准备train.txt,内容是图片名和box。格式如下

```
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
```
- image_path : 图片名
- x1,y1 : 左上角坐标
- x2,y2 : 右下角坐标
- id : 物体类别

我自己用的数据是自己制作的了一个小数据集，检测各种各样的硬币(也就1元，5角，1角三种)，为什么不使用其他的东西制作数据集呢，没有啊，手边只有这些硬币感觉比较合适，相对其他的东西也比较简单。

![UTOOLS1590383513325.png](https://user-gold-cdn.xitu.io/2020/5/25/1724a3e953909b1b?w=1649&h=791&f=png&s=1290382)

一共准备了没几张。

# 3. 参数设置

开始训练的时候我直接用原来的参数，batch size设为64，跑了几个epoch发现不对，我数据一共才二十多个。后修改网络更新策略，不是按照每个epoch的step更新，使用总的steps更新，观察loss貌似可以训练了，于是睡觉，明天再看训练如何(鬼知道我又改了啥)。

今天打开电脑一看，what xx,loss收敛到2.e+4下不去了，此种必又蹊跷，遂kill了。于是把batch size直接设为4，可以正常训练了。

```
Cfg.batch = 4
Cfg.subdivisions = 1
```

# 4. 开始训练

```
 python train.py -l 0.001 -g 4 -pretrained ./yolov4.conv.137.pth -classes 3 -dir /home/OCR/coins

-l 学习率
-g gpu id
-pretrained 预训练的主干网络，从AlexeyAB给的darknet的yolov4.conv.137转换过来的
-classes 类别种类
-dir 图片所在文件夹
```


看下loss曲线
```
tensorboard --logdir log --host 192.168.212.75 --port 6008
```
![UTOOLS1590386319240.png](https://user-gold-cdn.xitu.io/2020/5/25/1724a696148d13f3?w=1357&h=795&f=png&s=151465)

# 5. 验证

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

效果差强人意(训练数据只有3种类型硬币)。

# 附

- coins数据集 （链接：https://pan.baidu.com/s/1y701NRKSdpj6UKDIH-GpqA 
提取码：j09s）
- yolov4.conv.137.pth (链接：https://pan.baidu.com/s/1ovBie4YyVQQoUrC3AY0joA 提取码：kcel)
