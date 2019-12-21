# PyTorch-YOLOv3 win10版本
这份代码是来自 [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3),我只不过对它进行了些许修改和注释.

修改的地方:见代码,只要我更改了一般都会在代码中有注释以及我为什么这样做.不过改动的地方不多

### 代码环境:
win10 cuda10.1 cudnn7.6.1 python3.7 pytorch1.3 GPU:1660 CPU:i5-8400

数据文件和权重文件我把它放到百度云,需要的自取.[百度云](https://pan.baidu.com/s/1CG7zlJTAlDm-eImvQr0xTQ),提取码:fh3s

### 训练和测试
测试方式:然后运行detect_img.py或者detect_video.py.py(视频or摄像头检测).
测试mAP的话直接运行test.py即可

训练方式:运行train.py就行(前提是其中的参数得修改好).

如果想在其他模型基础上继续训练的话pretrained改为True,以及得填好模型路径.否则的话pretrained改为False,这样的话模型会随机生成权重

### 目前已实现的YOLOv3及其变体
- [x] YOLOv3
- [x] YOLOv3-Mobilenet
- [x] YOLOv3-MobilenetV2
- [x] YOLOv3-Tiny
- [x] YOLOv3-Lite
- [x] YOLOv3-Tiny-MobileNet

### 已有功能

训练(有precision,recall指标) 验证(有precision,recall,AP,mAP指标) 并且可以可视化训练loss和mAP(使用Visdom)

## 模型表现 

| 模型名称 | 输入尺寸| GPU | CPU |
| ----- | ------ |  ----- | ----- |
| YOLOv3 | 320x320 |  29ms | 29ms |
| YOLOv3-Mobilenet | 320x320 | unkwon | unkwon |
| YOLOv3-MobilenetV2 | 320x320 | unkwon | unkwon |
| YOLOv3-Tiny | 320x320 | 7.7ms | 103ms|
| YOLOv3-Lite | 320x320 | 5.6ms | 73ms|
|YOLOv3-Tiny-Mobilenet | 320x320 | unkwon | unkwon |
###### 注: 各个模型的性能指标我暂时没有时间去训练测试,不过网上应该可以找得到。我这里只是暂时罗列出他们的速度对比

### 将要做的事情

1.当遇到一整个Batch图片都没有目标时的应对方法

2.很多地方的代码还需要精简或者修改.


