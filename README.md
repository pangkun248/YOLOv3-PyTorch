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
| YOLOv3 | 320x320 |  31ms | 760ms |
| YOLOv3-Mobilenet | 320x320 | 20ms | 366ms |
| YOLOv3-MobilenetV2 | 320x320 | 29ms | 391ms |
| YOLOv3-Tiny | 320x320 | 9.5ms | 114ms|
| YOLOv3-Lite | 320x320 | 5.8ms | 77ms|
|YOLOv3-Tiny-Mobilenet | 320x320 | 14ms | 247ms |
###### 注: 不同的任务场景,不同模型表现往往不太一样.对于简单任务场景,往往简单模型要比复杂模型表现的要更好.
###### YOLOv3-MobilenetV2的网络结构有些特殊导致如果使用原生的V2代码则和YOLOv3消耗几乎一致的时间.
###### 如果想要更快可以参考网上其他版本的实现,把最后两个卷积层(输出维度为1280和320)砍掉或者其他方法
###### 不过感觉MobileNetV2相比较V1精度与速度以及占用资源等提升不是很大.

### 模型剪枝
参考 https://github.com/tanluren/yolov3-channel-and-layer-pruning 及其中列出的多个repo

目前来说,剪枝可达到80%-90%的剪枝率,而mAP掉2-3个点.

但是和每个任务场景及模型有很强的关联,也就说有些任务场景可以达到很高的剪枝率有些却比较低,模型也是同理.

https://github.com/Lam1360/YOLOv3-model-pruning 中提到的对于YOLOv3剪枝70% 速度翻倍的情况我没有遇到.问题1

emmm...我是复制了他提供的剪枝后的cfg文件以及权重文件然后测试模型forward时间.反而我的速度几乎没变,猜测和硬件平台以及运行环境有关.

意思就是说如果你只是普通玩家的话那速度基本没变,但你要是RMB玩家的话那提升速度就很可观了

不过这里还是提供一个大致的模型剪枝过程吧,以防以后用得上.

拿YOLOv3来说.剪枝方式就是跳过shortcut中起始末尾层、YOLO层前一层、upsamle前一层、maxpool前一层.

以上conv这些层中的卷积不参与剪枝.至于为什么不参与剪枝?问题2

1.shortcut层有残差连接,剪枝的话不好统一对应的通道,不过对于该种情况后续有解决办法.

2.YOLO前一层,因为该层的输出维度与num_cls有关,无法更改.即不参与剪枝,不过输入通道还是参与剪枝了的

3.upsample与maxpool前一层,这两种情况目前没有想到不参与剪枝的原因.只是别人这么做了.

目前来说,剪枝方式有无shortcut的普通剪枝,层剪枝.以及Slim论文中提到的剪枝方式,但是没有实现边训练边剪枝.

问题1及问题2的相关回答见 https://github.com/tanluren/yolov3-channel-and-layer-pruning/issues/54

####剪枝方式:

先开启稀疏化训练 import_param中 is_pruned参数 改为True,然后选择合适的pruned_id以及一个合适的稀疏因子. 运行train.py

训练之后在prune.py文件中设置一个剪枝率percent参数,最后对稀疏化训练后的模型进行剪枝运行相应pruned_id的xxx_prune.py即可