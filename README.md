# PyTorch-YOLOv3 win10版本
这份代码是来自 [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3),我只不过对它进行了些许修改和注释.
而且我这份代码没有多尺度训练,我把它注释掉了.因为对于我而言没有用处(一些游戏人物的检测)

修改的地方:见代码,只要我更改了一般都会在代码中有注释以及我为什么这样做.不过改动的地方不多

代码环境:
win10 cuda10.1 cudnn7.6.1 python3.7 pytorch1.3

数据文件和权重文件我把它放到百度云,需要的自取.[百度云](https://pan.baidu.com/s/1CG7zlJTAlDm-eImvQr0xTQ),提取码:fh3s

测试方式:首先需要把yolov3.weight下载到本目录,然后运行detect_img.py或者detect_video.py.py(视频or摄像头检测).
测试mAP的话直接运行test.py即可

训练方式:运行train.py就行(前提是其中的参数得修改好).

如果想继续训练的话pretrained改为True,以及args.weights得填好模型路径.否则的话pretrained改为False,这样的话模型会随机生成权重

如果你想尝试各种不同的YOLOv3,只要更改其中的模型配置文件名称即可,以及保存模型时的模型名称哦.


另:基本上代码中大部分代码都有注释.yolo_cfg文件夹中是各种YOLOv3的变体:

yolov3 -> YOLOv3

yolov3-m -> YOLOv3-MobileNet(注:我只是将主干网络替换成MobileNetV1,后面的部分还是照搬YOLOv3的卷积结构)

yolov3-m2 -> YOLOv3-MobileNetV2(注:我除了将主干网络换成MobileNetV2之外还将后面检测层所有的普通卷积都换成了V1中的卷积形式,
其实主要是用3*3的分组卷积+1*1的普通卷积对3*3的普通卷积方式进行了替换,当然每层yolo的最后一层卷积还是和YOLOv3一样)

yolov3-t -> YOLOv3-Tiny (这个只是一个精简版的YOLOv3,没什么特殊的)

yolov3-tm -> YOLOv3-Tiny-MobileNet(注:我将YOLOv3-Tiny的主干网络换成了MobileNetV1,但是结果好像不太理想)

暂时这份代码还有好几处需要修改的地方:

1.在测试图片阶段与原作者相比速度慢了1ms左右.

2.当遇到一整个Batch图片都没有目标时的应对方法

3.很多地方的代码还需要精简或者修改.

4.需要做的事情还有不少,但是我想按照上面的顺序一个一个来.

