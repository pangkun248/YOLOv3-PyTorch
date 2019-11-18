# PyTorch-YOLOv3 win10版本
这份代码是来自 [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3),我只不过对它进行了些许修改和注释.
而且我这份代码没有多尺度训练,我把它注释掉了.因为对于我而言没有用处(一些游戏人物的检测)

修改的地方:主要就是计算TP的时候,其他也无关痛痒.

代码环境:
win10 cuda10.0 cudnn7.6.1 python3.7 pytorch1.2

数据文件和权重文件我把它放到百度云,需要的自取.[百度云](https://pan.baidu.com/s/1CG7zlJTAlDm-eImvQr0xTQ),提取码:fh3s。需要注意的是你得删除里面的train.txt和val.txt以及ImageSets文件夹中的train.txt和val.txt。然后你自己重新依次运行makeTxt.py和voc_annotation.py。

测试方式:首先需要把yolov3.weight下载到本目录,然后运行detect_img.py或者detect_video.py.py(视频or摄像头检测).

训练方式:运行train.py就行(前提是其中的参数得修改好).

另:基本上代码中大部分代码都有注释.以及你可能会遇到一些路径问题,自己对着数据文件路径修改即可。
