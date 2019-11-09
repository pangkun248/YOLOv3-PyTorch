# PyTorch-YOLOv3 win10版本
这份代码是来自 [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3),我只不过对它进行了些许修改和注释.
而且我这份代码没有多尺度训练,我把它注释掉了.因为对于我而言没有用处(一些游戏人物的检测)

修改的地方:主要就是计算TP的时候,其他也无关痛痒.

代码环境:
win10 cuda10.0 cudnn7.6.1 python3.7 pytorch1.2

测试方式:首先需要把yolov3.weight下载到本目录,然后运行detect_img.py或者detect_video.py.py(视频or摄像头检测).

另:基本上代码中大部分代码都有注释.