from __future__ import division

from model import *
from util import *
from datasets import *
import time
import torch
from torch.utils.data import DataLoader
import colorsys
from PIL import Image, ImageFont, ImageDraw
import cv2


if __name__ == "__main__":
    map_name = 'wenyi'
    model_name = 'yolov3-mobileV2'
    import_param = {
        'batch_size': 1,
        'conf_thres': 0.8,
        'nms_thres': 0.4,
        'cfg_path': 'D:\py_pro\YOLOv3-PyTorch\yolo_cfg\\' + model_name + '.cfg',
        'weights_path': 'D:\py_pro\YOLOv3-PyTorch\weights\\' + map_name + '\\yolov3-mobileV2_ep60-map80.21-loss0.03100.weights',
        'class_path': 'D:\py_pro\YOLOv3-PyTorch\data\\' + map_name + '\dnf_classes.txt',
        'test_path': 'D:\py_pro\YOLOv3-PyTorch\\test\\',
    }
    print(import_param, '\n', "载入网络...")
    # 在GPU上加载模型
    model = Mainnet(import_param['cfg_path'])
    model.load_state_dict(torch.load(import_param['weights_path']))
    # 非训练阶段需要使用eval()模式
    model.eval()
    dataloader = DataLoader(
        ImageFolder(import_param['test_path'],
                    img_size=320),
                    batch_size=import_param['batch_size'],
                    shuffle=False,
                    num_workers=0,
    )
    # 加载类名
    classes = load_classes(import_param['class_path'])  # Extracts class labels from file
    # 为每个类名配置不同的颜色
    hsv_tuples = [(x / len(classes), 1., 1.)for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))

    imgs = []  # 图片保存路径
    img_detections = []  # 每张图片的检测结果
    print("\n准备开始检测:")
    cost_a = 0
    cost_b = 0
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = input_imgs.type(FloatTensor)
        with torch.no_grad():
            a = time.time()
            detections = model(input_imgs)
            b = time.time()
            detections = NMS(detections, import_param['conf_thres'], import_param['nms_thres'])
        if batch_i == 0:
            continue
        cost_a +=(time.time()-a)*1000
        cost_b +=(time.time()-b)*1000
        print(" Batch %d, 检测耗时: %.2fms NMS: %.2fms" % (batch_i, cost_a/batch_i, cost_b/batch_i))
        imgs.extend(img_paths)
        img_detections.extend(detections)

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        img = Image.open(path)
        w,h = img.size
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
        thickness = (img.size[0] + img.size[1]) // 300
        if detections is not None:
            # 在画图阶段需要转换一下坐标形式
            # detections = xywh2xyxy(detections)
            # 先将xyxy相对坐标转换成max(600,800)下的坐标
            detections[:,:4] *= max(h, w)/320
            # 如果h<w,则是一个宽边图,需要在y轴上减去(w - h) / 2,下同
            if h < w:
                detections[:,1:4:2] -= (w - h) / 2
            else:
                detections[:,0:3:2] -= (h - w) / 2
            # 随机取一个颜色
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                label = '{} {:.2f}'.format(classes[int(cls_pred)], cls_conf.item())
                draw = ImageDraw.Draw(img)
                # 获取文字区域的宽高
                label_w, label_h = draw.textsize(label, font)
                # 画出物体框 顺便加粗一些边框
                for i in range(thickness):
                    draw.rectangle([x1+i, y1+i, x2-i, y2-i],outline=colors[int(cls_pred)])
                # 画出label框
                draw.rectangle([x1, y1-label_h,x1+label_w,y1],fill=colors[int(cls_pred)])
                draw.text((x1, y1-label_h), label, fill=(0, 0, 0), font=font)
        img = np.array(img)[...,::-1]
        cv2.imshow('result',img)
        cv2.waitKey(0)
        # 保存测试结果,获取测试图片文件名
        # filename = path.split("\\")[-1].split(".")[0]
        # cv2.imwrite('D:\py_pro\YOLOv3-PyTorch\output\{}.jpg'.format(filename),img)