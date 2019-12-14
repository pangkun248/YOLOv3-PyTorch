from __future__ import division

from model import *
from util import *
from datasets import *

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

import colorsys
from PIL import Image, ImageFont, ImageDraw
import cv2

def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="test", help="测试图片文件夹")
    parser.add_argument("--model_def", type=str, default=r"E:\YOLOv3-PyTorch\yolov3_mobilenet.cfg",
                        help="yolov3网络配置文件")
    parser.add_argument("--weights_path", type=str, default="weights\wuhui\mobile_ep22-map92.03-loss0.18997.weights",
                        help="权重文件")
    parser.add_argument("--class_path", type=str, default="data\wuhui\dnf_classes.txt", help="标签文件路径")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="目标置信度")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="NMS中的iou阈值")
    parser.add_argument("--batch_size", type=int, default=1, help="每批batch多少张图片")
    parser.add_argument("--n_cpu", type=int, default=0, help="使用多少线程来生成数据")
    parser.add_argument("--img_size", type=int, default=320, help="网络输入尺寸")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("output", exist_ok=True)

    # 在GPU上加载模型
    model = Mainnet(opt.model_def).cuda()

    if opt.weights_path.endswith(".weights"):
        # 在模型上加载权重
        model.load_state_dict(torch.load(opt.weights_path))
    else:
        print('无检测模型')
    # 非训练阶段需要使用eval()模式
    model.eval()

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    # 加载类名
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    # 为每个类名配置不同的颜色
    hsv_tuples = [(x / len(classes), 1., 1.)for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))

    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    imgs = []  # 图片保存路径
    img_detections = []  # 每张图片的检测结果

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = input_imgs.type(FloatTensor)

        start_time = time.time()
        with torch.no_grad():
            detections = model(input_imgs)
            detections = NMS(detections, opt.conf_thres, opt.nms_thres)

        current_time = time.time()
        inference_time = current_time - prev_time
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s ms" % (batch_i, inference_time*1000/opt.batch_size))
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
            detections = xywh2xyxy(detections)
            # 先将xyxy相对坐标转换成max(600,800)下的坐标
            detections[:,:4] *= max(h, w)
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
        cv2.waitKey(300)
        # 保存测试结果
        filename = path.split("\\")[-1].split(".")[0]
        cv2.imwrite('D:\YOLOv3-PyTorch\output\{}.jpg'.format(filename),img)