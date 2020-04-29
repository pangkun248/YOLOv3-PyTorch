from model import *
from utils.util import *
from datasets import *
import time
import torch
from torch.utils.data import DataLoader
import colorsys
from PIL import Image, ImageFont, ImageDraw
import cv2


if __name__ == "__main__":
    model = YOLOv3(cfg.cfg_path).cuda()
    model.load_state_dict(torch.load(cfg.weights_path))
    model.eval()
    dataloader = DataLoader(ImageFolder(cfg.test_path,img_size=cfg.input_h),
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
    )
    # 为每个类名配置不同的颜色
    hsv_tuples = [(x / len(cfg.class_name), 1., 1.)for x in range(len(cfg.class_name))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))

    imgs = []  # 图片保存路径
    img_detections = []  # 每张图片的检测结果
    print("\n准备开始检测:")
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = input_imgs.type(FloatTensor)
        with torch.no_grad():
            a = time.time()
            detections = model(input_imgs)
            detections = NMS(detections, cfg.conf_thres, cfg.nms_thres)
        imgs.extend(img_paths)
        img_detections.extend(detections)

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        img = Image.open(path)
        w,h = img.size
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=16)
        if detections is not None:
            # 在画图阶段需要转换一下坐标形式
            # detections = xywh2xyxy(detections)
            # 先将xyxy相对坐标转换成max(600,800)下的坐标
            detections[:,:4] *= max(h, w)/cfg.input_h
            # 如果h<w,则是一个宽边图,需要在y轴上减去(w - h) / 2,下同
            if h < w:
                detections[:,1:4:2] -= (w - h) / 2
            else:
                detections[:,0:3:2] -= (h - w) / 2
            # 随机取一个颜色
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f x1:%d y1:%d x2:%d y2:%d" % (cfg.class_name[int(cls_pred)], cls_conf.item(),x1, y1, x2, y2))
                label = '{} {:.2f}'.format(cfg.class_name[int(cls_pred)], cls_conf.item())
                draw = ImageDraw.Draw(img)
                # 获取文字区域的宽高
                label_w, label_h = draw.textsize(label, font)
                # 画出物体框 顺便加粗一些边框
                draw.rectangle([x1, y1, x2, y2],outline=colors[int(cls_pred)],width=3)
                # 画出label框
                draw.rectangle([x1, y1-label_h,x1+label_w,y1],fill=colors[int(cls_pred)])
                draw.text((x1, y1-label_h), label, fill=(0, 0, 0), font=font)
        img = np.array(img)[...,::-1]
        cv2.imshow('result',img)
        cv2.waitKey(0)
        # 保存测试结果,获取测试图片文件名
        # filename = path.split("\\")[-1].split(".")[0]
        # cv2.imwrite('D:\py_pro\YOLOv3-PyTorch\output\{}.jpg'.format(filename),img)