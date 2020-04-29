from model import *
from utils.util import *
from datasets import *
import os
import time
import torch
import colorsys
from PIL import Image, ImageFont, ImageDraw
import cv2


if __name__ == "__main__":
    model = YOLOv3(cfg.cfg_path).cuda()  # 在GPU上加载模型
    model.load_state_dict(torch.load(cfg.weights_path))
    model.eval()
    # 为每个类名配置不同的颜色
    hsv_tuples = [(x / len(cfg.class_name), 1., 1.) for x in range(len(cfg.class_name))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # 开始读取视频源或摄像头
    vid = cv2.VideoCapture(cfg.video_in)
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    while True:
        return_value, frame = vid.read()
        if return_value:
            # out = cv2.VideoWriter(import_param['video_out'], video_FourCC, video_fps, video_size)
            h, w, c = frame.shape
            PIL_img = Image.fromarray(frame[:, :, ::-1])
            tensor_img = transforms.ToTensor()(PIL_img)
            img, _ = pad_to_square(tensor_img, 0)
            # Resize并增加一维,因为网络输入尺寸要求batch_size,channel,height,width
            img = resize(img, (cfg.input_h,cfg.input_w)).cuda().unsqueeze(0)
            start_time = time.time()
            with torch.no_grad():
                detections = model(img)
                detections = NMS(detections, cfg.conf_thres, cfg.nms_thres)[0]
            end_time = time.time()
            # FPS计算方式比较简单
            fps = 'FPS:%.2f' % (1/(end_time-start_time))
            # 加载字体文件
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=16)
            draw = ImageDraw.Draw(PIL_img)
            if (detections is not None):
                # 先将在320*320标准下的xyxy坐标转换成max(600,800)下的坐标 再将x向或y向坐标减一下就行
                detections[:, :4] *= (max(h, w) / cfg.input_h)
                if max(h - w, 0) == 0:
                    detections[:, 1:4:2] -= (w - h) / 2
                else:
                    detections[:, 0:3:2] -= (h - w) / 2
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    print("\t+ Label: %s, Conf: %.5f" % (cfg.class_name[int(cls_pred)], cls_conf.item()))
                    label = '{} {:.2f}'.format(cfg.class_name[int(cls_pred)], cls_conf.item())
                    # 获取文字区域的宽高
                    label_w, label_h = draw.textsize(label, font)
                    # 画出物体框 顺便加粗一些边框
                    draw.rectangle([x1, y1, x2, y2], outline=colors[int(cls_pred)], width=3)
                    # 画出label框与label和分类概率
                    draw.rectangle([x1, y1 - label_h, x1 + label_w, y1], fill=colors[int(cls_pred)])
                    draw.text((x1, y1 - label_h), label, fill=(0, 0, 0), font=font)
            draw.text((1,1), fps, fill=colors[0], font=font)
            cv_img = np.array(PIL_img)[..., ::-1]
            cv2.imshow('result', cv_img)
            # cv2.waitKey(300)
            # out.write(cv_img)
            cv2.waitKey(1)
        else:
            break
