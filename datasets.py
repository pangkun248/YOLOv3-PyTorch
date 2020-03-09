import glob
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    # 该函数目的只是将img填充为边长为max(h,w)的正方形,resize的操作后面才执行
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # 为了下面pad做准备
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # 通过比较宽高大小,来生成不同的pad数据
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # 填充paddimg,
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=320, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, reso):
        # 该文件中存放了用于训练或者测试的.jpg图片的路径, 同时根据此路径可以得到对应的 labels 文件
        # list_path :txt文件 内容如下↓
        #       D:\py_pro\yolo3-pytorch\data\JPGImages\001548.jpg
        #       D:\py_pro\yolo3-pytorch\data\JPGImages\001561.jpg
        #       D:\py_pro\yolo3-pytorch\data\JPGImages\001571.jpg

        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        # 根据图片的路径得到 label 的路径, label 的存储格式为一个图片对应一个.txt文件
        # 文件的每一行代表了该图片的 box 信息, 其内容为: class_id, x, y, w, h (xywh都是用小数形式存储的,相对坐标)
        self.label_files = [path.replace('JPGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                            for path in self.img_files]
        self.reso = reso  # 获取图片目标大小, 之后会将图片放缩到此大小, 并相应调整box的数据
        self.batch_count = 0

    def __getitem__(self, index):
        # 根据index获取对应的图片路径,并去除右边的空格
        img_path = self.img_files[index].rstrip()
        # ToTensor这一步已经包含了归一化(1/255.0)
        img = transforms.ToTensor()(Image.open(img_path))

        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # 获取图片对应的 label 文件的路径
        label_path = self.label_files[index].rstrip()
        # 对 label 文件中的 真实box坐标按比例进行缩放到标准(reso*reso)尺寸中的相对坐标
        if os.path.exists(label_path):
            # labels -> cls_id,xmin,ymin,xmax,ymax 绝对坐标
            labels = torch.from_numpy((np.loadtxt(label_path).reshape(-1, 5)))
            # 根据padding 的大小,更新这些坐标的值,由于坐标更新的时候只需要和左或上部分的padding的xy坐标相加,所以这里只有pad[0]和pad[2]
            padded_x1 = labels[:, 1] + pad[0]
            padded_y1 = labels[:, 2] + pad[2]
            padded_x2 = labels[:, 3] + pad[0]
            padded_y2 = labels[:, 4] + pad[2]
            # 重新将坐标转化成小数模式(padding后的相对坐标),并且从xyxy转换成xywh
            # 从长远来看 转换成相对坐标要比绝对坐标对以后的计算要方便一些,因为YOLOv3有三种不同尺寸下的坐标计算,相对坐标直接乘以grid_size就行
            labels[:, 1] = ((padded_x1 + padded_x2) / 2) / padded_w
            labels[:, 2] = ((padded_y1 + padded_y2) / 2) / padded_h
            labels[:, 3] = (padded_x2 - padded_x1) / padded_w
            labels[:, 4] = (padded_y2 - padded_y1) / padded_h
        else:
            print(label_path,'路径不存在')
            exit()
        targets = torch.zeros((len(labels), 6))
        # cls_id x y w h    targets[:, 0]留着给img在batch中的索引使用
        targets[:, 1:] = labels
        # 如果图片中没有检测目标,则令target为 None
        return img_path, img, targets

    def __len__(self):
        return len(self.img_files)

    def collate_fn(self, batch):
        img_path, imgs, targets = list(zip(*batch))
        # 将img在batch中的索引添加进去
        for i, boxes in enumerate(targets):
            # 作者源代码中对于这里的处理有一些小问题,在图片中没有检测物体时会发生labels错位的情况会影响模型收敛
            # 详情见:https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/243
            # 不过此issue中也有一点小问题,就是将那些空数据过滤掉的不是通过判断boxes的id是不是None,而是下面的torch.cat()操作
            # 因为在PyTorch中 一个空张量的内存地址 和 None的内存地址 不相等,至少在PyTorch 1.3版本不相等
            # 不过可以通过boxes.shape[0] == 0 来判断这个boxes是否为空张量
            # if boxes is  None:
            #     print(i)
            #     continue
            boxes[:, 0] = i
        # 这一步操做并不能筛选掉那些无标注目标的target,因为他是一个空的张量不是None,而cat可以
        # targets = [boxes for boxes in targets if boxes is not None]
        targets = torch.cat(targets, 0)
        # 多尺度训练,频率每10轮随机改变输入尺寸
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        imgs = torch.stack([resize(img, self.reso) for img in imgs])
        self.batch_count += 1
        return img_path, imgs, targets
