from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import *


def parse_cfg(cfgfile):
    # 加载文件并过滤掉文本中多余内容
    file = open(cfgfile, 'r')
    # 按行读取 相当于readlines
    lines = file.read().split('\n')
    # 去掉空行
    lines = [x for x in lines if len(x) > 0]
    # 去掉以#开头的注释行
    lines = [x for x in lines if x[0] != '#']
    # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # 这是cfg文件中一个层(块)的开始
            if len(block) != 0:  # 如果块内已经存了信息, 说明是上一个块的信息还没有保存
                blocks.append(block)  # 那么这个块（字典）加入到blocks列表中去
                block = {}  # 覆盖掉已存储的block,新建一个空白块存储描述下一个块的信息(block是字典)
            block["type"] = line[1:-1].rstrip()  # 把cfg的[]中的块名作为键type的值
        else:
            key, value = line.split("=")  # 按等号分割
            block[key.rstrip()] = value.lstrip()  # 边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对
    blocks.append(block)  # 退出循环，将最后一个未加入的block加进去
    return blocks


def create_modules(blocks):
    # blocks[0] [net]层 相当于超参数,网络全局配置的相关参数
    net_info = blocks[0]
    module_list = nn.ModuleList()
    # 因为图片就是三通道的,所以初始输入维数为3,
    prev_filters = 3
    output_filters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # 卷积层
        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # BN层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # leaky_relu层
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # 上采样层
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            # 将yolov3.cfg中upsample的stride填入到nn.Upsample中去
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            # 然后将上采样添加到小模型中去
            module.add_module("upsample_{}".format(index), upsample)

        # route层就是一个路由层
        elif (x["type"] == "route"):
            # 将当前layers层转换成list形式
            x["layers"] = x["layers"].split(',')
            # 将当前layers层中的数字转成整型
            start = int(x["layers"][0])
            # 接下来就要开始进行判断了,route层有两种情况,要么只有一个值和要么只有两个值的
            # 尝试获取layers层的第二个值并将其赋给end,如果没有第二个值就让end为0
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # 因为route层中的要求是负数(其实我觉得给出指定层数也是ok的,没必要负数。不过入乡随俗了)
            # 如果layers第一个值大于0,则让它减去当前层的index,则start就为负数了
            if start > 0:
                start = start - index
            # 如果layers层有两个值,并且第二个值大于0,那么也让它减去当前层的index,则end也就为负数了
            if end > 0:
                end = end - index
            # 将一个EmptyLayer()添加到小模型中
            module.add_module("route_{0}".format(index), EmptyLayer())
            # 这种情况就是layers有两个值,然后将指定层的维数相加起来
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            # 这种情况就是layers只有一个值,然后就直接输出那层的维数
            else:
                filters = output_filters[index + start]

        # 提前创建一个空的shortcut层并将其添加到一个小模型中
        elif x["type"] == "shortcut":
            module.add_module("shortcut_{}".format(index), EmptyLayer())

        # yolo是最终的检测识别层
        elif x["type"] == "yolo":
            # 将mask转换成list形式
            mask = x["mask"].split(",")
            # 将mask列表中的数字转换成整型
            mask = [int(x) for x in mask]
            # 将anchors转换成list形式
            anchors = x["anchors"].split(",")
            # 将anchors列表中的数字转换成整型
            anchors = [int(a) for a in anchors]
            # 将anchors中的18个anchor按两个一组分成九组
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            # 将当前mask对应的anchor赋值成当前yolo层的负责预测的anchors
            anchors = [anchors[i] for i in mask]
            num_classes = int(x["classes"])
            inp_dim = int(net_info["height"])
            # 将creat_model类中的YOLOLayer赋值给yolo_layer
            yolo_layer = YOLOLayer(anchors, num_classes, inp_dim)
            # 将detection添加到当前的小模型中
            module.add_module("YOLOLayer_{}".format(index), yolo_layer)
        # 将module一个小模型层添加到总模型中 具体想看结构的话可以打印看看
        module_list.append(module)
        # 每结束一层,将当前层的输出维数赋值给下一层的输入维数
        prev_filters = filters
        # 将每个卷积核的数量按 从前往后 顺序写入
        output_filters.append(filters)
        # 返回模型参数及模型结构

    return net_info, module_list


# 先创建一个空层 给route和shortcut层准备的,具体功能在Darknet类的forward中
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, inp_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.reso = inp_dim

    def forward(self, prediction, targets=None):
        # prediction.shape  -> batch_size,num_anchors*(self.num_classes + 5),grid_size,grid_size
        batch_size = prediction.size(0)
        grid_size = prediction.size(2)
        stride = self.reso / grid_size
        FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
        prediction = prediction.reshape(batch_size, self.num_anchors, (self.num_classes + 5), grid_size, grid_size)
        # 变换后的prediction.shape  -> batch_size, num_anchors, grid_size, grid_size, self.num_classes + 5
        prediction = prediction.permute(0, 1, 3, 4, 2)

        # 由于最终xywh都会在以stride为单位的featuremap上预测计算,所以这里anchors也要跟着改变
        scaled_anchors = FloatTensor([(a[0] / stride, a[1] / stride) for a in self.anchors])
        anchor_w = scaled_anchors[:, 0].reshape(1, self.num_anchors, 1, 1)
        anchor_h = scaled_anchors[:, 1].reshape(1, self.num_anchors, 1, 1)

        # 以下六个变量是要单独拿出来计算loss的,所以要单独拿出来
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # 这里的x_offset与y_offset只是表示每个grid的左上角坐标,方便后面相加
        x_offset = torch.arange(grid_size).repeat(grid_size, 1).reshape([1, 1, grid_size, grid_size]).type(FloatTensor)
        y_offset = torch.arange(grid_size).repeat(grid_size, 1).t().reshape([1, 1, grid_size, grid_size]).type(
            FloatTensor)

        # 这里为什要乘以压缩后的anchor(32倍)而不是原anchor的wh,因为pred_boxes中的xy坐标也都是在压缩32倍的环境下的.
        # 主要是为了保持一致,虽然马上就又恢复到正常大小了 (下面cat内容)
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + x_offset
        pred_boxes[..., 1] = y + y_offset
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        output = torch.cat(
            (
                pred_boxes.reshape(batch_size, -1, 4) * stride,
                pred_conf.reshape(batch_size, -1, 1),
                pred_cls.reshape(batch_size, -1, self.num_classes),
            ),
            -1,
        )
        # 如果是验证or测试的时候就到此为止了,直接返回预测的相关数据,否则返回loss进行更新梯度
        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=scaled_anchors,
                ignore_thres=self.ignore_thres,
                grid_size=grid_size,
            )
            # Loss : Mask layer_outputs to ignore non-existing objects (except with conf. loss)
            obj_mask = obj_mask.bool()
            noobj_mask = noobj_mask.bool()
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            # 含有目标的损失权重要大于没有目标的损失权重
            # 但是实际上这里的代码和 eriklindernoren / PyTorch-YOLOv3一样，它们在损失函数上的一些细节与yolo3论文中有些许不同
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            # 在target区域上正确预测含有目标且正确分类的概率 的 平均值 百分比
            cls_acc = 100 * class_mask[obj_mask].mean()
            # 在target区域上目标置信度的平均值
            conf_obj = pred_conf[obj_mask].mean()
            # 在target区域上无目标置信度的平均值
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask
            # 这里precision的定义为: (预测iou>0.5 * conf>0.5)且pre_box的种类与target_box一致 / 所有预测 conf>0.5的
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": loss,
                "x": loss_x.item(),
                "y": loss_y.item(),
                "w": loss_w.item(),
                "h": loss_h.item(),
                "conf": loss_conf.item(),
                "cls": loss_cls.item(),
                "cls_acc": cls_acc.item(),
                "recall_50": recall50.item(),
                "recall_75": recall75.item(),
                "precision": precision.item(),
                "conf_obj": conf_obj.item(),
                "conf_noobj": conf_noobj.item(),
                "grid_size": grid_size,
            }
            return output, loss


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        # 获取yolov3.cfg的文件配置信息
        self.blocks = parse_cfg(cfgfile)
        # 获取模型参数及模型结构
        self.net_info, self.module_list = create_modules(self.blocks)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        self.loss_names = ["total_loss", "x", "y", "w", "h", "conf", "cls", "recall", "precision", ]

    def forward(self, x, targets=None):
        # 获取前向传播的网络结构
        modules = self.blocks[1:]
        # 这个是为了route层结合而临时创建的,保存了每层网络输出的数据,共107层
        layer_outputs = {}
        yolo_outputs = []
        loss = 0
        for i, module in enumerate(modules):
            # 获取当前层的种类
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                # 开始输入数据, x就是处理好的batch_size张图片
                x = self.module_list[i](x)
            elif module_type == "route":
                # 获取route层中谁和谁连接,一般都是前几层和前一层相叠加
                layers = module["layers"]
                # 将其中的数字转为整型
                layers = [int(a) for a in layers]
                # 这里进行处理,由于格式需要,layers中的数字必须为负数
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                # 如果只有一个数字,则直接输出那层
                if len(layers) == 1:
                    x = layer_outputs[i + (layers[0])]
                # 如果有两个,则就是concat指定维度下相叠加
                else:
                    # 同上,需要对layers中的数字进行负处理。其实这里不进行负处理也是ok的,计算好指定cat层的index即可
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = layer_outputs[i + layers[0]]
                    map2 = layer_outputs[i + layers[1]]
                    # 两个特征相叠加
                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                # 从前几层开始跳跃
                from_ = int(module["from"])
                # 直接相加
                x = layer_outputs[i - 1] + layer_outputs[i + from_]
            elif module_type == 'yolo':
                # 此时x就是最后的特征图,三层yolo检测对应三种尺度的特征图    batch_size=4  num_class=16
                # x.shape  ->  [4, 507, 16]    [4, 2028,16]    [4, 8112,16]
                # 其实这个时候每张图片的各种预测数据就已经出来了,只是还需要经处理一下
                x, layer_loss = self.module_list[i][0](x, targets)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs[i] = x
        # 需要把三种尺度下的所有的anchors(3*(52*52+26*26+13*13))预测结果合并到一起.
        yolo_outputs = torch.cat(yolo_outputs, 1)

        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_weights(self, weights_path):
        # Open the weights file
        fp = open(weights_path, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
