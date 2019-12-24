from __future__ import division

import torch.nn as nn
from util import *

import torch


def create_modules(blocks):
    # blocks[0] [net]层 相当于超参数,网络全局配置的相关参数
    net_info = blocks.pop(0)
    module_list = nn.ModuleList()
    # 因为图片就是三通道的,所以初始输入维数为3,
    prev_filters = 3
    output_filters = []
    for block in blocks:
        # 注意module.add_module("some_layer", some_layer)中的第一个参数不可以重复.即如果一个模块中有多个conv或者bn或者relu等
        # module.add_module("sth", sth)第一个参数一定要唯一.
        # 例 module.add_module("conv1", conv1)
        #    module.add_module("conv2", conv2)
        # 否则后添加的"some_layer"会覆盖前面的同名"some_layer".
        module = nn.Sequential()
        block_type = block["type"]
        # 卷积层
        if block_type == "convolutional":
            bn = int(block["batch_normalize"])
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            # 其实有这一步即使cfg文件中没有pad这个属性也不影响卷积操作,所以我把cfg中的pad属性删了
            pad = (kernel_size - 1) // 2
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=not bn)
            module.add_module("conv", conv)
            # BN层
            if bn:
                bn = nn.BatchNorm2d(filters)
                module.add_module("bn", bn)
            # leaky_relu层
            if block["activation"] == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky", activn)

        elif block_type == "convolutional_dw":
            filters = int(block["filters"])
            stride = int(block["stride"])
            # 深度卷积层
            conv_dw = nn.Conv2d(prev_filters, prev_filters, 3, stride, 1,groups= prev_filters, bias=False)
            module.add_module("conv_dw", conv_dw)
            # BN层
            conv_dw_bn = nn.BatchNorm2d(prev_filters)
            module.add_module("conv_dw_bn", conv_dw_bn)

            # leaky_relu层
            activn = nn.LeakyReLU(0.1, inplace=True)
            module.add_module("conv_dw_bn_leaky", activn)
            # 逐点卷积层
            conv_pw = nn.Conv2d(prev_filters, filters, 1, 1, 0, bias=False)
            module.add_module("conv_pw", conv_pw)
            conv_pw_bn = nn.BatchNorm2d(filters)
            module.add_module("conv_pw_bn", conv_pw_bn)
            activn = nn.LeakyReLU(0.1, inplace=True)
            module.add_module("conv_pw_bn_leaky", activn)
        elif block_type == "bottleneck":
            expand = int(block["expand"])
            filters = int(block["filters"])
            stride = int(block["stride"])
            # 为创建bottleneck层做准备
            pw1 = nn.Conv2d(prev_filters,filters*expand,kernel_size=1,bias=False)
            bn1 = nn.BatchNorm2d(filters*expand)
            dw = nn.Conv2d(filters*expand,filters*expand,kernel_size=3,stride=stride,padding=1,groups=filters*expand,bias=False)
            bn2 = nn.BatchNorm2d(filters*expand)
            pw2 = nn.Conv2d(filters*expand,filters,kernel_size=1,bias=False)
            bn3 = nn.BatchNorm2d(filters)
            leaky_relu = nn.LeakyReLU(0.1,inplace=True)

            # bottleneck层中的升维卷积
            module.add_module("bottleneck_expand_pw", pw1)
            module.add_module("bottleneck_expand_bn", bn1)
            module.add_module("bottleneck_expand_leaky", leaky_relu)

            # bottleneck层中的分组卷积
            module.add_module("bottleneck_dw", dw)
            module.add_module("bottleneck_bn", bn2)
            module.add_module("bottleneck_leaky", leaky_relu)

            # bottleneck层中的降维卷积
            module.add_module("bottleneck_reduce_pw", pw2)
            module.add_module("bottleneck_reduce_bn", bn3)
        # 上采样层
        elif block_type == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample", upsample)

        elif block_type == "maxpool":
            size = int(block["size"])
            stride = int(block["stride"])
            if size == 2 and stride == 1:
                module.add_module("padding", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(size, stride)
            module.add_module("maxpool", maxpool)

        # route层 直译过来是路由层,但是我觉得翻译成"融合层"比较合适(或者叠加层？)
        elif block_type == "route":
            layers = [int(x) for x in block["layers"].split(",")]
            filters = sum([output_filters[i] for i in layers])
            # 这里的EmptyLayer是为了下面的改变输出维度使用的
            module.add_module("route", EmptyLayer())

        elif block_type == "shortcut":
            filters = output_filters[int(block['from'])]
            module.add_module("shortcut", EmptyLayer())

        # yolo层是最终的检测识别层
        elif block_type == "yolo":
            # 将mask转换成list形式,并将列表中的数字转换成整型
            anchor_index = [int(x) for x in block["mask"].split(",")]
            # 同上
            anchors = [int(a) for a in block["anchors"].split(",")]
            # 将anchors中的18个anchor按两个一组分成九组
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            # 将当前mask对应的anchor赋值成当前yolo层的负责预测的anchors
            anchors = [anchors[i] for i in anchor_index]
            num_classes = int(block["classes"])
            inp_dim = int(net_info["height"])
            # 将creat_model类中的YOLOLayer赋值给yolo_layer
            yolo_layer = YOLOLayer(anchors, num_classes, inp_dim)
            module.add_module("yolo", yolo_layer)

        # 将module一个小模型层添加到总模型中 具体想看结构的话可以打印看看
        module_list.append(module)
        # 每结束一层,将当前层的输出维数赋值给下一层的输入维数
        prev_filters = filters
        # 将每个卷积核的数量按 从前往后 顺序写入
        output_filters.append(filters)
        # 返回模型参数及模型结构
    return net_info, module_list


# 先创建一个空层 给route和shortcut层准备的,具体功能在Mainnet类的forward中
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
        self.inp_dim = inp_dim
        self.grid_size = 0

    def forward(self, prediction, targets=None):
        # prediction.shape  -> batch_size,num_anchors*(self.num_classes + 5),grid_size,grid_size
        batch_size = prediction.size(0)
        grid_size = prediction.size(2)
        # reshape后的prediction.shape  -> batch_size,num_anchors,(self.num_classes + 5),grid_size,grid_size
        prediction = prediction.reshape(batch_size, self.num_anchors, (self.num_classes + 5), grid_size, grid_size)
        # permute后的prediction.shape  -> batch_size, num_anchors, grid_size, grid_size, (self.num_classes + 5)
        prediction = prediction.permute(0, 1, 3, 4, 2)
        # 由于最终的xywh都会在以stride为单位的featuremap上预测计算,所以这里anchors的尺寸也要跟着改变(缩小)
        # 以下六个变量是要单独拿出来计算loss的,所以要单独拿出来
        x = torch.sigmoid(prediction[..., 0]) # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        # 我这里使用的是softmax分类方法,和原作者sigmoid不同的是,我自己训练的数据集中没有狗,哈士奇或者人,女人这种类别从属现象发生
        # 我自己遇到的都是 飞机 汽车 大炮 人 狗...等等互斥的类别.所以在此进行了更改
        pred_cls = torch.softmax(prediction[..., 5:], dim=1)  # Cls pred.
        # 下面这些数据时和yolo层绑定在一起的,所以只要yolo层固定了这些数据也就固定了.不然每次都要临时创建需要额外耗费 5 ms
        if grid_size != self.grid_size:
            self.grid_size = grid_size
            g = self.grid_size
            # 图片从网络输入到YOLO层时缩小的倍数 标准YOLOv3有三个YOLO层,所以有三个stride 8 16 32
            self.stride = self.inp_dim / self.grid_size
            # 这里的x_offset与y_offset只是表示每个grid的左上角坐标,方便后面相加
            self.x_offset = torch.arange(g).repeat(g, 1).reshape([1, 1, g, g]).type(FloatTensor)
            self.y_offset = torch.arange(g).repeat(g, 1).t().reshape([1, 1, g, g]).type(FloatTensor)
            # 由于最终的xywh都会在以stride为单位的featuremap上预测计算,所以这里anchors的尺寸也要跟着改变(缩小)
            self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
            self.anchor_w = self.scaled_anchors[:, 0:1].reshape((1, self.num_anchors, 1, 1))
            self.anchor_h = self.scaled_anchors[:, 1:2].reshape((1, self.num_anchors, 1, 1))

        # 这里的x_offset与y_offset只是表示每个grid的左上角坐标,方便后面相加
        # x_offset = torch.arange(grid_size).repeat(grid_size, 1).reshape([1, 1, grid_size, grid_size]).type(FloatTensor)
        # y_offset = torch.arange(grid_size).repeat(grid_size, 1).t().reshape([1, 1, grid_size, grid_size]).type(FloatTensor)
        # 这里为什要乘以压缩(8,16,32)倍后的anchor而不是原anchor的wh,因为pred_boxes中的wh值也都是在压缩(8,16,32)倍的环境下预测出来的.
        # 主要是为了保持一致,虽然马上就又恢复到正常大小了 (下面cat内容)
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + self.x_offset
        pred_boxes[..., 1] = y + self.y_offset
        pred_boxes[..., 2] = torch.exp(w) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h) * self.anchor_h

        output = torch.cat(
            (
                # 这里的 -1 指的是 num_anchors*grid_size*grid_size
                # 即最终output shape -> (batch_size,num_anchors*grid_size*grid_size,self.num_classes + 5)
                # 这里的pred_boxes数据格式为 xywh在图片中的的相对大小 (0,1)
                pred_boxes.reshape(batch_size, -1, 4) * self.stride,
                pred_conf.reshape(batch_size, -1, 1),
                pred_cls.reshape(batch_size, -1, self.num_classes),
            ),
            -1,
        )
        # 如果是验证or测试的时候就到此为止了,直接返回预测的相关数据,否则返回loss进行更新梯度
        if targets is None:
            return output, 0
        elif targets.size(0):
            # 这个build_targets方法主要是为了计算loss做准备,把target转换成和pred_box相同的数据格式,方便计算
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
                grid_size=grid_size,
            )
            # 这里bool化的原因是方便下面进行切片取值
            obj_mask = obj_mask.bool()
            noobj_mask = noobj_mask.bool()
            # 坐标损失
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            # 检测损失
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            # 含有目标的损失权重要大于没有目标的损失权重
            # 但是实际上这里的代码和 eriklindernoren / PyTorch-YOLOv3一样，它们在损失函数上的一些细节与yolo3论文中有些许不同
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            # 分类损失
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            # 损失累加
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # 判断指标
            # 在target区域上正确预测含有目标且正确分类的概率的平均值 相当于分类recall
            cls_acc = 100 * class_mask[obj_mask].mean()
            # 在target区域上目标置信度的平均值 相当于检测recall
            conf_obj = pred_conf[obj_mask].mean()
            # 在target区域上无目标置信度的平均值
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask
            # 这里precision的定义为: (预测iou>0.5 * 预测conf>0.5)且pre_box的种类与target_box一致 / 所有预测 conf>0.5的
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-8)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-8)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-8)
            self.metrics = {
                # 相关损失
                "loss": loss.item(),
                "x": loss_x.item(),
                "y": loss_y.item(),
                "w": loss_w.item(),
                "h": loss_h.item(),
                "conf": loss_conf.item(),
                "cls": loss_cls.item(),
                # 指标
                "cls_acc": cls_acc.item(),
                "recall_50": recall50.item(),
                "recall_75": recall75.item(),
                "precision": precision.item(),
                "conf_obj": conf_obj.item(),
                "conf_noobj": conf_noobj.item(),
                "grid_size": grid_size,
            }
        else:
            loss_conf_noobj = self.bce_loss(pred_conf, torch.zeros_like(pred_conf))
            loss = self.noobj_scale * loss_conf_noobj
            conf_noobj = pred_conf.mean()

            self.metrics = {
                "loss": loss.item(),
                "conf_noobj": conf_noobj.item(),
            }
        return output, loss

class Mainnet(nn.Module):
    def __init__(self, cfgfile):
        super(Mainnet, self).__init__()
        # 获取yolov3.cfg的文件配置信息
        self.blocks = parse_cfg(cfgfile)
        # 获取模型参数及模型结构
        self.net_info, self.module_list = create_modules(self.blocks)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        # 获取前向传播的网络结构,第一层是net参数层
        # 这个是为了route层结合而临时创建的,保存了每层网络输出的数据,共107层
        layer_outputs, yolo_outputs = [], []
        loss = 0
        for block, module in zip(self.blocks,self.module_list):
            # 获取当前层的种类
            block_type = block["type"]
            if block_type in ["convolutional", "convolutional_dw", "upsample",'maxpool']:
                # 开始输入数据, x就是处理好的batch_size张图片
                x = module(x)
            # 如果当前层为bottleneck时,并判断是否需要跳跃链接

            elif block_type == "bottleneck":
                skip = int(block["skip"])
                x = module(x) + x * skip

            elif block_type == "route":
                # route层只有一个数字n的话就是将该route层等于第n层输出,
                # 如果有两个数字(n,m)则该route层等于将第n层与第m层相加(channel维度相加)然后输出
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in block["layers"].split(",")], 1)

            elif block_type == "shortcut":
                # 从前几层开始跳跃
                from_ = int(block["from"])
                # 直接相加
                x = layer_outputs[-1] + layer_outputs[from_]

            elif block_type == 'yolo':
                # 此时x就是最后的特征图,三层yolo检测对应三种尺度的特征图    batch_size=4  num_class=16
                # x.shape  ->  [4, 507, 16]    [4, 2028,16]    [4, 8112,16]
                # 其实这个时候每张图片的各种预测数据就已经出来了,只是还需要经处理一下
                x, layer_loss = module[0](x, targets)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        # 需要把三种尺度下的所有的anchors(3*(52*52+26*26+13*13))预测结果合并到一起.
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        return yolo_outputs if targets is None else (loss, yolo_outputs)
#
    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.blocks, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).reshape_as(conv_layer.weight)
                conv_layer.weight.detach().copy_(conv_w)
                ptr += num_w
                if module_def["batch_normalize"]:
                    # 加载 bn层的  weights, bias, running_mean, running_var等参数
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.weight)
                    bn_layer.weight.detach().copy_(bn_w)
                    ptr += num_b
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.bias)
                    bn_layer.bias.detach().copy_(bn_b)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.running_mean)
                    bn_layer.running_mean.detach().copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.running_var)
                    bn_layer.running_var.detach().copy_(bn_rv)
                    ptr += num_b
                else:
                    # 加载conv层的bias参数
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(conv_layer.bias)
                    conv_layer.bias.detach().copy_(conv_b)
                    ptr += num_b
            elif module_def["type"] == "convolutional_dw":
                # 加载conv_dw层的weight参数
                conv_layer = module[0]
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).reshape_as(conv_layer.weight)
                conv_layer.weight.detach().copy_(conv_w)
                ptr += num_w

                # 加载 bn层的  weights, bias, running_mean, running_var等参数
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()
                # Weight
                bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.weight)
                bn_layer.weight.detach().copy_(bn_w)
                ptr += num_b
                # Bias
                bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.bias)
                bn_layer.bias.detach().copy_(bn_b)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.running_mean)
                bn_layer.running_mean.detach().copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.running_var)
                bn_layer.running_var.detach().copy_(bn_rv)
                ptr += num_b

                # conv_pw层 同理
                conv_layer = module[3]
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).reshape_as(conv_layer.weight)
                conv_layer.weight.detach().copy_(conv_w)
                ptr += num_w

                # 加载 bn层的  weights, bias, running_mean, running_var等参数
                bn_layer = module[4]
                num_b = bn_layer.bias.numel()
                # Weight
                bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.weight)
                bn_layer.weight.detach().copy_(bn_w)
                ptr += num_b
                # Bias
                bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.bias)
                bn_layer.bias.detach().copy_(bn_b)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.running_mean)
                bn_layer.running_mean.detach().copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).reshape_as(bn_layer.running_var)
                bn_layer.running_var.detach().copy_(bn_rv)
                ptr += num_b


