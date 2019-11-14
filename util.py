from __future__ import division

import torch
import random
import numpy as np
import cv2
import tqdm

CUDA = torch.cuda.is_available()

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def NMS(prediction, conf_thres, nms_thres):
    """
    # NMS 非最大值抑制
    # NMS方法需要的数据为,8,10647,16经过yolov3最后三层合并后的数据
    # nms_thres nms阈值为0.4,大于即认为两个box大概率属于同一物体
    # conf_thres 目标置信度阈值为0.5 超过则认为该pred_box内含有目标
    # 1.先把conf_thres小于0.5的过滤掉
    # 2.然后根据conf_thres*max(pre_class)的大小来排序各个pred_box,然后获取每个pred_box的score得分
    # 3.让image_pred根据score大小重新排序
    # 4.获取每个pred_box中所有预测类的最大值与其索引
    # 5.然后就是合并以上三个 最终image_pred[x,y,w,h,conf,max(class),ind(max_class)]
    # 6.获取那些大于nms_thres并且属于同类的pred_box的索引
    # 7.将这些公共部分的pred_box的conf提取出来当作权重weights,分别乘在各自的xyxy上,最后加起来除以sum(权重weights)
    # 8.将7步获得的值赋给image_pred排序第一的那个pred_box,再然后添加到一个事先准备好的列表中去,
    # 9.进行取反操作,把那些没有参与合并的pred_box赋值给原先的image_pred然后重复以上步骤,直到image_pred为空为止
    # 10.最后循环结束将列表中的pred_box给stack到一个张量上
    # 然后把这个张量添加到一个ouput列表中,最后返回一个batch_size,n,7  n是指最终一张图片中预测的物体数量
    prediction.shape -> [8,10647,16]
    返回数据形状:(x, y, w, h, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 筛选出那些目标置信度大于conf_thres的pre_box      image_pred.shape  -> [10647, 16]
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # 筛选出那些分类置信度大于conf_thres的pre_box
        if image_pred.size(0) == 0:
            continue
        image_pred = image_pred[image_pred[:, 5:].max(1)[0] >= conf_thres]
        # 如果没有一个pre_box的置信度大于conf_thres,则跳过本张图片     # 假设image_pred.shape  -> [40, 16]
        if image_pred.size(0) == 0:
            continue
        #        是否含有目标概率   预测的16个类别中概率最大的  为什么是[0],因为max返回(最大值,最大值索引)
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]     # score.shape  -> [40, ]
        # 这里是按conf*max(cls)大小来排序的
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # 开始执行NMS
        keep_boxes = []
        while detections.size(0):
            # 匹配那些iou大于nms_thres的
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # 匹配同类的
            label_match = detections[0, -1] == detections[:, -1]
            # 合并以上两个条件
            invalid = large_overlap & label_match
            # 注意这里这个变量名为什么叫weights,是因为代码作者(eriklindernoren)想要整合多个nms_thres大于阈值且预测属于同一类的pred_box
            # weights越大代表含有某一类物体的特征越多,同理越小代表越少,但是蚊子再小也是块肉对吧.
            # 我觉得代码作者应该是想让预测的结果更加精确些,希望尽可能的能把目标特征保留下来.
            weights = detections[invalid, 4]
            # 对于这里还是有些问题的 weights 和detections[invalid, :4]的shape 经常会让它两没办法相乘,
            # 因为Pytorch里广播对数据的shape有一定要求.所以这里需要临时增加一个维度,好让[n,1]匹配 [n,4]
            detections[0, :4] = (weights.unsqueeze(1) * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            # 这一步就是把剩下那些iou小于nms_thres或者和detections[0]不同类的重新赋值给detections,
            # 因为每次至少减少一个pred_box,所以最后一detections一定会成为[]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def compute_ap(recall, precision):
    # recall 从小到大, precision 从大到小
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # 获得那些recall变化的点的索引,为下面计算PR曲线下的面积做准备
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # 这里计算的AP会比实际偏大,因为它没有充分利用每一个TP.不过影响不是很大
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    # tp.shape -> (274,)    tp pred_cls conf长度一般比target_cls大,代表会有一些误检框
    # pred_cls.shape -> (274,)
    # conf.shape -> (274,)
    # len(target_cls) -> 264
    # 根据目标置信度从大到小排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # 将target_box中出现的所有类索引从小到大去重排序
    unique_classes = np.unique(target_cls)
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        # c类下实际目标的数量
        n_gt = (target_cls == c).sum()
        # c类下预测正确的目标的数量
        n_p = i.sum()
        # 这种情况是有物体但没有检测出来,就算该类的ap,r,p为0
        if n_p == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # 统计所有的FP TP
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall 其中的值总体越来越大,遇到FP不变,遇到TP变大
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision 其中的值总体越来越小,遇到FP则变小,遇到TP则变大.
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # 通过计算PR曲线下的面积来获得AP的值
            ap.append(compute_ap(recall_curve, precision_curve))

    # 计算F1-score recall和precision的调和平均数
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def get_batch_statistics(outputs, targets, iou_threshold):
    batch_metrics = []
    for sample_i in range(len(outputs)):
        # 这种情况下就是一张图片中没有超过一个nms_thres的pred_box
        if outputs[sample_i] is None:
            continue
        # 现在的output是经过NMS筛选的数据,即是最终预测的排过序的pred_boxshape  -> len(outputs) == batch_size
        # output[i]  -> [n*(x, y, w, h, object_conf, class_score, class_index)]
        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]
        true_positives = torch.zeros(pred_boxes.shape[0])
        # annotations[i] -> [n,5]  n->目标数 5->[class_pred, x, y, w, h]
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        # 如果图片中没有标注物体则让labels为空
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # 当target_box都被预测到时,就结束循环
                if len(detected_boxes) == len(annotations):
                    break
                # 如果pred_box的class不在target_boxes的class中则跳出
                if pred_label not in target_labels:
                    continue
                # 这里的box_index是pred_box与target_box最大iou的target_box的索引,主要是防止某一个target_box被两次预测到
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                # 这里对true_positives的计算和作者代码有些不一样,因为我觉得即使是两个不同label的物体iou也可能大于阈值
                # 详情见https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/233
                # 判断条件1.iou阈值
                #        2.与pred_box最大iou的target_box的索引是否出现两次,如果出现则代表某一target_box被预测两次,这不能算TP
                #        3.pred_box的class是否与target_box的class一致
                if iou >= iou_threshold and box_index not in detected_boxes and pred_label == target_labels[box_index]:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]

        batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])
    return batch_metrics


def bbox_iou(box1, box2):
    """
    返回box1和box2的ious  box1,box2 -> xywh
    box1:box2可以是1:N 可以是1:1 可以是N:1 (N:M没测试过)
    """
    # 关于如何计算iou的,各位可以自己在草稿纸上画一个有部分重合的两块矩形,然后再标上相应的x,y坐标
    # 结合下面的操作步骤,即可一目了然
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # 公共区域(交集)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)
    # 所有区域(并集)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

    def my_collate(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([cv2.resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, grid_size):
    #     pred_boxes  -> [8, 3, 13, 13, 4]  pred_cls -> [8, 3, 13, 13, 16]
    #     target  ->     [29, 6]            anchors  -> [3, 2]
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    nB = pred_boxes.size(0)  # batch_size
    nA = len(anchors)  # num_anchors
    nC = pred_cls.size(-1)  # num_classes
    nG = grid_size

    # Output tensors      8    3   13  13
    # 最后这些 想象一下是一个(batch_size,number_anchors,grid,grid)的feature_map,几乎大部分的loss计算都是在这上面操作的
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # 源target是padding后的0~1之间相对坐标,现在需要转换为以grid_size为单位下的坐标
    target_boxes = target[:, 2:6] * nG
    # target_xy target_wh shape -> (len(target),2)
    target_xy = target_boxes[:, :2]
    target_wh = target_boxes[:, 2:]
    # ious.shape -> (3,len(target)) 三种anchors尺寸下和各个target目标的宽高iou大小
    ious = torch.stack([bbox_wh_iou(anchor, target_wh) for anchor in anchors])
    # 获取3种anchors大小下的和真实box的iou最大的anchor索引
    best_ious, best_ind = ious.max(0)
    # 每个target所在batch中的索引及目标种类id
    i_in_batch, target_labels = target[:, :2].long().t()
    gx, gy = target_xy.t()
    gw, gh = target_wh.t()
    # gi和gj代表target_xy所在grid的左上角坐标
    gi, gj = target_xy.long().t()
    # 在obj_mask中,那些有target_boxes的的区域都设置为1.同理在noobj_mask中,有target_boxes的的区域都设置为0
    # obj_mask第一维度最大本应为8,但是这里不出意外的话应该会超过8,因为target_box会在同一张图片中有多个.
    obj_mask[i_in_batch, best_ind, gj, gi] = 1
    noobj_mask[i_in_batch, best_ind, gj, gi] = 0
    # 在noobj_mask中除了那些含有target_box的区域为1,那些iou大于一定阈值的也会设为0
    # anchors_ious为某个target_box在三种anchor下的iou值
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[i_in_batch[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
    # Coordinates
    tx[i_in_batch, best_ind, gj, gi] = gx - gx.floor()
    ty[i_in_batch, best_ind, gj, gi] = gy - gy.floor()
    # Width and height
    tw[i_in_batch, best_ind, gj, gi] = torch.log(gw / anchors[best_ind][:, 0] + 1e-16)
    th[i_in_batch, best_ind, gj, gi] = torch.log(gh / anchors[best_ind][:, 1] + 1e-16)
    # 这是一个标签掩膜,有target的为1
    tcls[i_in_batch, best_ind, gj, gi, target_labels] = 1
    # pred_cls[i_in_batch, best_ind, gj, gi] 是一个 len(target),num_class的数据.
    # 即网络在target_box(29个)位置预测的每个种类(16)得概率值  shape  -> 29,16
    # pred_cls[i_in_batch, best_ind, gj, gi].argmax(-1) 代表网络在target_box位置预测的最大概率的类的索引(即class_labels) 29,
    class_mask[i_in_batch, best_ind, gj, gi] = (pred_cls[i_in_batch, best_ind, gj, gi].argmax(-1) == target_labels).float()
    # 同理pred_boxes[i_in_batch, best_ind, gj, gi]为29,4的tensor,这里只是计算网络在target_boxes位置预测的xywh与真实的xywh的iou
    iou_scores[i_in_batch, best_ind, gj, gi] = bbox_iou(pred_boxes[i_in_batch, best_ind, gj, gi], target_boxes)
    # tconf 代表了一张featrue_map中哪些位置为1,有目标
    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


# 单个anchors和一个batch下的Gbox的iou
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    # 这里只是在目标形状上比较iou,
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area
