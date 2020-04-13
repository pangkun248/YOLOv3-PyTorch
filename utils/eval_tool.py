import torch
import numpy as np
from tqdm import tqdm
from utils.util import compute_iou


def compute_ap(recall, precision):
    # recall 从小到大, precision 从大到小, 原作者在recall和precision后面分别加了 [1.0] [0.0],我发现没有必要就注释了
    # mrec = np.concatenate(([0.0], recall, [1.0]))
    # mpre = np.concatenate(([0.0], precision, [0.0]))
    # recall和precision前面的添加的两个[0.0]是为了计算当recall为最小值时求该recall值下的precision面积而准备的
    # 不过需要和下面的mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])部分代码 联合起来使用才能看出这里的细节
    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([0.0], precision))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # 获得那些recall变化的点的索引,为下面计算PR曲线下的面积做准备
    c = np.where(mrec[1:] != mrec[:-1])[0]
    # 这里计算的AP会比实际偏大,因为它实际上将P-R面积放大了.不过影响不大
    ap = np.sum((mrec[c + 1] - mrec[c]) * mpre[c + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    # 这里假设验证集中预测了274个目标,一般比target_cls大,代表会有一些误检框.当然,可大可小
    # tp.shape -> (274,)
    # pred_cls.shape -> (274,)
    # conf.shape -> (274,)
    # len(target_cls) -> 264
    # 根据目标置信度从大到小排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 将target_box中出现的所有类索引从小到大去重排序
    unique_classes = np.unique(target_cls)

    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="计算 AP"):
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
            tpc = tp[i].cumsum()

            # Recall 其中的值总体越来越大,遇到FP不变,遇到TP变大
            recall_curve = tpc / (n_gt + 1e-8)
            r.append(recall_curve[-1])

            # Precision 其中的值总体越来越小,遇到FP则变小,遇到TP则变大.
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # 通过计算PR曲线下的面积来获得AP的值
            ap.append(compute_ap(recall_curve, precision_curve))

    # 计算F1-score recall和precision的调和平均数
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-8)

    return p, r, ap, f1, unique_classes.astype("int32")


def get_batch_statistics(outputs, targets, iou_threshold):
    """
    :param outputs: 长度为batch_size的一个列表 batch_size*(len(pred_boxes),7) 7 -> x,y,w,h,conf_score,cls_score,cls_id
    :param targets: batch_size张图片合并到一起的target_box (len(target_boxes),6) -> batch_i,cls_id,x,y,w,h
    :param iou_threshold: pred_box与target_box的iou阈值,大于它则保留。否则舍弃
    :return:
    """
    batch_metrics = []
    for batch_i in range(len(outputs)):
        # 这种情况下就是一张图片中没有一个pre_box的conf_score或cls_score大于score_thres,详情见NMS方法前三行代码
        if outputs[batch_i] is None:
            continue
        # 现在的output是经过NMS筛选的数据,即是最终预测的排过序的pred_box.shape  -> len(outputs) == batch_size
        # output[batch_i] -> [n*(x, y, x, y, object_conf, class_score, class_index)] n是指一张图片中经过NMS筛选后的pred_box数量
        output = outputs[batch_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        class_index = output[:, -1]
        true_positives = torch.zeros(pred_boxes.shape[0])
        # target_id_boxes[i] -> [n,5]  n->一张图片中target_box数量 5->[target_labels, x, y, w, h]
        # 为了让targets中的图片索引与outputs中的对齐,即都是同一张图片的数据,一个预测数据一个真实数据
        target_id_boxes = targets[targets[:, 0] == batch_i][:, 1:]
        # 如果图片中没有标注物体则让labels为空
        target_labels = target_id_boxes[:, 0] if len(target_id_boxes) else []
        if len(target_id_boxes):
            # 存放已经预测到的target_box的索引
            detected_boxes = []
            target_boxes = target_id_boxes[:, 1:]
            # 对每张图片预测的pred_boxes进行循环计算
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, class_index)):
                # 当target_box都被预测到时,就结束循环
                if len(detected_boxes) == len(target_id_boxes):
                    break
                # 如果pred_box的class不在target_boxes的class中则跳出
                if pred_label not in target_labels:
                    continue
                # 这里的box_index是所有pred_box与某一个target_box的最大iou的pred_box的索引,主要是防止某一个pred_box预测两个target_box
                iou, target_index = compute_iou(pred_box.unsqueeze(0), target_boxes,xywh=False).max(0)
                # 这里对true_positives的计算和作者代码有些不一样,因为我觉得即使是两个不同label的物体iou也可能大于阈值
                # 详情见https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/233
                # 判断条件1.iou阈值
                #        2.与pred_box最大iou的target_box的索引是否已经出现,如果出现两次则代表某一target_box被预测两次最大IOU,这不能算TP
                #        3.pred_box的class是否与target_box的class一致
                if iou >= iou_threshold and target_index not in detected_boxes and pred_label == target_labels[target_index]:
                    true_positives[pred_i] = 1
                    detected_boxes.append(target_index)

        batch_metrics.append([true_positives, pred_scores.cpu(), class_index.cpu()])
    return batch_metrics