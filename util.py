import torch
import numpy as np
from tqdm import tqdm


FloatTensor = torch.cuda.FloatTensor if True else torch.FloatTensor

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def parse_cfg(cfgfile):
    # 加载文件并过滤掉文本中多余内容
    file = open(cfgfile, 'r')
    # 按行读取 相当于readlines
    lines = file.read().split('\n')
    # 去掉空行,并去掉以#开头的注释行
    lines = [x for x in lines if x and x[0] != '#']
    # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)
    lines = [x.strip() for x in lines]

    blocks = []
    for line in lines:
        if line[0] == "[":  # 这是cfg文件中一个层(块)的开始
            blocks.append({})
            blocks[-1]['type'] = line[1:-1].strip()  # 把cfg的[]中的块名作为键type的值
            # 如果是convolutional模块的话,默认batch_normalize为 0 .YOLOv3中有些层比如yolo前一层
            # 或者主干网络为mobilenetv2中bottleneck最后一层都是线性激活函数,没有batch_normalize这个参数的
            if blocks[-1]['type'] == 'convolutional':
                blocks[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")  # 按等号分割
            blocks[-1][key.strip()] = value.strip()
    return blocks


def NMS(prediction, conf_thres, nms_thres):
    '''
    NMS 非最大值抑制过程
    1.先把conf_thres小于0.5的过滤掉
    2.然后获取每个pred_box的conf_thres*max(pre_class)值 记作score
    3.让image_pred根据score大小重新排序
    4.获取每个pred_box中所有预测类的最大值与其索引
    5.然后就是合并以上三个 最终image_pred[x,y,w,h,conf,max(class),ind(max_class)]
    6.获取那些大于nms_thres并且属于同类的pred_box的索引
    7.将这些公共部分的pred_box的conf提取出来当作权重weights,分别乘在各自的xyxy上,最后加起来除以sum(权重weights)
    8.将7步获得的值赋给image_pred排序第一的那个pred_box,再然后添加到一个事先准备好的列表中去,
    9.进行取反操作,把那些没有参与合并的pred_box赋值给原先的image_pred然后重复以上步骤,直到image_pred为空为止
    10.最后循环结束将列表中的pred_box给stack到一个张量上
    然后把这个张量添加到一个ouput列表中,最后返回一个batch_size,n,7  n是指最终一张图片中预测的物体数量
    返回数据形状:(x1, y1, x2, y2, object_conf, class_score, class_pred)

    :param prediction: NMS方法需要的数据为(batch_size,10647,classes_num+5)经过YOLOv3最后三层合并后的数据
    :param conf_thres: 目标置信度阈值为0.5 超过则认为该pred_box内含有目标
    :param nms_thres: NMS阈值为0.4,大于即认为两个box大概率属于同一物体
    :return:
    '''
    # 由于在NMS中要多次循环,所以在计算iou的时候尽量选择计算简单的方式来,这里就是提前将pred_box转换成xyxy
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 过滤掉所有目标概率小于conf_thres的pre_box      image_pred.shape  -> [10647, 16]
        image_pred = image_pred[image_pred[:, 4] > conf_thres]
        # 这里是为了防止image_pred为空时,后续操作会报错报错而准备的,同下面那个continue同理
        if not image_pred.size(0):
            continue
        # 过滤掉所有分类概率小于conf_thres的pred_box
        image_pred = image_pred[image_pred[:, 5:].max(1)[0] > conf_thres]
        if not image_pred.size(0):
            continue
        #        是否含有目标概率   预测的16个类别中概率最大的  为什么是[0],因为max返回(最大值,最大值索引)
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]  # score.shape  -> [40, ]
        # 这里是按conf*max(cls)从大到小来重新排序的,注意image_pred应该算是一张图片中所有pred_box的集合
        image_pred = image_pred[(-score).argsort()]
        # 获取重新排序后的每个pred_box的分类概率最大值及其索引
        class_max, class_max_index = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_max.float(), class_max_index.float()), 1)
        # 开始执行NMS
        keep_boxes = []
        while detections.size(0):
            # a = time.time()
            # 匹配那些iou大于nms_thres的 unsqueeze(0)是在 第0维增加一维方便与detections[:, :4]进行计算
            large_overlap = compute_iou(detections[0, :4].unsqueeze(0), detections[:, :4], xywh=False) > nms_thres
            # 注:这里的compute_iou操作会有小概率发生意外,导致无限while循环
            # 以下几行注释参考自:https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/315
            # 这下面三行代码是防止detections中出现一些异常box例如x1y1x2y2 =(nan,nan,nan,nan,)或(0,100,0,150)
            # 这种无论怎么计算iou都为 nan 或 0 那么large_overlap及invalid就全为False,
            # 不过由于这里的detections是Tensor,即最后 分子tensor([0., 0., 0., 0.], device='cuda:0') 分母tensor(0., device='cuda:0')
            # 最后detections[0, :4]就tensor([nan, nan, nan, nan], device='cuda:0')
            # 注意这里进行除法操作时并不会报分母为0的错误,即使分母为int型的0也不会报错,可能是因为在PyTorch中的张量算术运算比较特殊?
            # 至少在PyTorch1.3 CUDA10.1 python3.7 Win10环境中是这样
            if not any(large_overlap):
                detections = detections[1:]
                continue
            # 匹配同类的,其实这个条件在部分情况下省略会有更好的效果,即不区分是否在同一种类下的NMS
            # label_match = detections[0, -1] == detections[:, -1]
            # 合并以上两个条件

            # invalid = large_overlap & label_match
            # 注意这里这个变量名为什么叫weights,是因为代码作者(eriklindernoren)想要整合多个nms_thres大于阈值且预测属于同一类的pred_box
            # weights越大代表含有某一类物体的特征越多,同理越小代表越少,但是蚊子再小也是块肉对吧.
            # 我觉得作者应该是想让预测的结果更加精确些,希望尽可能的能把目标特征保留下来.整合特征吧
            weights = detections[large_overlap, 4:5]
            detections[0, :4] = (weights * detections[large_overlap, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            # 这一步就是把剩下那些iou小于nms_thres或者和detections[0]不同类(~取反操作)的重新赋值给detections,
            # 一般情况下每次循环会减少一个或多个pred_box,所以最后detections.size(0)不出意外的话会等于 0
            # 注意！如果detections中出现了一些奇怪的box,像上面说的那样,那么这里的invalid就全部为False,
            # 即detections还是等于其本身.不会减少任何pred_box.就陷入无限循环了
            detections = detections[~large_overlap]
        # 如果某张图片中有检测目标则合并所有的检测目标 否则默认为None
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output


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
    for c in tqdm(unique_classes, desc="Computing AP"):
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
    batch_metrics = []
    for batch_i in range(len(outputs)):
        # 这种情况下就是一张图片中没有一个pre_box的目标置信度大于conf_thres,详情见NMS方法前三行代码
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
                iou, box_index = compute_iou(pred_box.unsqueeze(0), target_boxes,xywh=False).max(0)
                # 这里对true_positives的计算和作者代码有些不一样,因为我觉得即使是两个不同label的物体iou也可能大于阈值
                # 详情见https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/233
                # 判断条件1.iou阈值
                #        2.与target_box最大iou的pred_box的索引是否出现两次,如果出现两次则代表某一pred_box预测两次最大IOU,这不能算TP
                #        3.pred_box的class是否与target_box的class一致
                if iou >= iou_threshold and box_index not in detected_boxes and pred_label == target_labels[box_index]:
                    true_positives[pred_i] = 1
                    detected_boxes.append(box_index)

        batch_metrics.append([true_positives, pred_scores.cpu(), class_index.cpu()])
    return batch_metrics


def compute_iou(box1, box2,xywh=True):
    """
    返回box1和box2的ious  box1,box2 -> xywh
    box1:box2可以是1:N 可以是1:1 可以是M:1 但是不支持N:M
    关于如何计算iou的,各位可以自己在草稿纸上画一个有部分重合的两块矩形,然后再标上相应的x,y坐标.结合下面的操作步骤,即可一目了然
    """
    if xywh:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # 公共区域(交集)
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1, min=0) * torch.clamp(inter_rect_y2-inter_rect_y1+1, min=0)
    # 所有区域(并集)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-8)
    return iou


def load_classes(namesfile):
    # 加载各个类的名称
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


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, grid_size):
    '''
    :param pred_boxes: 预测目标的四个值xywh shape -> [batch_size, 3, grid_size, grid_size, 4]
    :param pred_cls: 预测目标的所有分类概率 shape -> [batch_size, 3, grid_size, grid_size, classes_num]
    :param target:  真实目标的相关信息(图片索引,class_id,x,y,w,h) shape -> [len(target), 6]
    :param anchors: anchors  某一YOLO层下的anchor尺寸 shape -> [3, 2]
    :param ignore_thres: iou忽略阈值,当iou超过这一值时,将noobj_mask设为 0
    :param grid_size: 某一YOLO层下的grid尺寸
    :return: iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf 详情见下面注释
    '''
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    nB = pred_boxes.size(0)  # batch_size
    nA = len(anchors)  # num_anchors
    nC = pred_cls.size(-1)  # num_classes
    nG = grid_size

    # Output tensors      8    3   13  13       (只是其中一个YOLO层)
    # 最后这些 想象一下是一个(batch_size,number_anchors,grid,grid)的feature_map,几乎大部分的loss计算都是在这上面操作的
    # 目标掩膜 有真实目标的位置为1,否则默认为0
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    # 非目标掩膜 有真实目标的位置为0,否则默认为1,与obj_mask对立
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    # 类掩膜 在target_boxes位置上预测的分类概率最大的那个class_id与target_boxes的class_id一致才会令该处的值为 1, 否则默认为 0
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    # 网络在target_boxes位置预测的xywh与target_boxes的xywh的iou
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # 源target是padding后的0~1之间相对坐标,现在需要转换为以grid_size为单位下的坐标 (len(target),4) 方便下面计算box_iou
    target_boxes = target[:, 2:6] * nG
    # target_xy target_wh shape -> (len(target),2)
    target_xy = target_boxes[:, :2]
    target_wh = target_boxes[:, 2:]
    # ious.shape -> (3,len(target)) 三种anchors尺寸下和各个target目标的宽高iou大小
    ious = torch.stack([bbox_wh_iou(anchor, target_wh) for anchor in anchors])
    # 获取3种anchors大小下的和真实box的iou最大的anchor索引
    best_ious, best_ind = ious.max(0)
    # 每个target所在图片在一个batch中的索引及目标种类id,注意这里的i_in_batch和target_labels可能会重复的,即一张图片中有两个同类目标！！！
    i_in_batch, target_labels = target[:, :2].long().t()
    gx, gy = target_xy.t()
    gw, gh = target_wh.t()
    # gi和gj代表target_xy所在grid的左上角坐标,这个long()方法可以理解为向下取整操作
    gi, gj = target_xy.long().t()
    # 在obj_mask中,那些有target_boxes的的区域都设置为1.同理在noobj_mask中,有target_boxes的的区域都设置为0
    # obj_mask第一维度最大本应为8(如果batch_size=8),但是这里不出意外的话应该会超过8,因为target_box会在同一张图片中有多个.
    # 这里obj_mask中的值如何才能算作1呢,就是target_boxes的坐标向下取整后和哪个grid坐标相同,target_boxes就属于那个grid里.
    # anchor这里也是一样target_boxes的长宽和哪个尺寸的anchor的iou最接近就属于哪个anchor(best_ind).
    # 以及这个target_boxes原本属于哪张图片的(i_in_batch),最后就由这四个值决定的.noobj_mask同理
    obj_mask[i_in_batch, best_ind, gj, gi] = 1
    noobj_mask[i_in_batch, best_ind, gj, gi] = 0
    # 在noobj_mask中除了某些grid含有target_box并且和target_box有最佳iou的区域为0,那些iou大于一定阈值的也会设为 0
    # anchors_ious为某个target_box和三个尺寸下的anchor的iou值 ious.t() shape -> (len(target),3)
    # 这里相当于有意减少一些负样本吧,这在正负样本极不平衡的条件下也是个好处
    # 那为什么不顺便也在obj_mask中设为 1,因为这会影响到后面一系列loss以及指标的计算.
    # 让Loss虚假的降低,mAP虚假的提升(甚至在单个YOLO层中验证集上mAP可能大于1)实际上并没有
    # 在YOLOv3中每个YOLO检测层中正样本是唯一的(如果对于标准YOLOv3来说一个正样本应该出现了三次)
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[i_in_batch[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
    # target_xy的相对坐标
    tx[i_in_batch, best_ind, gj, gi] = gx - gx.floor()
    ty[i_in_batch, best_ind, gj, gi] = gy - gy.floor()
    # target_wh的相对宽高
    tw[i_in_batch, best_ind, gj, gi] = torch.log(gw / anchors[best_ind][:, 0] + 1e-8)
    th[i_in_batch, best_ind, gj, gi] = torch.log(gh / anchors[best_ind][:, 1] + 1e-8)
    # 这是一个标签掩膜,有target的那一类target_label为1
    tcls[i_in_batch, best_ind, gj, gi, target_labels] = 1
    # pred_cls[i_in_batch, best_ind, gj, gi] 是一个 (len(target),num_class)的数据.
    # 即网络在target_box位置预测的所有种类(16)的概率值  shape  -> len(target),16
    # pred_cls[i_in_batch, best_ind, gj, gi].argmax(-1) 代表网络在target_box位置预测的最大概率的类的索引(即max_class_index)
    class_mask[i_in_batch, best_ind, gj, gi] = (pred_cls[i_in_batch, best_ind, gj, gi].argmax(-1) == target_labels).float()
    # pred_boxes[i_in_batch, best_ind, gj, gi]为(len(target),4)的tensor,这里只是计算网络在target_boxes位置预测的xywh与真实的xywh的iou
    iou_scores[i_in_batch, best_ind, gj, gi] = compute_iou(pred_boxes[i_in_batch, best_ind, gj, gi], target_boxes, xywh=True)
    # tconf 这里进行float处理的原因是为了后面计算loss时和pred_box的float类型对齐
    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def bbox_wh_iou(anchor_wh, true_boxes_wh):
    '''
    :param anchor_wh: 单个anchor的宽高 (2)
    :param true_boxes_wh: 一个batch下的true_box (29,2)
    :return: 长宽iou (29)
    '''
    # 将true_boxes_wh进行转置 (29,2)->(2,29) 主要是方便进行iou
    true_boxes_wh = true_boxes_wh.t()
    w1, h1 = anchor_wh[0], anchor_wh[1]
    w2, h2 = true_boxes_wh[0], true_boxes_wh[1]
    # 这里只是在目标形状上比较iou,方法应该就是将两个矩形的左上角对齐,然后计算 重合部分 / 公共部分
    # 最小宽和高然后相乘得重合部分面积
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    # 矩形1面积+矩形2面积-重合部分
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area
