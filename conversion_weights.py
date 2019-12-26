import torch
from util import *
from model import *
from datasets import *


# 转换pytorch权重模型致libtorch可以调用的模型
# if __name__ == "__main__":
#     model_name = 'yolov3'
#     model = Mainnet('D:\py_pro\YOLOv3-PyTorch\yolo_cfg\\'+model_name+'.cfg').cuda()
#     model.load_state_dict(torch.load('D:\py_pro\YOLOv3-PyTorch\weights\wenyi\\yolov3_ep87-map70.33-loss0.06912.weights'))
#     print("网络权重加载成功.")
#     example = torch.rand(1, 3, 320, 320).cuda() # 注意,我这里导出的是CUDA版的模型`
#     model = model.eval()
#     traced_script_module = torch.jit.trace(model, example)
#     output = traced_script_module(torch.ones(1,3,320,320).cuda())
#     traced_script_module.save('D:\py_pro\YOLOv3-PyTorch\weights\yolov3.pt')
#     print(output)

def init_anchors(img_size=800, reduce=16):
    # g是grid的个数
    g = img_size // reduce
    # anchor长宽比例 1:0.5 1:1 1:2
    ratios = (0.5, 1, 2)
    # anchor面积种类 16*8 16*16 16*32
    anchor_scale = (8, 16, 32)
    # grid的中心坐标
    grid_center_x = torch.arange(0, g * reduce, reduce).repeat(g, 1).reshape([g, g]) + 8
    grid_center_y = torch.arange(0, g * reduce, reduce).repeat(g, 1).t().reshape([g, g]) + 8
    # 通过stack操作在二维上进行叠加,
    grid_center = torch.stack((grid_center_x, grid_center_y), 2)
    # 在grid_center第二维上新增一维,然后复制9次以形成每个grid中有9种anchor的目的
    anchor_center = grid_center.unsqueeze(2).repeat(1, 1, 9, 1)
    anchor_xyxy = torch.zeros(50, 50, 9, 4)
    for i in range(len(ratios)):
        for j in range(len(anchor_scale)):
            h = reduce * anchor_scale[j] * np.sqrt(ratios[i])
            w = reduce * anchor_scale[j] * np.sqrt(1. / ratios[i])
            # 这里的坐标形式是yxyx
            anchor_xyxy[:, :, i * 3 + j, 0] = anchor_center[:, :, i, 1] - h / 2
            anchor_xyxy[:, :, i * 3 + j, 1] = anchor_center[:, :, i, 0] - w / 2
            anchor_xyxy[:, :, i * 3 + j, 2] = anchor_center[:, :, i, 1] + h / 2
            anchor_xyxy[:, :, i * 3 + j, 3] = anchor_center[:, :, i, 0] + w / 2
    anchor_xyxy = anchor_xyxy.reshape(g * g * 9, 4)
    # 过滤掉那些超出图片边界的anchor
    valid_anchor_index = torch.where(
        (anchor_xyxy[:, 0] >= 0) &
        (anchor_xyxy[:, 1] >= 0) &
        (anchor_xyxy[:, 2] <= 800) &
        (anchor_xyxy[:, 3] <= 800)
    )[0]  # tensor([ 1404,  1413,  1422,  ..., 21069, 21078, 21087])        torch.Size([8940])

    valid_anchor = anchor_xyxy[valid_anchor_index]
    return anchor_xyxy, valid_anchor.numpy(), valid_anchor_index


#   计算 N:M类型的iou
def compute_iou(box1, box2):
    ious = np.zeros((box1.shape[0], box2.shape[0]))
    for i, box1_ in enumerate(box1):
        ya1, xa1, ya2, xa2 = box1_
        anchor_area = (ya2 - ya1) * (xa2 - xa1)  # anchor框面积
        for j, box2_ in enumerate(box2):
            yb1, xb1, yb2, xb2 = box2_
            box_area = (yb2 - yb1) * (xb2 - xb1)  # 目标框面积
            inter_x1 = max(xb1, xa1)
            inter_y1 = max(yb1, ya1)
            inter_x2 = min(xb2, xa2)
            inter_y2 = min(yb2, ya2)
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)  # anchor框和目标框的相交面积
                ious[i, j] = iter_area / (anchor_area + box_area - iter_area)  # IOU计算
            else:
                ious[i, j] = 0.
    return ious


def get_sample(ious, valid_anchor_len=0, pos_thres=0.7, neg_thres=0.3, pos_percent=0.5, n_sample=256):
    # 这是每个target_box与所有valid_anchor的最大iou值集合,(2,)
    gt_argmax_ious = ious.argmax(0)
    gt_max_iou = ious[gt_argmax_ious, np.arange(ious.shape[1])]
    # 这是每个valid_anchor与target_box的最大iou值集合,(8940,)
    anchor_argmax_ious = ious.argmax(1)
    # anchor_max_iou = ious[np.arange(ious.shape[0]),anchor_argmax_ious]
    # 所有anchor与target_box的IOU的最大值,注:可能会出现多个anchor与同一个target的IOU同为最大值
    gt_argmax_ious = np.where(ious == gt_max_iou)[0]
    print(gt_argmax_ious.shape[0])
    # 先将所有有效框的labels默认为 -1
    labels = np.empty((valid_anchor_len,), dtype=np.int32).fill(-1)
    labels[labels >= pos_thres] = 1
    labels[labels < neg_thres] = 0
    pos_nums = n_sample*pos_percent
    pos_index = np.where(labels==1)[0]
    if len(pos_index) > (pos_nums-gt_argmax_ious.shape[0]):
        disable_index = np.random.choice(pos_index,size=(len(pos_index) - pos_nums+gt_argmax_ious.shape[0]), replace=False)
        labels[disable_index] = -1
    # 将那些与target_box的IOU为最大值的anchor在最后赋值
    labels[gt_argmax_ious] = 1
    neg_nums = n_sample - np.sum(labels==1)
    neg_index = np.where(labels==0)[0]
    if neg_nums > len(neg_index):
def get_pos_neg_sample(ious, valid_anchor_len, pos_iou_threshold=0.7, neg_iou_threshold=0.3, pos_ratio=0.5,
                       n_sample=256):
    gt_argmax_ious = ious.argmax(axis=0)  # 找出每个目标实体框最大IOU的anchor框index，共2个, 与图片内目标框数量一致
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]  # 获取每个目标实体框最大IOU的值，与gt_argmax_ious对应, 共2个，与图片内目标框数量一致
    argmax_ious = ious.argmax(axis=1)  # 找出每个anchor框最大IOU的目标框index，共8940个, 每个anchor框都会对应一个最大IOU的目标框
    max_ious = ious[np.arange(valid_anchor_len), argmax_ious]  # 获取每个anchor框的最大IOU值， 与argmax_ious对应, 每个anchor框内都会有一个最大值

    gt_argmax_ious = np.where(ious == gt_max_ious)[0]  # 根据上面获取的目标最大IOU值，获取等于该值的index
    # print gt_argmax_ious.shape  # (18,) 共计18个
    label = np.empty((valid_anchor_len,), dtype=np.int32)
    label.fill(-1)
    # print label.shape  # (8940,)
    label[max_ious < neg_iou_threshold] = 0  # anchor框内最大IOU值小于neg_iou_threshold，设为0
    label[gt_argmax_ious] = 1  # anchor框有全局最大IOU值，设为1
    label[max_ious >= pos_iou_threshold] = 1  # anchor框内最大IOU值大于等于pos_iou_threshold，设为1

    n_pos = pos_ratio * n_sample  # 正例样本数

    # 随机获取n_pos个正例，
    pos_index = np.where(label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
        label[disable_index] = -1

    n_neg = n_sample - np.sum(label == 1)
    neg_index = np.where(label == 0)[0]

    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
        label[disable_index] = -1

    return label, argmax_ious


bbox = np.array([
    [20, 30, 400, 500],
    [300, 400, 500, 600]], dtype=np.float32
)  # [y1, x1, y2, x2] format

anchors, valid_anchor_boxes, valid_anchor_index = init_anchors()

a = time.time()
ious = compute_iou(valid_anchor_boxes, bbox)
# print((time.time()-a)*1000)
label, argmax_ious = get_sample(ious, valid_anchor_boxes.shape[0], pos_thres=0.7, neg_thres=0.3, pos_percent=0.5,
                                n_sample=256)
