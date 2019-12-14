from __future__ import division

from model import *
from util import *
from datasets import *
import time
import argparse
import tqdm
import torch
from torch.utils.data import DataLoader


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    dataset = ListDataset(path, img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=6, collate_fn=dataset.collate_fn
    )

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        labels += targets[:, 1].numpy().tolist()
        # 这里的targets中xywh还是以(0,1)之间的相对坐标
        targets = targets.cuda()
        imgs = imgs.cuda()

        with torch.no_grad():
            outputs = model(imgs)
            # 这里的outputs由于在网络中预测时的第一数据形式时以stride为单位的xywh坐标形式
            outputs = NMS(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    # 如果sample_metrics为空,说明网络预测出的物体conf_thres全部小于0.5,此时可以提前结束计算mAP相关值了

    # 合并sample_metrics
    # 这里的sample_metrics是一个list,[(batch*batch_size)*[true_positives, pred_scores.cpu(), pred_labels.cpu()]]
    if sample_metrics:
        true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives.numpy(), pred_scores.numpy(), pred_labels.numpy(),labels)
    else:
        precision, recall, AP, f1 = [np.zeros_like(np.unique(labels))]*4
        ap_class = np.unique(labels).astype("int32")
        print('未发现检测目标')
    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="yolo_cfg/yolov3-m.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/mogu/yolov3-m_ep90-map92.83-loss0.21698.weights",
                        help="path to weights file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="在计算TP时,条件之一就是两个box的iou>iou_thres")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_path = r'E:\YOLOv3-PyTorch\data\mogu\val.txt'
    class_names = r'E:\YOLOv3-PyTorch\data\mogu\dnf_classes.txt'
    with open(class_names, 'r') as file:
        class_list = [i.replace('\n', '') for i in file.readlines()]
    # Initiate model
    model = Mainnet(opt.model_def).cuda()
    if opt.weights_path.endswith(".weights"):
        # 加载模型文件
        model.load_state_dict(torch.load(opt.weights_path))
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("\n计算 mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )
    for i, c in enumerate(ap_class):
        print("+ Class '{}' ({}) - AP:{}  recall:{} precision:{}".format(c, class_list[c], round(AP[i],4),round(float(recall[i]),4),round(float(precision[i]),4)))

    print("mAP: ", round(AP.mean(),4))
    print("recall: ", round(recall.mean(),4))
    print("precision: ", round(precision.mean(),4))
