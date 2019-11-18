from __future__ import division

from model import *
from util import *
from datasets import *

import argparse
import tqdm
import torch
from torch.utils.data import DataLoader


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    dataset = ListDataset(path,img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=6, collate_fn=dataset.collate_fn
    )

    FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        labels += targets[:, 1].numpy().tolist()
        # Rescale target
        targets[:, 2:] *= img_size
        targets = targets.cuda()
        imgs = imgs.type(FloatTensor)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = NMS(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # 合并sample_metrics
    # 这里的sample_metrics是一个list,[(batch*batch_size)*[true_positives, pred_scores.cpu(), pred_labels.cpu()]]
    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives.numpy(), pred_scores.numpy(), pred_labels.numpy(), labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/kalete/ep893-map80.55-loss0.00.weights", help="path to weights file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="在计算TP时,条件之一就是两个box的iou>iou_thres")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_path = r'D:\py_pro\YOLOv3-PyTorch\data\kalete\val.txt'
    class_names = r'D:\py_pro\YOLOv3-PyTorch\data\kalete\dnf_classes.txt'
    with open(class_names, 'r') as file:
        class_list = [i.replace('\n','') for i in file.readlines()]
    # Initiate model
    model = Darknet(opt.model_def).cuda()
    if opt.weights_path.endswith(".weights"):
        # 加载模型文件
        model.load_state_dict(torch.load(opt.weights_path))
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print("+ Class '{}' ({}) - AP: {}".format(c, class_list[c], AP[i]))

    print("mAP: ", AP.mean())
