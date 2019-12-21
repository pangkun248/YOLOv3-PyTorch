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
        # 这里的targets中xywh还是以(0,1)之间的相对坐标,在计算TP时tensor在cpu上计算的会更快
        targets[:, 2:] = xywh2xyxy(targets[:, 2:]*320)
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
    map_name = 'wenyi'
    model_name = 'yolov3'
    import_param = {
        'batch_size': 1,
        'conf_thres': 0.8,
        'iou_thres': 0.5,
        'nms_thres': 0.4,
        'cfg_path': 'D:\py_pro\YOLOv3-PyTorch\yolo_cfg\\' + model_name + '.cfg',
        'weights_path': 'D:\py_pro\YOLOv3-PyTorch\weights\\' + map_name + '\\yolov3_ep3-map13.07-loss4.51528.weights',
        'class_path': 'D:\py_pro\YOLOv3-PyTorch\data\\' + map_name + '\dnf_classes.txt',
    }
    print(import_param, '\n', "载入网络...")
    valid_path = r'D:\py_pro\YOLOv3-PyTorch\data\wenyi\val.txt'
    class_names = r'D:\py_pro\YOLOv3-PyTorch\data\wenyi\dnf_classes.txt'
    with open(class_names, 'r') as file:
        class_list = [i.replace('\n', '') for i in file.readlines()]
    # 在GPU上初始化模型
    model = Mainnet(import_param['cfg_path']).cuda()
    if import_param['weights_path'].endswith(".weights"):
        # 加载模型文件
        model.load_darknet_weights(import_param['weights_path'])
        # model.load_state_dict(torch.load(opt.weights_path))
    else:
        print('无检测模型')
        exit()
        # model.load_state_dict(torch.load(import_param['weights_path']))

    print("计算 mAP...")
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=import_param['iou_thres'],
        conf_thres=import_param['conf_thres'],
        nms_thres=import_param['nms_thres'],
        img_size=int(model.net_info['height']),
        batch_size=import_param['batch_size'],
    )
    for i, c in enumerate(ap_class):
        print("+ Class '{}' ({}) - AP:{}  recall:{} precision:{}".format(c, class_list[c], round(AP[i],4),round(float(recall[i]),4),round(float(precision[i]),4)))

    print("mAP: ", round(AP.mean(),4))
    print("recall: ", round(recall.mean(),4))
    print("precision: ", round(precision.mean(),4))
