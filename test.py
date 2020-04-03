from model import *
from util import *
from datasets import *
import tqdm
import torch
from torch.utils.data import DataLoader
from terminaltables import AsciiTable


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    dataset = ListDataset(path, img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    )

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        labels += targets[:, 1].numpy().tolist()
        # 这里的targets中xywh还是以(0,1)之间的相对坐标,在计算TP时tensor在cpu上计算的会更快
        targets[:, 2:] = xywh2xyxy(targets[:, 2:]*int(model.net_info["height"]))
        imgs = imgs.cuda()
        # 事实上如果只设置model.eval()足以获得预期的效果。但是如果加上with torch.no_grad()将另外节省一些内存
        # eval只会改变BN与drop的行为方式,而no_grad则会节约显存,因为它不会存储任何中间张量
        # 而在eval模式下,bn使用的是保存的统计信息而不是来自每个batch的数据
        with torch.no_grad():
            outputs = model(imgs)
            # 这里的outputs由于在网络中预测时的第一数据形式时以stride为单位的xywh坐标形式
            outputs = NMS(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    # 如果sample_metrics为空,说明网络预测出的物体conf_thres或cls_score全部小于0.5,此时可以提前结束计算mAP相关值了

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
    map_name = 'mouse'
    model_name = 'yolov3-lite'
    import_param = {
        'batch_size': 4,
        'conf_thres': 0.5,
        'iou_thres': 0.5,
        'nms_thres': 0.5,
        'cfg_path': 'yolo_cfg\\' + model_name + '.cfg',
        'weights_path': 'weights\\' + map_name + '\\yolov3_ep87-map70.33-loss0.06912.weights',
        'class_path': 'data\\' + map_name + '\\dnf_classes.txt',
        'valid_path': 'data\\' + map_name + '\\val.txt',
    }
    for k,v in import_param.items():
        print(k, ':', v)
    with open(import_param['class_path'], 'r') as file:
        class_list = [i.replace('\n', '') for i in file.readlines()]
    # 在GPU上初始化模型
    model = Mainnet(import_param['cfg_path']).cuda()
    if import_param['weights_path']:
        # 加载模型文件
        # model.load_darknet_weights(import_param['weights_path'])
        model.load_state_dict(torch.load(import_param['weights_path']))
    else:
        print('无检测模型')
        exit()

    print("计算 mAP...")
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=import_param['valid_path'],
        iou_thres=import_param['iou_thres'],
        conf_thres=import_param['conf_thres'],
        nms_thres=import_param['nms_thres'],
        img_size=int(model.net_info['height']),
        batch_size=import_param['batch_size'],
    )
    ap_table = [["Index", "Class name", "Precision", "Recall", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_list[c], round(AP[i],2),round(float(recall[i]),2),round(float(precision[i]),2)]]
    print(AsciiTable(ap_table).table)
    print("mAP: ", round(AP.mean(),4))
