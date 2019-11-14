from __future__ import division
import time
import torch
import torch.nn as nn
import numpy as np
import datetime
from util import *
import argparse
from model import Darknet
from torch.utils.data import DataLoader
from datasets import *
from test import evaluate
from terminaltables import AsciiTable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv3检测模型')

    parser.add_argument("--epochs", help="训练轮数", default=1000)
    parser.add_argument("--batch_size", help="Batch size", default=8)
    parser.add_argument("--gradient_accumulations", help="每隔几次更新梯度", default=2)
    parser.add_argument("--confidence", help="目标检测结果置信度阈值", default=0.5)
    parser.add_argument("--iou_thres", help="在计算TP时,条件之一就是两个box的iou>iou_thres", default=0.5)
    parser.add_argument("--nms_thresh", help="NMS非极大值抑制阈值", default=0.4)
    parser.add_argument("--cfg", help="配置文件", default="yolov3.cfg", type=str)
    parser.add_argument("--weights", help="模型权重",
                        default="D:\py_pro\YOLOv3-PyTorch\weights\kalete\ep893-map80.55-loss0.00.weights", type=str)
    parser.add_argument("--evaluation_interval", type=int, default=2, help="每隔几次使用验证集")
    args = parser.parse_args()
    print(args)

    class_names = load_classes(r"D:\py_pro\YOLOv3-PyTorch\data\kalete\dnf_classes.txt")  # 加载所有种类名称
    train_path = r'D:\py_pro\YOLOv3-PyTorch\data\kalete\train.txt'
    val_path = r'D:\py_pro\YOLOv3-PyTorch\data\kalete\val.txt'

    print("载入网络...")
    model = Darknet(args.cfg)

    pretrained = True
    if pretrained:
        model.load_state_dict(torch.load(args.weights))
    else:
        # 随机初始化权重,会对模型进行高斯随机初始化
        model.apply(weights_init_normal)
    print("网络权重加载成功.")

    # 设置网络输入图片尺寸大小与学习率
    reso = int(model.net_info["height"])
    lr = float(model.net_info["learning_rate"])

    assert reso % 32 == 0  # 判断如果不是32的整数倍就抛出异常
    assert reso > 32  # 判断如果网络输入图片尺寸小于32也抛出异常

    if CUDA:
        model.cuda()

    train_dataset = ListDataset(train_path, reso)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=train_dataset.collate_fn,
    )

    # 使用Adam优化器, 不懂得可以参考https://www.sohu.com/a/149921578_610300
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    class_metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall_50",
        "recall_75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    mAP = 0
    for epoch in range(args.epochs):
        model.train()  # 训练的时候需要这一步,如果是测试的时候那就改成model.eval()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(train_dataloader):
            batches_done = len(train_dataloader) * epoch + batch_i
            imgs = imgs.cuda()
            targets = targets.cuda()
            loss, outputs = model(imgs, targets)
            loss.backward()
            # if batches_done % args.gradient_accumulations:
            optimizer.step()
            optimizer.zero_grad()
            # ----------------
            #   日志处理相关
            # ----------------

            # 获取每个yolo层的损失相关数据
            batch_metrics = [yolo.metrics for yolo in model.yolo_layers]
            # 打印当前训练状态的各项损失值
            print(
                "[Epoch %d/%d, Batch %d/%d] [Total_loss: %f, cls_acc %f, precision: %.5f, recall_50: %.5f, recall_75: %.5f]"
                % (
                    epoch,
                    args.epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.item(),
                    (batch_metrics[0]["cls_acc"]+batch_metrics[1]["cls_acc"]+batch_metrics[2]["cls_acc"])/3,
                    (batch_metrics[0]["precision"]+batch_metrics[1]["precision"]+batch_metrics[2]["precision"])/3,
                    (batch_metrics[0]["recall_50"]+batch_metrics[1]["recall_50"]+batch_metrics[2]["recall_50"])/3,
                    (batch_metrics[0]["recall_75"]+batch_metrics[1]["recall_75"]+batch_metrics[2]["recall_75"])/3,
                )
            )
            # model.seen += imgs.size(0)
        # 每epoch输出一次详细loss
        log_str = "\n [Epoch %d/%d] " % (epoch, args.epochs)
        log_str += " Total loss:" + str(loss.item()) + '\n'
        metric_table = [["Metrics", *["YOLO Layer " + str(i + 1) for i in range(len(model.yolo_layers))]]]
        for i, metric in enumerate(class_metrics):
            formats = {m: "%.6f" for m in class_metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

            # Tensorboard logging
            # tensorboard_log = []
            # for j, yolo in enumerate(model.yolo_layers):
            #     for name, metric in yolo.metrics.items():
            #         if name != "grid_size":
            #             tensorboard_log += [("{name}_{j+1}", metric)]
            # tensorboard_log += [("loss", loss.item())]
            # logger.list_of_scalars_summary(tensorboard_log, batches_done)

        log_str += AsciiTable(metric_table).table
        print(log_str)
        # 训练阶段每隔一定epoch在验证集上测试效果
        if epoch % args.evaluation_interval == 1:
            print("\n---- 评估模型 ----")
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=val_path,
                iou_thres=args.iou_thres,
                conf_thres=args.confidence,
                nms_thres=args.nms_thresh,
                img_size=reso,
                batch_size=args.batch_size,
            )
            # evaluation_metrics = [
            #     ("val_precision", precision.mean()),
            #     ("val_recall", recall.mean()),
            #     ("val_mAP", AP.mean()),
            #     ("val_f1", f1.mean()),
            # ]

            # 输出 class APs 和 mAP
            ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.3f" % precision[i], "%.3f" % recall[i], "%.3f" % AP[i], "%.3f" % f1[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            # 根据mAP的值保存最佳模型
            if AP.mean() > mAP:
                mAP = AP.mean()
                torch.save(model.state_dict(), 'weights\kalete\ep' + str(epoch) + '-map%.2f' % (
                        AP.mean() * 100) + '-loss%.5f' % loss.item() + '.weights')
    torch.cuda.empty_cache()
