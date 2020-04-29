from utils.util import *
from model import YOLOv3
from torch.utils.data import DataLoader
from datasets import *
from prune.prune_tool import parse_blocks_normal,  parse_blocks_layer, parse_blocks_slim, updateBN
from config import cfg
from test import evaluate
import visdom
from terminaltables import AsciiTable

if __name__ == "__main__":
    model = YOLOv3(cfg.cfg_path).cuda()
    if cfg.pretrained:
        model.load_state_dict(torch.load(cfg.weights_path))
    else:
        # 随机初始化权重,会对模型进行高斯随机初始化
        model.apply(weights_init_normal)
    prune_set = {
        1 : parse_blocks_normal(model.blocks),  # 通道剪枝
        2 : parse_blocks_layer(model.blocks),   # 层剪枝
        3 : parse_blocks_slim(model.blocks),    # slim剪枝
    }
    _, _, prune_idx = prune_set[cfg.pruned_id]

    # 设置网络输入图片尺寸大小与学习率
    reso = int(cfg.input_h)
    lr = float(cfg.lr)

    assert reso % 32 == 0  # 判断如果不是32的整数倍就抛出异常
    assert reso > 32  # 判断如果网络输入图片尺寸小于32也抛出异常

    train_dataset = ListDataset(cfg.train_path, reso)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=train_dataset.collate_fn,
    )
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
    # 初始化mAP
    mAP = 0
    # 创建visdom可视化端口
    vis = visdom.Visdom(env='YOLOv3')
    for epoch in range(1, cfg.epochs):
        # if epoch % 20 == 0:
        #     lr = 0.2*lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        model.train()
        for batch_i, (img_path, imgs, targets) in enumerate(train_dataloader):
            imgs = imgs.cuda()
            targets = targets.cuda()
            loss, outputs = model(imgs, targets)
            loss.backward()
            if cfg.is_pruned:
                updateBN(model.module_list, cfg.sparse_rate, prune_idx)
            optimizer.step()
            optimizer.zero_grad()

            # ----------------
            #   日志处理相关
            # ----------------
            # 获取每个yolo层的损失相关数据
            batch_metrics = [yolo.metrics for yolo in model.yolo_layers]
            if targets.size(0):  # 这是在整个batch中有标注目标情况下
                # 打印当前训练状态的各项损失值 这里我省略了几个指标没有输出,
                # 因为我感觉以它们的数据表现来评判模型性能的话不是那么的清楚直观.需要的可以自行加上
                precision = 0
                recall_50 = 0
                recall_75 = 0
                # cls_acc = 0
                # conf_obj = 0
                # conf_noobj = 0
                for batch_metric in batch_metrics:
                    precision += batch_metric["precision"]
                    recall_50 += batch_metric["recall_50"]
                    recall_75 += batch_metric["recall_75"]
                    # cls_acc += batch_metric["cls_acc"]
                    # conf_obj += batch_metric["conf_obj"]
                    # conf_noobj += batch_metric["conf_noobj"]
                print("[Epoch %d/%d, Batch %d/%d] [Total_loss: %f, precision: %.5f, recall_50: %.5f, recall_75: %.5f]" %
                      (epoch,
                       cfg.epochs,
                       batch_i,
                       len(train_dataloader),
                       loss.item(),
                       precision / 3,
                       recall_50 / 3,
                       recall_75 / 3,
                       # cls_acc / 3,
                       # conf_obj / 3,
                       # conf_noobj / 3,
                       )
                      )
            else:
                conf_noobj = 0
                for batch_metric in batch_metrics:
                    conf_noobj += batch_metric["conf_noobj"]
                print("[Epoch %d/%d, Batch %d/%d] [Total_loss: %f, conf_noobj: %.5f, 该batch无标注目标]" %
                      (epoch, cfg.epochs, batch_i, len(train_dataloader), loss.item(), conf_noobj / 3,)
                      )
        # 每epoch输出一次详细loss
        log_str = "\n [Epoch %d/%d] " % (epoch, cfg.epochs)
        log_str += " Total loss:" + str(loss.item()) + '\n'
        metric_table = [["Metrics", *["YOLO Layer " + str(i + 1) for i in range(len(model.yolo_layers))]]]
        for i, metric in enumerate(class_metrics):
            formats = {m: "%.6f" for m in class_metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]
            # 这里原本是由TensorBoard的可视化输出的,但是我不太喜欢TensorFlow的东西.就把他它去掉了 感兴趣的可以去作者那里找一下
        log_str += AsciiTable(metric_table).table
        print(log_str)
        # 可视化 Loss输出 这里我使用的是Visdom的可视化 包括下面的mAP
        vis.line(X=torch.tensor([epoch]), Y=torch.tensor([loss.item()]), win='Loss',
                 update=None if epoch == 1 else 'append', opts={'title': 'Loss', })
        # 训练阶段每隔一定epoch在验证集上测试效果
        print("\n---- 评估模型 ----lr:" + str(lr))
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=cfg.val_path,
            iou_thres=cfg.iou_thres,
            conf_thres=cfg.conf_thres,
            nms_thres=cfg.nms_thres,
            img_size=reso,
            batch_size=cfg.batch_size,
        )
        # 可视化mAP输出
        vis.line(X=torch.tensor([epoch]), Y=torch.tensor([AP.mean()]), win='mAP',
                 update=None if epoch == 1 else 'append', opts={'title': 'mAP', })

        # 输出 class APs 和 mAP
        ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, cfg.class_name[c], "%.3f" % precision[i], "%.3f" % recall[i], "%.3f" % AP[i], "%.3f" % f1[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")
        # 根据mAP的值保存最佳模型
        if AP.mean() > mAP:
            mAP = AP.mean()
            torch.save(model.state_dict(),
                       'weights\\' + cfg.map_name + '\\' + cfg.model_name + '_ep' + str(epoch) + '-map%.2f' % (
                               AP.mean() * 100) + '-loss%.5f' % loss.item() + '.pt')
    torch.cuda.empty_cache()
