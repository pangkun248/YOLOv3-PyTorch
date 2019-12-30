from __future__ import division
import time
from util import *
import argparse
from model import Mainnet
from torch.utils.data import DataLoader
from datasets import *
from test import evaluate
import visdom
from terminaltables import AsciiTable
from torchsummary import summary

if __name__ == "__main__":
    map_name = 'wenyi'
    model_name = 'yolov3-tiny'
    import_param = {
        'epochs':100,
        'batch_size':16,
        'conf_thres':0.5,
        'iou_thres':0.5,    # 计算mAP的时候,tp的条件之一的阈值 1.pred_box和所有target_box的最大iou 大于iou_thres 2.且类别一致 3.同一box不能被算作tp两次
        'nms_thres':0.4,
        'evaluation_interval': 1,
        'cfg_path': 'D:\py_pro\YOLOv3-PyTorch\yolo_cfg\\'+model_name+'.cfg',
        'weights':'D:\py_pro\YOLOv3-PyTorch\weights\\'+map_name+'\\yolov3-t_ep95-map80.90-loss0.49322.weights',
        'class_path':'D:\py_pro\YOLOv3-PyTorch\data\\'+map_name+'\dnf_classes.txt',
        'train_path':'D:\py_pro\YOLOv3-PyTorch\data\\'+map_name+'\\train.txt',
        'val_path':'D:\py_pro\YOLOv3-PyTorch\data\\'+map_name+'\\val.txt',
        'pretrained':True
    }
    print(import_param,'\n',"载入网络...")
    model = Mainnet(import_param['cfg_path']).cuda()
    if import_param['pretrained']:
        # model.load_darknet_weights(import_param['weights'])
        model.load_state_dict(torch.load(import_param['weights']))
    else:
        # 随机初始化权重,会对模型进行高斯随机初始化
        model.apply(weights_init_normal)
    print("网络权重加载成功.")
    # 设置网络输入图片尺寸大小与学习率
    reso = int(model.net_info["height"])
    lr = float(model.net_info["learning_rate"])

    assert reso % 32 == 0  # 判断如果不是32的整数倍就抛出异常
    assert reso > 32  # 判断如果网络输入图片尺寸小于32也抛出异常

    train_dataset = ListDataset(import_param['train_path'], reso)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=import_param['batch_size'],
        shuffle=True,
        num_workers=0,
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
    class_names = load_classes(import_param['class_path'])  # 加载所有种类名称
    for epoch in range(1, import_param['epochs']):
        lr *= 0.98
        # 使用Adam优化器, 不懂得可以参考https://www.sohu.com/a/149921578_610300
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()  # 训练的时候需要这一步,如果是测试的时候那就改成model.eval()
        for batch_i, (img_path, imgs, targets) in enumerate(train_dataloader):
            # batches_done = len(train_dataloader) * epoch + batch_i
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
            if targets.size(0):
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
                print(
                    "[Epoch %d/%d, Batch %d/%d] [Total_loss: %f, precision: %.5f, recall_50: %.5f, recall_75: %.5f]"
                    % (
                        epoch,
                        import_param['epochs'],
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
                print(
                    "[Epoch %d/%d, Batch %d/%d] [Total_loss: %f, conf_noobj: %.5f, 该batch无标注目标]"
                    % (
                        epoch,
                        import_param['epochs'],
                        batch_i,
                        len(train_dataloader),
                        loss.item(),
                        conf_noobj / 3,
                    )
                )
        # 每epoch输出一次详细loss
        log_str = "\n [Epoch %d/%d] " % (epoch, import_param['epochs'])
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
        vis.line(X=torch.tensor([epoch]), Y=torch.tensor([loss.item()]), win='Loss', update='append',
                 opts={
                     'title': 'Loss',
                     'linecolor':np.array([[25, 25, 100]])
                 })
        # 训练阶段每隔一定epoch在验证集上测试效果
        if epoch % import_param['evaluation_interval'] == 0:
            print("\n---- 评估模型 ----lr:" + str(lr))
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=import_param['val_path'],
                iou_thres=import_param['iou_thres'],
                conf_thres=import_param['conf_thres'],
                nms_thres=import_param['nms_thres'],
                img_size=reso,
                batch_size=import_param['batch_size'],
            )
            # 可视化mAP输出
            vis.line(X=torch.tensor([epoch]), Y=torch.tensor([AP.mean()]), win='mAP', update='append',
                     opts={'title': 'mAP','linecolor':np.array([[25, 25, 100]])})

            # mAP_path = 'D:\py_pro\YOLOv3-PyTorch\mAP\\'+map_name+'-'+model_name+'.txt'
            # with open(mAP_path,'a+') as f:
            #     f.write(str(epoch)+' '+ str(AP.mean())+'\n')
            # 输出 class APs 和 mAP
            ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.3f" % precision[i], "%.3f" % recall[i], "%.3f" % AP[i], "%.3f" % f1[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            # 根据mAP的值保存最佳模型
            if AP.mean() > mAP:
                mAP = AP.mean()
                torch.save(model.state_dict(),'weights\\'+map_name+'\\'+model_name+'_ep' + str(epoch) + '-map%.2f' % (
                        AP.mean() * 100) + '-loss%.5f' % loss.item() + '.weights')
    torch.cuda.empty_cache()
