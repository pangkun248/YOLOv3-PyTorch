class Config:
    def __init__(self):
        self.map_name = 'wenyi'
        self.model_name = 'yolov3-mobile'
        self.input_h = 320
        self.input_w = 320
        self.lr = 1e-3
        self.epochs = 50
        self.batch_size = 8
        self.conf_thres = 0.5  # nms时pred_box的obj_conf以及cls_conf阈值,目标置信度以及类别置信度小于此阈值的过滤掉
        self.iou_thres = 0.5
        # 计算mAP的时候,tp的条件之一的阈值 1.pred_box和所有target_box的最大iou 大于iou_thres 2.且类别一致 3.同一target_box不能被算作tp两次
        self.nms_thres = 0.5  # nms时iou的阈值,与最大score的pred_boxIOU超过此值的pred_box一律过滤掉,
        self.cfg_path = 'yolo_cfg\\' + self.model_name + '.cfg'
        self.weights_path = 'weights\\' + self.map_name + '\\yolov3-mobile_ep5-map27.70-loss1.80639.pt'
        self.train_path = 'data\\' + self.map_name + '\\train.txt'
        self.val_path = 'data\\' + self.map_name + '\\val.txt'
        self.class_path = 'data\\' + self.map_name + '\\dnf_classes.txt'
        self.pretrained = False  # 是否基于已有模型继续训练
        self.is_pruned = False  # 是否对模型进行稀疏化训练
        self.pruned_id = 1  # 剪枝方式 1为普通无shortcut剪枝 2为layer剪枝 3为slim剪枝
        self.sparse_rate = 0.01  # 稀疏因子,如果稀疏化训练出来的bn中γ值普遍偏大,则可以调大该值 理想γ值大部分应落在[0, 0.1]区间
        self.class_name = ["WhitehairedBanshee", "UndeadSkeleton", "WhitehairedMonster", "SlurryMonster", "MiniZalu",
                      "Dopelliwin", "ShieldAxe", "SkeletonKnight", "Zalu", "Cyclone", "SlurryBeggar", "Gerozaru",
                      "Catalog", "InfectedMonst", "Gold", "StormRider", "Close", "Door", ]
        self.test_path = 'test\\'   # 测试图片文件夹
        self.video_in = r'D:\BaiduNetdiskDownload\wenyi.avi'    # 测试视频输入保存路径
        self.video_out = 'wenyi_out.avi'   # 测试视频输出保存路径


cfg = Config()
