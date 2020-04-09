from utils.prune_tool import *
from model import YOLOv3
from datasets import *
from test import evaluate
from copy import deepcopy
import time
from terminaltables import AsciiTable

if __name__ == "__main__":
    map_name = 'kalete'
    model_name = 'yolov3'
    import_param = {
        'batch_size': 16,
        'img_size': 320,
        'conf_thres': 0.5,
        'iou_thres': 0.5,
        'nms_thres': 0.1,
        'cfg_path': 'D:\py_pro\YOLOv3-PyTorch\yolo_cfg\\' + model_name + '.cfg',
        'weights': 'D:\py_pro\YOLOv3-PyTorch\weights\\' + map_name + '\\yolov3_ep46-map82.09-loss0.14039.pt',
        'train_path': 'D:\py_pro\YOLOv3-PyTorch\data\\' + map_name + '\\train.txt',
        'val_path': 'D:\py_pro\YOLOv3-PyTorch\data\\' + map_name + '\\val.txt',
        'prune_num': 16,    # YOLOv3标准网络中有23个res块,这里代表剪掉多少块
    }
    model = YOLOv3(import_param['cfg_path']).cuda()
    model.load_state_dict(torch.load(import_param['weights']))

    precision, recall, before_AP, f1, ap_class = evaluate(
        model,
        path=import_param['val_path'],
        iou_thres=import_param['iou_thres'],
        conf_thres=import_param['conf_thres'],
        nms_thres=import_param['nms_thres'],
        img_size=import_param['img_size'],
        batch_size=import_param['batch_size'],
    )
    # 剪枝前模型参数总量
    before_parameters = sum([param.nelement() for param in model.parameters()])
    print(f'稀疏化训练后剪枝前模型mAP:{before_AP.mean():.4f}')

    CBL_idx, _, shortcut_idx = parse_blocks_layer(model.blocks)

    # 将所有要剪枝的BN层的绝对值化γ参数，拷贝到bn_weights一维tensor上
    bn_weights = gather_bn_weights(model.module_list, shortcut_idx)
    # torch.sort return: (value, index) 是排序后的值列表,排序后的值在排序前的索引 默认从小到大排序
    sorted_bn = torch.sort(bn_weights)[0]

    # 这里更改了选层策略,由最大值排序改为均值排序,均值一般表现要稍好,但不是绝对,可以自己切换尝试.
    bn_mean = torch.zeros(len(shortcut_idx))
    for i, idx in enumerate(shortcut_idx):
        bn_mean[i] = model.module_list[idx][1].weight.data.abs().mean().clone()
    _, sorted_index_thre = torch.sort(bn_mean)

    # 确定需要剪枝掉的res层索引
    prune_shortcuts = torch.tensor(shortcut_idx)[[sorted_index_thre[:import_param['prune_num']]]]
    prune_shortcuts = [int(x) for x in prune_shortcuts]

    index_all = list(range(len(model.blocks)))
    index_prune = []
    for idx in prune_shortcuts:
        # 这个idx为shortcut前一层的索引
        index_prune.extend([idx - 1, idx, idx + 1])
    # index_remain代表的是网络中哪些层是保留下来的
    index_remain = [idx for idx in index_all if idx not in index_prune]

    def prune_and_eval(model, prune_shortcuts):
        model_copy = deepcopy(model)
        for idx in prune_shortcuts:
            for i in [idx, idx - 1]:
                bn_module = model_copy.module_list[i][1]
                # 将要剪枝的shortcut层的前两层conv中的bn中的γ全部置为0,然后测试mAP
                mask = torch.zeros(bn_module.weight.data.shape[0]).cuda()
                bn_module.weight.data.mul_(mask)

        with torch.no_grad():
            precision, recall, ori_AP, f1, ap_class = evaluate(
                model,
                path=import_param['val_path'],
                iou_thres=import_param['iou_thres'],
                conf_thres=import_param['conf_thres'],
                nms_thres=import_param['nms_thres'],
                img_size=320,
                batch_size=import_param['batch_size'],
            )

        print(f'将那些需要剪枝的CBL中的bn层中的γ参数置为0之后的mAP是 {ori_AP.mean():.4f}')

    prune_and_eval(model, prune_shortcuts)


    def get_filters_mask(model, CBL_idx, prune_shortcuts):

        filters_mask = []
        # 先将所有CBL中bn的γ掩膜全部初始化为1
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            mask = np.ones(bn_module.weight.data.shape[0], dtype='float32')
            filters_mask.append(mask.copy())
        CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
        # 然后将那些需要剪掉的conv中bn的γ置全部置为0
        for idx in prune_shortcuts:
            for i in [idx, idx - 1]:
                bn_module = model.module_list[i][1]
                mask = np.zeros(bn_module.weight.data.shape[0], dtype='float32')
                CBLidx2mask[i] = mask.copy()
        return CBLidx2mask
    CBLidx2mask = get_filters_mask(model, CBL_idx, prune_shortcuts)

    # 虽然上面已经能看到将待裁剪bn的γ值置为0后的mAP,但bn层的γ和β都还没有处理
    # 现在将稀疏化训练后的模型中待剪枝层的bn中的β参数移植到后面的层.并返回移植后的模型
    pruned_model = beta2next_layer(model, CBL_idx, CBL_idx, CBLidx2mask)

    precision, recall, AP, f1, ap_class = evaluate(
        pruned_model,
        path=import_param['val_path'],
        iou_thres=import_param['iou_thres'],
        conf_thres=import_param['conf_thres'],
        nms_thres=import_param['nms_thres'],
        img_size=320,
        batch_size=import_param['batch_size'],
    )
    print('剪枝之后的mAP', AP.mean())

    compact_blocks = deepcopy(model.blocks)

    # 对新创建的cfg文件中的rout层中layers参数进行修改
    for module_def in compact_blocks:
        if module_def['type'] == 'route':
            from_layers = [int(s) for s in module_def['layers'].split(',')]
            if len(from_layers) == 2:
                count = 0
                for i in index_prune:
                    if i <= from_layers[1]:
                        count += 1
                from_layers[1] = from_layers[1] - count
                from_layers = ', '.join([str(s) for s in from_layers])
                module_def['layers'] = from_layers

    compact_blocks = [compact_blocks[i] for i in index_remain]
    compact_model = YOLOv3([model.net_info.copy()] + compact_blocks).cuda()
    for i, index in enumerate(index_remain):
        compact_model.module_list[i] = pruned_model.module_list[index]
    after_parameters = sum([param.nelement() for param in compact_model.parameters()])

    # 计算模型前向传播平均耗费时间,及结果
    def obtain_avg_forward_time(input, model, repeat=200):
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output

    # pruned_output 与 compact_output的结果理论上来说应该完全一致,否则就是剪枝过程中出现了问题
    random_input = torch.rand((1, 3, 320, 320)).cuda()
    before_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    after_time, compact_output = obtain_avg_forward_time(random_input, compact_model)
    if not (compact_output == pruned_output).all():
        print('剪枝过程出了问题')

    # 在验证集上测试剪枝后的模型
    with torch.no_grad():
        precision, recall, after_AP, f1, ap_class = evaluate(
            compact_model,
            path=import_param['val_path'],
            iou_thres=import_param['iou_thres'],
            conf_thres=import_param['conf_thres'],
            nms_thres=import_param['nms_thres'],
            img_size=import_param['img_size'],
            batch_size=import_param['batch_size'],
        )
    after_AP = round(after_AP.mean(), 4)
    # 比较剪枝前后的参数数量、mAP、FPS的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{before_AP.mean():.6f}', f'{after_AP}'],
        ["Parameters", f"{before_parameters}", f"{after_parameters}"],
        ["Inference", f'{before_time:.4f}', f'{after_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    # 生成剪枝后的cfg文件并保存模型
    cfg_path = f'D:\py_pro\YOLOv3-PyTorch\yolo_cfg\\{map_name}_layer_pruned_{import_param["prune_num"]}_mAP_{after_AP}.cfg'
    pruned_cfg_file = write_cfg(cfg_path, [model.net_info.copy()] + compact_blocks)
    print(f'剪枝后的配置文件已被保存: {cfg_path}')

    weight_path = f'D:\py_pro\YOLOv3-PyTorch\weights\\{map_name}\layer_pruned_{import_param["prune_num"]}_mAP_{after_AP}.pt'
    torch.save(compact_model.state_dict(), weight_path)
    print('剪枝后的权重文件已被保存:', weight_path)