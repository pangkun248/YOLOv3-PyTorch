from prune.prune_tool import parse_blocks_slim, gather_bn_weights, beta2next, write_cfg, merge_mask, init_weights_from_loose_model
from model import YOLOv3
import torch
from test import evaluate
from copy import deepcopy
import time
from terminaltables import AsciiTable
# slim剪枝方法时channel_prune方法的一种改进,即对shortcut的每层剪枝掩膜进行合并.
# 那些在shorcut每层mask都为0的通道才会被剪掉,否则就保留

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
        'weights': 'D:\py_pro\YOLOv3-PyTorch\weights\\' + map_name + '\\yolov3_ep43-map82.67-loss0.15187.pt',
        'train_path': 'D:\py_pro\YOLOv3-PyTorch\data\\' + map_name + '\\train.txt',
        'val_path': 'D:\py_pro\YOLOv3-PyTorch\data\\' + map_name + '\\val.txt',
        'percent': 0.5,
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
    print(f'稀疏化训练后模型mAP:{before_AP.mean():.4f}')

    CBL_idx, Conv_idx, prune_idx = parse_blocks_slim(model.blocks)

    # 将所有要剪枝的BN层的绝对值化γ参数，拷贝到bn_weights一维tensor上
    bn_weights = gather_bn_weights(model.module_list, prune_idx)
    # torch.sort return: (value, index) 是排序后的值列表,排序后的值在排序前的索引 默认从小到大排序
    sorted_bn = torch.sort(bn_weights)[0]

    thresh_index = int(len(bn_weights) * import_param['percent'])
    thresh = sorted_bn[thresh_index].cuda()

    print(f'实际剪枝比例 {import_param["percent"]} 实际γ阈值 {thresh:.4f}')


    def get_filters_mask(model, thre, CBL_idx, prune_idx):

        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            if idx in prune_idx:
                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]
                # 每层保留的最低通道数,最小为1.防止整个层被剪掉
                min_channel_num = int(channels * 0.01) if int(channels * 0.01) > 0 else 1
                mask = weight_copy.gt(thresh).float()
                # 这里防止剪枝后的通道数少于最低通道比例(默认0.1),如果剪枝比例超过0.9,则默认剪枝比例为0.9
                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.clone())

        prune_ratio = pruned / total
        print(f'剪掉的通道数: {pruned}\t总通道数为: {total}\t剪枝比例为: {prune_ratio:.3f}')

        return num_filters, filters_mask

    num_filters, filters_mask = get_filters_mask(model, thresh, CBL_idx, prune_idx)
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}

    for i in model.blocks:
        if i['type'] == 'shortcut':
            i['is_access'] = False

    merge_mask(model, CBLidx2mask, CBLidx2filters)

    def prune_and_eval(model, CBL_idx, CBLidx2mask):
        model_copy = deepcopy(model)

        for idx in CBL_idx:
            bn_module = model_copy.module_list[idx][1]
            mask = CBLidx2mask[idx].cuda()
            bn_module.weight.data.mul_(mask)

        with torch.no_grad():
            precision, recall, ori_AP, f1, ap_class = evaluate(
                model_copy,
                path=import_param['val_path'],
                iou_thres=import_param['iou_thres'],
                conf_thres=import_param['conf_thres'],
                nms_thres=import_param['nms_thres'],
                img_size=320,
                batch_size=import_param['batch_size'],
            )

        print(f'将bn_γ参数置为0后,模型mAP为 {ori_AP.mean():.4f}')

    prune_and_eval(model, CBL_idx, CBLidx2mask)

    for i in CBLidx2mask:
        CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()

    # 虽然上面已经能看到将待裁剪bn的γ值置为0后的mAP,但bn层的γ和β都还没有处理
    # 现在将稀疏化训练后的模型中待剪枝层的bn中的β参数移植到后面的层.并返回移植后的模型
    pruned_model = beta2next(model, prune_idx, CBL_idx, CBLidx2mask)
    with torch.no_grad():
        precision, recall, AP, f1, ap_class = evaluate(
            pruned_model,
            path=import_param['val_path'],
            iou_thres=import_param['iou_thres'],
            conf_thres=import_param['conf_thres'],
            nms_thres=import_param['nms_thres'],
            img_size=320,
            batch_size=import_param['batch_size'],
        )
    print('剪枝层β值移植之后的mAP', round(AP.mean(),4))

    for i in model.blocks:
        if i['type'] == 'shortcut':
            i.pop('is_access')

    compact_blocks = deepcopy(model.blocks)
    for idx in CBL_idx:
        assert compact_blocks[idx]['type'] == 'convolutional'
        compact_blocks[idx]['filters'] = str(CBLidx2filters[idx])


    compact_model = YOLOv3([model.net_info.copy()] + compact_blocks).cuda()
    after_parameters = sum([param.nelement() for param in compact_model.parameters()])
    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

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
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{before_AP.mean():.6f}', f'{after_AP}'],
        ["Parameters", f"{before_parameters}", f"{after_parameters}"],
        ["Inference", f'{before_time:.4f}', f'{after_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    # 生成剪枝后的cfg文件并保存模型
    cfg_path = f'D:\py_pro\YOLOv3-PyTorch\yolo_cfg\\{map_name}_slim_{import_param["percent"]}_mAP_{after_AP}.cfg'
    write_cfg(cfg_path, [model.net_info.copy()] + compact_blocks)
    print(f'剪枝后的配置文件已被保存: {cfg_path}')

    weight_path = f'D:\py_pro\YOLOv3-PyTorch\weights\\{map_name}\_slim_{import_param["percent"]}_mAP_{after_AP}.pt'
    torch.save(compact_model.state_dict(), weight_path)
    print('剪枝后的权重文件已被保存:', weight_path)