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
        'conf_thres': 0.5,
        'iou_thres': 0.5,
        'nms_thres': 0.1,
        'cfg_path': 'yolo_cfg\\' + model_name + '.cfg',
        'weights': 'weights\\' + map_name + '\\yolov3_ep1-map4.64-loss26.28099.pt',
        'train_path': 'data\\' + map_name + '\\train.txt',
        'val_path': 'data\\' + map_name + '\\val.txt',
        'percent': 0.79     # 代表了剪枝率
    }
    model = YOLOv3(import_param['cfg_path']).cuda()
    model.load_state_dict(torch.load(import_param['weights']))

    precision, recall, ori_AP, f1, ap_class = evaluate(
        model,
        path=import_param['val_path'],
        iou_thres=import_param['iou_thres'],
        conf_thres=import_param['conf_thres'],
        nms_thres=import_param['nms_thres'],
        img_size=320,
        batch_size=import_param['batch_size'],
    )
    # 剪枝前模型参数总量
    ori_num_parameters = sum([param.nelement() for param in model.parameters()])
    print(f'稀疏化训练后剪枝前模型mAP:{ori_AP.mean():.4f}')

    CBL_idx, Conv_idx, prune_idx = parse_module_defs(model.blocks)

    # 将所有要剪枝的BN层的绝对值化γ参数，拷贝到bn_weights一维tensor上
    bn_weights = gather_bn_weights(model.module_list, prune_idx)
    # torch.sort return: (value, index) 是排序后的值列表,排序后的值在排序前的索引 默认从小到大排序
    sorted_bn = torch.sort(bn_weights)[0]

    # 避免剪掉所有channel的最高阈值(从每个BN层的gamma的最大值中选出最小值即为阈值上限,再小就会有一些conv被整个剪掉)
    highest_gammas = []
    for idx in prune_idx:
        highest_gammas.append(model.module_list[idx][1].weight.data.abs().max().item())
    highest_gamma = min(highest_gammas)

    # 找到highest_gamma对应的下标对应的百分比
    percent_limit = (sorted_bn == highest_gamma).nonzero().item() / len(bn_weights)

    print(f'剪枝的gamma阈值最小为:{highest_gamma:.4f}.对应剪枝比例应小于 {percent_limit:.2f}')


    def prune_and_eval(model, sorted_bn, percent=import_param['percent']):
        """
        该方法主要是根据一个剪枝率来找到 sorted_bn中相对应位置的γ值
        然后将副本模型中需要剪枝位置的γ值置为0后,之后计算副本模型的mAP值及剪枝情况
        :param model: 原始模型
        :param sorted_bn: 从小到大排序过的 bn层中γ值,为1维数据
        :param percent: 剪枝百分比
        :return: 使剪枝率达到 percent大小的 γ值
        """
        model_copy = deepcopy(model)
        # 对应指定剪枝率下,bn中对应γ值的索引
        gamma_index = int(len(sorted_bn) * percent)
        # 获得α参数的阈值，小于该值的α参数对应的通道，全部裁剪掉
        gamma = sorted_bn[gamma_index]

        print(f'剪裁比例为{percent}的话,γ参数小于{gamma:.2f}的对应的通道,全部裁剪掉')

        remain_num = 0
        for idx in prune_idx:
            bn_module = model_copy.module_list[idx][1]
            # 获取每个bn层的剪枝情况 为1的表示保留,0的表示需要剪掉
            mask = get_bn_mask(bn_module, gamma)
            # 每个参与剪枝的bn层保留的通道数进行累加
            remain_num += int(mask.sum())
            # 对需要剪枝的bn层的γ系数置为0
            bn_module.weight.data.mul_(mask)

        precision, recall, AP, f1, ap_class = evaluate(
            model_copy,
            path=import_param['val_path'],
            iou_thres=import_param['iou_thres'],
            conf_thres=import_param['conf_thres'],
            nms_thres=import_param['nms_thres'],
            img_size=320,
            batch_size=import_param['batch_size'],
        )
        print(f'理论剪枝率为{percent},'
              f'待剪枝的conv通道总数经过剪枝从 {len(sorted_bn)} 到 {remain_num},'
              f'实际剪枝率为{1 - remain_num / len(sorted_bn):.2f},'
              f'将bn层中应该剪枝掉的层的γ参数置为0之后的模型mAP为 {AP.mean():.4f}')
        return gamma


    threshold = prune_and_eval(model, sorted_bn, 0.80)

    # 虽然上面已经能看到将待裁剪bn的γ值置为0后的一些效果,但还有bn层的β没有处理
    num_filters, filters_mask = get_filters_mask(model, threshold, CBL_idx, prune_idx)

    # CBLidx2mask conv层索引(yolo前一层除外) : conv层剪枝掩膜(1为保留 0为剪掉)
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}

    # 将稀疏化训练后的模型中待剪枝层的bn中的β参数移植到下后面的层.并返回移植后的模型
    pruned_model = beta2next(model, prune_idx, CBL_idx, CBLidx2mask)

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

    # 重新创建一份和原始cfg一模一样的网络配置文件,并更改剪枝层的卷积核个数
    compact_module_defs = deepcopy(model.blocks)
    for idx, num in zip(CBL_idx, num_filters):
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(num)

    # 通过剪枝后的cfg文件初始化一个新模型,并计算新模型的参数量
    compact_model = YOLOv3([model.net_info.copy()] + compact_module_defs).cuda()
    compact_nparameters = sum([param.nelement() for param in compact_model.parameters()])

    # 这一步就是将pruned_model中的部分权重移植到刚刚初始化后的compact_model,简单来说就是将pruned_model中conv层中多余的通道剪掉
    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

    # 随机生成测试数据
    random_input = torch.rand((1, 3, 320, 320)).cuda()


    def obtain_avg_forward_time(input, model, repeat=200):

        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output


    # 测试剪枝前后两个模型的前向传播时间,并且理论上来说 compact_output和compact_output应该完全一致
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)
    diff = (pruned_output - compact_output).abs().gt(0.001).sum().item()
    if diff > 0:
        print('剪枝过程中可能出现了某些问题导致两个模型的输出不一致')

    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    precision, recall, AP, f1, ap_class = evaluate(
        compact_model,
        path=import_param['val_path'],
        iou_thres=import_param['iou_thres'],
        conf_thres=import_param['conf_thres'],
        nms_thres=import_param['nms_thres'],
        img_size=320,
        batch_size=import_param['batch_size'],
    )

    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{ori_AP.mean():.4f}', f'{AP.mean():.4f}'],
        ["Parameters", f"{ori_num_parameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    # 生成剪枝后的cfg文件并保存模型
    weight_path = f'weights/{map_name}/prune_{import_param["percent"]}.cfg'
    pruned_cfg_file = write_cfg(weight_path, [model.net_info.copy()] + compact_module_defs)
    print(f'剪枝后的配置文件已被保存: {weight_path}')

    cfg_path = f'weights/{map_name}/prune_{import_param["percent"]}.pt'
    torch.save(compact_model.state_dict(), cfg_path)
    print('剪枝后的权重文件已被保存:', cfg_path)
