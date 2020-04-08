import torch
from copy import deepcopy
import torch.nn.functional as F
import numpy as np


def write_cfg(cfg_file, module_defs):
    # 根据剪枝后的模型配置写入cfg文件
    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


def get_input_mask(module_defs, idx, CBLidx2mask):
    # 获取每层conv的输入mask
    # 第一层是固定的 图像的RGB三通道
    if idx == 0:
        return np.ones(3)

    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))
        if len(route_in_idxs) == 1:
            return CBLidx2mask[route_in_idxs[0]]
        elif len(route_in_idxs) == 2:
            # 这里不是平白无故剪去1的,是因为rout层中layers有两个值实际所代表的层都是shortcut或upsmale层,故需要进行-1
            return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
        else:
            print("rout模块出了某些问题,layers属性可能有多个值或没有值")
            exit()


def get_filters_mask(model, gamma_thres, CBL_idx, prune_idx):
    """
    主要方法就是循环整个网络中的所有CBL层,通过给定的γ值来筛选一个conv层中那些通道要保留那些剪掉,最后返回整个网络的剪枝数量及掩膜
    :param model: 原始模型
    :param gamma_thres:  剪枝的γ阈值
    :param CBL_idx: 所有的 conv层索引,包含了upsample前一层 shortcut层中的起始末尾层,还有maxpool层(如果有的话),不含yolo前一层
    :param prune_idx: 实际上要剪枝的conv层索引
    :return: 原始模型每个conv层要保留的数量,以及对应的掩膜
    """
    pruned = 0
    # total = 0
    num_filters = []
    filters_mask = []
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:
            # 获取某一bn层剪枝的掩膜,为1的表示保留,0的表示需要剪掉
            mask = get_bn_mask(bn_module, gamma_thres).cpu().numpy()
            # 计算一个bn层剪枝后剩余的通道总和
            remain = int(mask.sum())
            # 对每个bn层将要剪掉的通道数进行累加
            pruned += (mask.shape[0] - remain)

            if remain == 0:
                print("该通道将整个被删减", idx)  # 一般上不会出现这种情况,除非实际剪枝幅度超过理论最大剪枝幅度
                exit()
        else:
            # 该bn层的掩膜全部为1(所有通道全部保留) 即upsample前一层 shortcut层中的起始末尾层,还有maxpool层(如果有的话),yolo前一层
            mask = np.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]
        # 统计所有bn层的理论通道数
        # total += mask.shape[0]
        # 每个bn层保留的通道数量
        num_filters.append(remain)
        # 每个conv层剪枝的掩膜 1保留 0剪去
        filters_mask.append(mask.copy())

    return num_filters, filters_mask


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):
    """
    该方法主要就是将loose_model中保留的conv层、bn层的权重移植到 compact_model中去
    :param compact_model: 根据剪枝后的cfg文件重新初始化的模型
    :param loose_model:   移植了β值的稀疏化模型
    :param CBL_idx:       有bn层的conv索引
    :param Conv_idx:      无bn层的conv索引
    :param CBLidx2mask:   key:有bn层的conv索引 value:该层的剪枝掩膜
    :return:
    """
    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        # bn中保留的通道索引            (num_channels,1) -> (num_channels,) -> tolist()
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        # 将原模型中保留下来的bn中的γ和β以及均值和方差复制到剪枝后模型上去
        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        # 获取模型pruned_modex中每层conv的输入mask
        input_mask = get_input_mask(loose_model.blocks, idx, CBLidx2mask)
        # 计算出输入通道的mask中非0的索引,即保留的索引,然后得出剪枝过的输入(上一层输出即这一层输入),输入通道的更改完成.
        # 将其存为中间值后,再通过本层输出通道的mask得出最终剪切后的conv层的权重,输出通道的更改完成.整个conv层完成剪枝
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    # Conv_idx:[81, 93, 105] YOLO层前一层的conv只需要更改输入通道即可,输出通道是固定的(num_class+5)*3
    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.blocks, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data = loose_conv.bias.data.clone()


def updateBN(module_list, s, prune_idx):
    for idx in prune_idx:
        # Squential(Conv, BN, L-relu)
        bn_module = module_list[idx][1]
        bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1


def beta2next(model, prune_idx, CBL_idx, CBLidx2mask):
    """
    该方法主要是将待剪枝层bn的β值做一下处理,然后用它的下一层的conv层的bn中的mean或者conv层中的bias来吸收这个处理值
    例: 第n层 y = ReLu{BN1[CONV1(x)]}   第n+1层 z = ReLu{BN2[CONV2(y)]} 假设如果有BN和ReLu的话
    -> y = ReLU(γ1 * [(x - mean1) / std1] + β1)
    -> z = ReLU(γ2 * [(y - mean2) / std1] + β2)
    第n层剪枝后(将待剪枝通道的γ置为0)y就分为了两部分,保留下来的γ与β 与γ置为0的β 设该层的剪枝掩膜为mask 1保留 0剪掉
    -> y = ReLU(γ1*mask * [(x - mean1) / std1] + β1*mask) + ReLU(0*(1-mask) * [(x - mean1) / std1] + β1*(1-mask))
    -> y = ReLU(γ1*mask * [(x - mean1) / std1] + β1*mask) + ReLU(β1*(1-mask))
    ->   = y' + ReLU(β1*(1-mask))

    所以第n+1层 z = ReLU(γ2 * [(CONV2(y') + CONV2(ReLU(β1*(1-mask))) - mean2) / std2] + β2)
    带入上面的式子我们可以发现,在保证第二层采用同样的计算方式和结果不变的情况下:令 mean2' = mean2 - CONV2(ReLU(β1*(1-mask)))
    -> z = ReLU(γ2 * [CONV2(y') - mean2'] / std2 + β2)

    同理,如果第n+1层是无bn的conv层的话,z = CONV2(y') + bias  令 bias' = bias+CONV2(ReLU(β1*(1-mask)))
    -> z = CONV2(y') + bias'
    :param model:       原始稀疏化训练后的模型
    :param prune_idx:   待剪枝的conv层索引 不包括upsample前一层、maxpool前一层、YOLO前一层、shortcut起始末尾层这些层
    :param CBL_idx:     有bn层的conv索引,YOLO层前一层除外
    :param CBLidx2mask: CBL_idx中conv层对应的剪枝掩膜 1保留 0剪掉
    :return: 处理后的剪枝模型
    """
    pruned_model = deepcopy(model)
    for idx in prune_idx:
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        bn_module = pruned_model.module_list[idx][1]

        bn_module.weight.data.mul_(mask)
        # mask 中0为剪去1为保留 这样会把剪去的bn中β参数保留下来,而留下来的bn层中的β则为0
        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)

        # 两个上采样层前的卷积层,注意这里只是针对于darknet-53层的标准网络,如果主干网络变更的话,这里的数字也要进行相应的变更
        next_idx_list = [idx + 1]
        # 因为yolo层的前三层网络会有两个分叉一个直接通往yolo层,一个继续进行卷积.所以这里需要额外处理一下
        # 注意! 这里主干网络默认是darknet-53,如果是mobilenet或其他则需要手动更改
        if idx == 79:
            next_idx_list.append(84)
        elif idx == 91:
            next_idx_list.append(96)

        for next_idx in next_idx_list:
            # 待剪枝conv层的下一层
            next_conv = pruned_model.module_list[next_idx][0]
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            # matmul方法中如果第一个参数或者第二个参数是1维的,它会提升该参数为矩阵(根据另一个参数维数,给该参数增加一个为1的维数).
            # 矩阵相乘之后会将为1的维数去掉,所以这里可以不用reshape扩维以及缩维
            # 即(64,32).matmul(32) -> (64,32).matmul(32,1) -> (64,1) -> (64)
            # offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            offset = conv_sum.matmul(activation)
            if next_idx in CBL_idx:
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)
            else:
                next_conv.bias.data.add_(offset)
        # 将下一层吸收了当前bn层中的β值后,将当前层的β值置为0.不然可能会对后续造成影响.
        bn_module.bias.data.mul_(mask)

    return pruned_model


def parse_module_defs(module_defs):
    """
    该方法就是一句网路结构返回 完整conv,普通conv,以及最终需要剪枝的索引
    :param module_defs: 网络结构的配置文件
    :return: CBL_idx:   有bn_relu的conv层
             Conv_idx:  无bn_relu的conv层(yolo层前一层)
             prune_idx: 最终参与剪枝的层
    """
    CBL_idx = []  # 有bn可以剪枝的conv层
    Conv_idx = []  # 没有bn不能剪枝的conv层(yolo前一层)
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            ignore_idx.add(i - 1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
        elif module_def['type'] == 'upsample':
            ignore_idx.add(i-1)
        # 这一步是多余的
        # elif module_defs[identity_idx]['type'] == 'shortcut':
        #     ignore_idx.add(identity_idx - 1)
    # 这个是从 有bn可以剪枝的conv层中 剔除 shortcut层中的其实末尾conv层
    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]
    # 总结一下,哪些conv层无法剪枝 1.yolo层前一层。2.shortcut的起始和末尾层无法剪枝3.upsample层前一层无法剪枝
    return CBL_idx, Conv_idx, prune_idx


def gather_bn_weights(module_list, prune_idx):
    """
    该方法相当于将稀疏化后的模型权重上待剪枝部分的bn上的权重绝对值拷贝下来并拉伸到一维的tensor上来
    :param module_list: 网络结构单元的列表
    :param prune_idx: 待剪枝的conv索引
    :return: 包含所有待剪枝的bn中|γ|值的一维tensor,长度为所有待剪枝conv层通道数总和
    """
    # 是一个包含每个待剪枝的conv层的通道数量
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    # 初始化一个默认值为 0, 总长度为 所有待剪枝conv层的通道数量和
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    # 将所有待剪枝的conv层后面的bn层中的γ绝对值化 然后赋值给 bn_weights中相对应的位置上
    for idx, size in zip(prune_idx, size_list):
        # 位置与长度与conv层在module_list中一致
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size
    # 至此 bn_weights 包含了所有待剪枝conv层后面的bn层中所有的 |γ|
    return bn_weights


def get_bn_mask(bn_module, thres):
    # 根据阈值获取bn层的γ值的掩膜,1为大于阈值,0为小于阈值
    thres = thres.cuda()
    mask = bn_module.weight.data.abs().ge(thres).float()

    return mask
