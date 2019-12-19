import torch
from util import *
from model import Mainnet
from datasets import *

if __name__ == "__main__":
    # a = torch.Tensor(2)
    # b = torch.tensor(2)
    # print(a)
    # print(b)
    a = torch.Tensor(3,4).cuda().fill_(1)
    b = [False,False,False,]
    print(a[b,1:2].sum())
    # model_name = 'yolov3-m'
    # model = Mainnet('D:\py_pro\YOLOv3-PyTorch\yolo_cfg\\'+model_name+'.cfg').cuda()
    # model.load_state_dict(torch.load('D:\py_pro\YOLOv3-PyTorch\weights\wenyi\\yolov3-m_ep61-map77.32-loss0.07387.weights'))
    # print("网络权重加载成功.")
    # example = torch.rand(1, 3, 320, 320).cuda() # 注意，我这里导出的是CUDA版的模型，因为我的模型是在GPU中进行训练的
    # model = model.eval()
    # traced_script_module = torch.jit.trace(model, example)
    # output = traced_script_module(torch.ones(1,3,320,320).cuda())
    # traced_script_module.save('D:\py_pro\YOLOv3-PyTorch\weights\yolov3-m.pt')
    # print(output)