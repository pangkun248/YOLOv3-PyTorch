import torch
from util import *
from model import *
from datasets import *
import copy


# 转换pytorch权重模型致libtorch可以调用的模型
# if __name__ == "__main__":
#     model_name = 'yolov3'
#     model = Mainnet('D:\py_pro\YOLOv3-PyTorch\yolo_cfg\\'+model_name+'.cfg').cuda()
#     model.load_state_dict(torch.load('D:\py_pro\YOLOv3-PyTorch\weights\wenyi\\yolov3_ep87-map70.33-loss0.06912.weights'))
#     print("网络权重加载成功.")
#     example = torch.rand(1, 3, 320, 320).cuda() # 注意,我这里导出的是CUDA版的模型`
#     model = model.eval()
#     traced_script_module = torch.jit.trace(model, example)
#     output = traced_script_module(torch.ones(1,3,320,320).cuda())
#     traced_script_module.save('D:\py_pro\YOLOv3-PyTorch\weights\yolov3.pt')
#     print(output)

def init_anchors(img_size=800,reduce=16):
    # g是grid的个数
    g=img_size//reduce
    # anchor长宽比例 1:0.5 1:1 1:2
    ratios = (0.5, 1, 2)
    # anchor面积种类 16*8 16*16 16*32
    anchor_scale = (8, 16, 32)
    # grid的中心坐标
    grid_center_x = torch.arange(0,g*reduce,reduce).repeat(g, 1).reshape([ g, g])+8
    grid_center_y = torch.arange(0,g*reduce,reduce).repeat(g, 1).t().reshape([ g, g])+8
    anchors = torch.zeros(g*g*9,4)
    print(anchors.shape)
init_anchors()