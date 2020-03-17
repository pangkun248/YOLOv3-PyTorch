import torch
from util import *
from model import *
from datasets import *
import cv2 as cv
from PIL import Image
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
