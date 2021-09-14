# 用于将 .pth 模型转为 .onnx 供 c++ 调用
from yolo import YOLO
import onnxruntime
import torch
from nets.yolo4 import YoloBody


weight_file = 'logs/Epoch28-3Classes-newAnchors-0417-Total_Loss7.9950-Val_Loss5.9606.pth'
image_path = 'data/dog.jpg'
onnx_file_name = 'model/yolov4_3classes.onnx'

IN_IMAGE_H = 416
IN_IMAGE_W = 416

model = YoloBody(3, 3).eval()

model_dict = model.state_dict()
pretrained_dict = torch.load(weight_file, map_location=torch.device('cpu'))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

input_names = ["input"]   #onnx输入接口名字，需要与模型输入结果对应
output_names = ['boxes', 'confs']   #onnx输出接口的名字 ，需要与模型输出结果对应
x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
#dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}


torch.onnx.export(model,
                  x,
                  onnx_file_name,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=input_names, 
                  output_names=output_names,
                  #dynamic_axes=dynamic_axes
                  )

