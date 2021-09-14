import torch
import torchvision
from nets.yolo4 import YoloBody
import numpy as np


classes_path = 'train_parameters/classes.txt'
anchors_path = 'train_parameters/yolo_anchors.txt'
train_anno_path = 'train_parameters/train_anno.txt'

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]


if __name__ == '__main__':
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    model = YoloBody(len(anchors[0]), num_classes)
    model.load_state_dict(torch.load("v4.pth"))#保存的训练模型
    model.eval()#切换到eval（）
    example = torch.rand(1, 3, 320, 480)#生成一个随机输入维度的输入
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("v4.pt")
