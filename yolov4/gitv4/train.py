import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.yolo4 import YoloBody
from nets.yolo_training import Generator, YOLOLoss
from utils.dataloader import YoloDataset, yolo_dataset_collate
from torch.optim.lr_scheduler import LambdaLR
import math


classes_path = 'train_parameters/classes.txt'
anchors_path = 'train_parameters/yolo_anchors.txt'
train_anno_path = 'train_parameters/train_anno.txt'
Cuda = True
Use_Data_Loader = True
normalize = False
input_shape = (608,608)
mosaic = True
Cosine_lr = False
my_lr = False
resume = True
resume_model_path = 'v4v5runs/Epoch394-Total_Loss19.3836-Val_Loss14.1627.pth'
smoooth_label = 0
#8.25 lr=5e-4 ~ 5e-5 T=5 Cosine_lr = True
lr = 1e-3
Batch_size = 10
Init_Epoch = 400
Freeze_Epoch = 1000 #总共epoch次数



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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_learning_rate(optimizer, epoch, start_lr):
    if epoch < 3 : # warm up
        lr = 0.0001
    elif epoch < 30:
        lr = start_lr * pow(0.5,(epoch//10))
    elif epoch < 400:
        # lr = 5e-5 ~ 5e-6
        # lr = 1e-4 ~
        # 150epoch后lr*0.25
        index = epoch % 10
        b      = [1  , 0.8, 0.6, 0.4, 0.2, 0.1, 0.2, 0.4, 0.6,0.8]
        angles = [90 , 75 , 60 , 45 , 30 , 15 , 30 , 45 , 60 , 70]
        lr = 5e-4 * (math.sin(math.radians(angles[index]))*b[index])
        lr = lr if epoch < 250 else lr * 0.5
    else:
        lr = 5e-5

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        adjust_learning_rate(optimizer, epoch, lr)
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            # ----------------------#

            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            # ----------------------#
            outputs = net(images)
            losses = []
            num_pos_all = 0
            # ----------------------#
            # ----------------------#
            for i in range(3):
                # print("targets: ", targets)
                loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            # ----------------------#
            # ----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(3):
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    if epoch % 1 == 0:
        torch.save(model.state_dict(), 'v4v5runs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


if __name__ == '__main__':
    #                         model parameters\config
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    model = YoloBody(len(anchors[0]), num_classes)
    if resume:
        model.load_state_dict(torch.load(resume_model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.train()
    if Cuda:
        net = net.cuda()
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, \
                                    (input_shape[1], input_shape[0]), smoooth_label, Cuda, normalize))

    with open(train_anno_path) as f:
        lines = f.readlines()

    val_split = 0.1
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split )
    num_train = int((len(lines) - num_val)  )

    # if True:

    optimizer = optim.Adam(net.parameters(),lr)

    if Cosine_lr:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    elif my_lr:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.5)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=1)



    train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
    val_dataset = YoloDataset(lines[num_train:num_train+num_val], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=0, pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=0,pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset_collate)

    epoch_size = max(1, num_train//Batch_size) # epoch
    epoch_size_val = num_val//Batch_size

    for param in model.backbone.parameters():
        param.requires_grad = False

    for epoch in range(Init_Epoch,Freeze_Epoch):
        # net,loss,epoch一共有多少比赛,每个epoch要跑几个圈，train/val_data, freeze_Epoch,Cuda

        fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
        lr_scheduler.step()
