import random
import glob
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np


classes = ['head','person','cellphone','lamp']
colors = [(255,0,0),(255,255,0),(0,255,255),(0,255,0)]

def random_test(train_txt_path, random_num = False):
    with open(train_txt_path, 'r') as f:
        lines = f.readlines()
    num = random_num if random_num else len(lines)
    random.shuffle(lines)
    img_coors = [line.split() for line in lines]
    for i, img_coor in enumerate(img_coors[:num]):
        try:
            img_path = img_coor[0]
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            coors = img_coor[1:]

            for co in coors:
                co = [int(x) for x in co.split(',')]
                coor = [(co[0], co[1]), (co[2], co[3])]
                gt = classes[co[4]]
                color = colors[co[4]]
                cv2.rectangle(img, coor[0], coor[1], color, 2, 2)
                cv2.putText(img, gt, coor[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            name = f"random_test/{i}.jpg"
            cv2.imencode('.jpg', img)[1].tofile(name)
        except:
            print(img_path)




if __name__ == '__main__':
    try:
        os.makedirs("random_test")
    except OSError:
        pass

    train_txt_path = 'train_anno.txt'
    # random_num = 59

    random_test(train_txt_path)



