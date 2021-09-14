'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
import glob
from PIL import Image
import xml.etree.ElementTree as ET
from yolo import YOLO
import numpy as np
import random


yolo = YOLO()

if __name__=="__main__":

    # img = 'test.jpg'
    xml_root = 'boltdata/xml'
    xml_list = glob.glob(f'{xml_root}/*')
    random.seed(1)
    random.shuffle(xml_list)
    box_correct = 0
    predict_right = 0
    all_num = 0
    for i,xml_name in enumerate(xml_list[26:28]):
        # try:
        #     tree = ET.parse(xml_name)
        #     root = tree.getroot()
        #     filename = tree.findtext('./filename')
        #     coors = []
        #     print("-" * 100)
        #     print(filename)
        #     for obj in tree.iter('object'):
        #         xmin = int(float(obj.findtext('bndbox/xmin')))
        #         ymin = int(float(obj.findtext('bndbox/ymin')))
        #         xmax = int(float(obj.findtext('bndbox/xmax')))
        #         ymax = int(float(obj.findtext('bndbox/ymax')))
        #         xmin = np.float64(xmin)
        #         ymin = np.float64(ymin)
        #         xmax = np.float64(xmax)
        #         ymax = np.float64(ymax)
        #         coor = [xmin,ymin,xmax,ymax]
        #         coors.append(coor)
        #         all_num+=1
        #
        #     image = Image.open(f'boltdata/image/{filename}')
        #     if len(coors) == 0:
        #         continue
        #         # r_image = yolo.detect_image(image)
        #         # r_image.save(f"test/{i}.jpg")
        #     else:
        #         r_image, correct, cut_img  = yolo.detect_image(image, correct, coors)
        #         r_image.save(f"test/{i}.jpg")
        #         cut_img.save(f"test/cut{i}.jpg")
        #     print("-"*100)
        #     print("all_num = ",all_num)
        #     print("correct_num = ",correct)
        #     print("img_num = ",i)
        #
        # except:
        #     print("error")
        #     continue
        tree = ET.parse(xml_name)
        root = tree.getroot()
        filename = tree.findtext('./filename')
        coors = []
        gts = []
        print("-" * 100)
        print(filename)
        for obj in tree.iter('object'):
            gt = obj.findtext('name')
            xmin = int(float(obj.findtext('bndbox/xmin')))
            ymin = int(float(obj.findtext('bndbox/ymin')))
            xmax = int(float(obj.findtext('bndbox/xmax')))
            ymax = int(float(obj.findtext('bndbox/ymax')))
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            coor = [xmin, ymin, xmax, ymax]
            gts.append(gt)
            coors.append(coor)
            all_num += 1
        image = Image.open(f'boltdata/image/{filename}')
        if len(coors) == 0:
            continue
            # r_image = yolo.detect_image(image)
            # r_image.save(f"test/{i}.jpg")
        else:
            r_image, box_correct, predict_right, cut_imgs = yolo.detect_image(image, box_correct,predict_right, coors, gts)
            r_image.save(f"test/{i}.jpg")
            #for j,cut_img in enumerate(cut_imgs):
            #    cut_img.save(f"cut/{i}_cut_{j}.jpg")
        print("-" * 100)
        print("all_num = ", all_num)
        print("box_correct_num = ", box_correct)
        print("predict_right_num = ", predict_right)
        print("img_num = ", i)
#     r_image.show()
        # except:
            # print('Open Error! Try again!')
        # else:
        #     r_image = yolo.detect_image(image)
        #     r_image.show()
