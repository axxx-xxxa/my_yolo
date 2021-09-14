'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
import glob

from PIL import Image

from yolo_ori import YOLO
import argparse

yolo = YOLO()



if __name__=="__main__":
    img_pathss = []
    img_path1 = glob.glob("data/1/img/*")
    img_path2 = glob.glob("data/2/img/*")
    img_path3 = glob.glob("data/3/img/*")
    img_path4 = glob.glob("data/4/img/*")
    img_path5 = glob.glob("data/5/img/*")
    img_pathss.append(img_path1)
    img_pathss.append(img_path2)
    img_pathss.append(img_path3)
    img_pathss.append(img_path4)
    img_pathss.append(img_path5)

    img = 'test.png'
    # try:
    #     image = Image.open(img)
    # except:
    #     print('Open Error! Try again!')
    # else:
    #     r_image = yolo.detect_image(image)
    #     r_image.save(f"test_res.jpg")
    for j,img_paths in enumerate(img_pathss):
        for i,img_path in enumerate(img_paths):
            if i % 100 == 0:
                try:
                    image = Image.open(img_path)
                except:
                    print('Open Error! Try again!')
                else:
                    r_image = yolo.detect_image(image)
                    r_image.save(f"data_test_res/{j+1}/{i+1}.jpg")
