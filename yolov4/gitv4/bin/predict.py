'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from PIL import Image

from yolo import YOLO
import argparse

yolo = YOLO()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    return parser.parse_args()


if __name__=="__main__":

    args = get_args()


    img = args.image
    img = 'test.jpg'
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.save("test_res.jpg")
