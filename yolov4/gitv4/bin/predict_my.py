
import cv2
import onnxruntime
import numpy as np
import PIL.Image as Image

input_shape = (416,416)

if __name__ == '__main__':
    # img = cv2.imread("test.jpg")
    h, w = input_shape
    img = Image.open("test.jpg")
    print(np.shape(img))
    img = img.resize((w, h), Image.ANTIALIAS)
    print(np.shape(img))
    x = np.expand_dims(np.array(img).astype(np.float32),0)
    print(np.shape(x.transpose(0,3,1,2)))
    x = x.transpose(0,3,1,2)
    print(np.shape(x))
    # x = np.ones([1, 3, 416, 416]).astype(np.float32)
    print(np.shape(x))
    # exit()
    # onnx
    session = onnxruntime.InferenceSession('model/yolov4_3classes.onnx')
    inputs = {session.get_inputs()[0].name: x}
    outs = session.run(None, inputs)
    ##### output (batch_size,3*(4+1+numclasses),13(24)(52),13(24)(52))
    #4不是坐标 是先验框的调整参数
    for out in outs:
        print(np.shape(out))
        # for ou in out:
        #     print(np.shape(ou))
    # print('onnx result is:', outs[0])
