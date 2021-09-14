import onnx
import numpy as np
from onnx_tf.backend import prepare

model = onnx.load('best.onnx')  # yolov5 pt模型转换得到的onnx模型
tf_model = prepare(model)
tf_model.export_graph('best_save_model')  # onnx模型转换为tfserving的savedmode模型
