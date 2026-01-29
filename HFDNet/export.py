import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# pip install onnx onnxsim onnxruntime-gpu 
# cpu版本 onnxruntime 

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples
# 导出onnx,复杂算子可能导出不成功
if __name__ == '__main__':
    model = YOLO('runs/train/exp2/weights/best.pt')
    model.export(format='onnx', simplify=True, opset=13) # opset 跟torch版本有关系，官网上有版本对应，例如 1.13.1--17