from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model_path = "/home/huytnc/LuuTru/HFDNet/runs/train/v8/s2c/weights/best.pt"
image_path = "/home/huytnc/LuuTru/HFDNet/tests/AIUit000062_0p00_0p20_000479.jpg"
output_path = "/home/huytnc/LuuTru/HFDNet/tests/vis_AIUit000062_0p00_0p20_000479.jpg"


model = YOLO(model_path)

results = model(image_path)

for result in results:

    annotated_frame = result.plot()

    cv2.imwrite(output_path, annotated_frame)

print(f"{output_path}")