from ultralytics import YOLO

# Load a model
model = YOLO("/home/huytnc/LuuTru/FA_Teather/runs/detect/train3/weights/best.pt")  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model("/home/huytnc/LuuTru/FA_Teather/tests/AIUit000062_0p00_0p20_000479.jpg")  # predict on an image