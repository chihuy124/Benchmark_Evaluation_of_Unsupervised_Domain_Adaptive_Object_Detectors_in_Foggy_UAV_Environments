from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train3/weights/best.pt')  # load a custom model
engine_path = "runs/detect/train3/weights/CNNPPP_best.pt"
# Validate the model
metrics = model.val(data="VOC.yaml", Fog_engine_path = engine_path)  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category