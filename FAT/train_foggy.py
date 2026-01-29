from ultralytics import YOLO
import os
os.environ["WANDB_DISABLED"] = "true"
model = YOLO("yolov8s.yaml")  
model = YOLO("runs/detect/train2/weights/best.pt")  
model.train(data="VOC.yaml", cfg="ultralytics/yolo/cfg/default.yaml", cfg_t="ultralytics/yolo/cfg/default_teacher.yaml")  
metrics = model.val()  
