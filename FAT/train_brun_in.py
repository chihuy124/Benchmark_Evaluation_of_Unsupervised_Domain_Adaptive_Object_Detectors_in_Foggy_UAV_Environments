from ultralytics import YOLO
import os
os.environ["WANDB_DISABLED"] = "true"
# Load a model
model = YOLO('yolov8s.yaml')  # build a new model from YAML
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='VOC.yaml',cfg="ultralytics/yolo/cfg/default_burn_in.yaml",train_type = "burn_in")