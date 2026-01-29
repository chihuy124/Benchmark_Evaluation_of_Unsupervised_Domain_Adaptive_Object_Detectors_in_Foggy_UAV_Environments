import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# Hook function to capture feature map
def hook_fn(module, input, output):
    global feature_maps
    feature_maps.append(output)

# Load YOLOv8 model

model = YOLO("yolov8s.yaml")

print(model)

# Create a hook to capture feature maps from the first convolutional layer
# You can modify this to capture from other layers as needed
hook_layer = model  # Select the first Conv layer
hook_layer.register_forward_hook(hook_fn)

# Load and preprocess image
img_path = "/home/huytnc/LuuTru/HFDNet/fog_1.jpg"
img = Image.open(img_path)
net_input_size = (640, 640)  # Resize to input size of YOLOv8 model
frame = img.resize(net_input_size)

# Transform to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
frame = transform(frame).unsqueeze(0)  # Add batch dimension
frame = frame.to(model.device)  # Move to the same device as model

# Run inference to capture feature maps
with torch.no_grad():
    _ = model(frame)  # Forward pass through the model

# Visualize feature maps
if feature_maps:
    for idx, fm in enumerate(feature_maps):
        fm = fm[0].cpu().detach().numpy()  # Get the first batch, detach from the graph
        print(f"Shape of fm[0] for layer {idx}: {fm.shape}")

        # Visualize the first feature map from the batch
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(fm[0], cmap='jet')  # Display the first feature map in the batch
        ax.axis('off')
        plt.title(f"Feature Map of First Filter from Layer {idx}")
        plt.show()
