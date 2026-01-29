import os
import json
import cv2
import matplotlib.pyplot as plt

# File paths
test_json_path = "/storageStudents/nguyenvd/Huytnc/dataset/uit_drone/test.json"
bbox_json_path = "/storageStudents/nguyenvd/Huytnc/SIGMA/experiments/sigma/NMS_uit_drone_to_foggy_vgg16/inference/uit_drone_foggy_test_cocostyle/bbox.json"  
image_folder = "/storageStudents/nguyenvd/Huytnc/dataset/uit_drone/images/test_fog/"  # Folder containing original images
output_folder = "/storageStudents/nguyenvd/Huytnc/SIGMA/experiments/sigma/NMS_uit_drone_to_foggy_vgg16/visualize"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load test.json and bbox.json files
with open(test_json_path, "r") as f:
    test_data = json.load(f)

with open(bbox_json_path, "r") as f:
    bbox_data = json.load(f)

# Create a mapping from image_id -> file_name
image_id_to_filename = {img["id"]: img["file_name"] for img in test_data["images"]}

# Create a mapping from category_id -> category name
category_id_to_name = {cat["id"]: cat["name"] for cat in test_data["categories"]}

# Define colors for different categories
colors = {
    0: (255, 0, 0),    # pedestrian - Blue
    1: (0, 255, 0),    # motor - Green
    2: (0, 0, 255),    # car - Red
    3: (255, 255, 0)   # bus - Yellow
}

# Initialize an empty map for all annotations
image_bbox_map = {}

# Process each annotation and group by image_id
for annotation in bbox_data:
    image_id = annotation["image_id"]

    # Add the annotation to the corresponding image_id entry
    if image_id not in image_bbox_map:
        image_bbox_map[image_id] = []
    
    image_bbox_map[image_id].append(annotation)

# Process each image and its bounding boxes
for image_id in image_id_to_filename.keys():
    # Check if there are annotations for this image_id
    if image_id not in image_bbox_map:
        continue

    annotations = image_bbox_map[image_id]
    image_filename = image_id_to_filename[image_id]
    image_path = os.path.join(image_folder, image_filename)

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # Read the image only once
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # Draw all bounding boxes on the image
    for annotation in annotations:
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        score = annotation["score"]

        # Convert bbox from (x, y, width, height) to (x1, y1, x2, y2)
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h

        # Draw the bounding box
        color = colors.get(category_id, (255, 255, 255))  # Default to white if category is unknown
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label the bounding box (category + score)
        label = f"{category_id_to_name[category_id]}: {score:.2f}"

        # Background rectangle for better visibility
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, thickness=-1)

        # Put text on top of background
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Save the annotated image to the output folder
    output_path = os.path.join(output_folder, image_filename)
    cv2.imwrite(output_path, image)

    print(f"Processed: {output_path}")

print("Bounding box visualization completed!")
