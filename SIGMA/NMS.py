import json
import torch

def nms(boxes, scores, iou_threshold=0.9):
    """
    Apply Non-Maximum Suppression (NMS) to remove redundant bounding boxes.
    """
    if len(boxes) == 0:
        return [], []

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < iou_threshold]

    return [boxes[idx].tolist() for idx in keep], [scores[idx].item() for idx in keep], keep

input_path = "/storageStudents/nguyenvd/Huytnc/SIGMA/experiments/sigma/uit_drone_to_foggy_vgg16/inference/uit_drone_foggy_test_cocostyle/bbox.json"
output_path = "/storageStudents/nguyenvd/Huytnc/SIGMA/experiments/sigma/uit_drone_to_foggy_vgg16/inference/uit_drone_foggy_test_cocostyle/filtered_bbox.json"

with open(input_path, "r") as f:
    data = json.load(f)

image_dict = {}
for item in data:
    image_id = item["image_id"]
    if image_id not in image_dict:
        image_dict[image_id] = []
    image_dict[image_id].append(item)

filtered_results = []
for image_id, items in image_dict.items():
    boxes = [item["bbox"] for item in items]
    scores = [item["score"] for item in items]
    categories = [item["category_id"] for item in items]

    filtered_boxes, filtered_scores, keep = nms(boxes, scores, iou_threshold=0.3)

    for idx in keep:
        filtered_results.append({
            "image_id": image_id,
            "category_id": categories[idx], 
            "bbox": filtered_boxes[keep.index(idx)],  
            "score": filtered_scores[keep.index(idx)]  
        })

with open(output_path, "w") as f:
    json.dump(filtered_results, f, indent=4)

print(f"Da luu vao file: {output_path}")
