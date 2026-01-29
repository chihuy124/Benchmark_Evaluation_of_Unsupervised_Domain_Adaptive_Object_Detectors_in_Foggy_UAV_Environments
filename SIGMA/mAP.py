import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import auc

# JSON file paths
input_path = "/storageStudents/nguyenvd/Huytnc/SIGMA/experiments/sigma/uit_drone_to_foggy_vgg16/inference/uit_drone_foggy_test_cocostyle/filtered_bbox.json"
ground_truth_path = "/storageStudents/nguyenvd/Huytnc/dataset/uit_drone/test.json"

# Read prediction file
with open(input_path, "r") as f:
    try:
        predictions = json.load(f)
        if isinstance(predictions, str):  # If JSON is stored as a string
            predictions = json.loads(predictions)
    except json.JSONDecodeError:
        print("Error: bbox.json is not a valid JSON.")
        exit()

# Read ground truth file
with open(ground_truth_path, "r") as f:
    try:
        ground_truth_data = json.load(f)
    except json.JSONDecodeError:
        print("Error: test.json is not a valid JSON.")
        exit()

# Extract annotations and categories from COCO format
annotations = ground_truth_data["annotations"]
categories = {cat["id"]: cat["name"] for cat in ground_truth_data["categories"]}

# Organize ground truth by category and image_id
ground_truth_by_class = defaultdict(lambda: defaultdict(list))
for ann in annotations:
    category_id = ann["category_id"]
    image_id = ann["image_id"]
    ground_truth_by_class[category_id][image_id].append(ann["bbox"])

# Organize predictions by category and image_id
predictions_by_class = defaultdict(lambda: defaultdict(list))
for pred in predictions:
    category_id = pred["category_id"]
    image_id = pred["image_id"]
    predictions_by_class[category_id][image_id].append(pred)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def compute_ap(ground_truth, predictions, iou_threshold=0.5):
    """Compute Average Precision (AP) for each class."""
    all_gt_bboxes = []
    all_pred_bboxes = []
    all_scores = []

    # Match predictions and ground truth by image_id
    for image_id, pred_list in predictions.items():
        gt_bboxes = ground_truth.get(image_id, [])  # Get GT for this image_id

        for pred in pred_list:
            all_pred_bboxes.append(pred["bbox"])
            all_scores.append(pred["score"])
            all_gt_bboxes.append(gt_bboxes)

    if not all_gt_bboxes or not all_pred_bboxes:
        return 0.0  # If no bounding boxes, AP = 0

    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_pred_bboxes = [all_pred_bboxes[i] for i in sorted_indices]
    sorted_scores = [all_scores[i] for i in sorted_indices]
    sorted_gt_bboxes = [all_gt_bboxes[i] for i in sorted_indices]

    tp = np.zeros(len(sorted_pred_bboxes))
    fp = np.zeros(len(sorted_pred_bboxes))
    matched_gt = set()

    for i, pred_box in enumerate(sorted_pred_bboxes):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(sorted_gt_bboxes[i]):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and (i, best_gt_idx) not in matched_gt:
            tp[i] = 1
            matched_gt.add((i, best_gt_idx))
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / max(len(all_gt_bboxes), 1)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # If precision or recall is empty, AP = 0
    if len(recall) == 0 or len(precision) == 0:
        return 0.0
    
    return auc(recall, precision)

# Compute AP for each class
ap_per_class = {}
for category_id in set(predictions_by_class.keys()).union(set(ground_truth_by_class.keys())):
    ap = compute_ap(ground_truth_by_class[category_id], predictions_by_class[category_id])
    ap_per_class[category_id] = ap

# Compute mean Average Precision (mAP)
mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0

# Display results
print("AP per class:")
for category_id, ap in ap_per_class.items():
    class_name = categories.get(category_id, f"Class {category_id}")
    print(f"{class_name}: AP = {ap:.4f}")

print(f"\nMean Average Precision (mAP) = {mAP:.4f}")
