"""
   Additional evaluation metrics for a 
"""
import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.structures import Instances


def get_predicted_boxes(predictor, image_path):
    # Read the image
    img = read_image(image_path, format="BGR")

    # Use the predictor to make predictions
    with torch.no_grad():
        outputs = predictor(img)

    # Get the predicted boxes in x1, y1, x2, y2 format
    predicted_boxes = [
        box.tolist() for box in outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
    ]

    return predicted_boxes


def calculate_iou(bbox1, bbox2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    Bounding box format: [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox1
    x1_gt, y1_gt, x2_gt, y2_gt = bbox2

    # Calculate intersection area
    intersection_area = max(0, min(x2, x2_gt) - max(x1, x1_gt)) * max(
        0, min(y2, y2_gt) - max(y1, y1_gt)
    )

    # Calculate union area
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def eval(predictor, data, iou_threshold=0.75):
    true_positives = 0  # correctly localized mice
    false_positives = 0  # hallucinated mice
    false_negatives = 0  # unrecovered mice
    ground_truth_detections = 0
    for img_annotation in data:
        image = img_annotation["file_name"]
        predicted_bboxes = get_predicted_boxes(predictor, image)
        gt_bboxes = [instance["bbox"] for instance in img_annotation["annotations"]]
        ground_truth_detections += len(gt_bboxes)
        matched_gt_boxes = []
        for pred_box in predicted_bboxes:
            match_found = False
            for gt_box in gt_bboxes:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold and gt_box not in matched_gt_boxes:
                    true_positives += 1
                    match_found = True
                    matched_gt_boxes.append(gt_box)
                    break
            if not match_found:
                false_positives += 1
        false_negatives += len(gt_bboxes) - len(matched_gt_boxes)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    false_negative_rate = false_negatives / ground_truth_detections if ground_truth_detections > 0 else 0
    
    # Return metrics in a dictionary
    metrics = {
        "precision": precision*100,
        "recall": recall*100,
        "f1_score": f1_score*100,
        "false_negative_rate": false_negative_rate*100
    }
    return metrics
