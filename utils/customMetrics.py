import torch
import numpy as np
import tqdm
from typing import Dict

def calculate_iou(boxes1, boxes2):
    """
    Calculate Intersection over Union (IoU) between two sets of boxes
    
    Args:
        boxes1 (torch.Tensor): First set of bounding boxes (N, 4)
        boxes2 (torch.Tensor): Second set of bounding boxes (M, 4)
    
    Returns:
        torch.Tensor: IoU matrix of shape (N, M)
    """
    # Convert [x_min, y_min, x_max, y_max] format
    # Compute coordinates of intersection rectangles
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    
    # Compute areas of intersection
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Compute areas of boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute union
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-8)
    
    return iou

def compute_average_precision(recalls, precisions):
    """
    Compute Average Precision using the VOC method
    
    Args:
        recalls (np.ndarray): Recall values
        precisions (np.ndarray): Precision values
    
    Returns:
        float: Average Precision
    """
    # Convert inputs to numpy arrays if they aren't already
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    # Add sentinel values for proper interpolation
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))
    
    # Compute the maximum precision for recall r and all recalls greater than r
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    
    # Find indices where recall changes
    i = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Calculate area under precision-recall curve
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    
    return ap

def calculate_map(predictions, ground_truth, num_classes, iou_threshold=0.5):
    """
    Calculate Mean Average Precision (mAP)
    
    Args:
        predictions (List[Dict]): List of prediction dictionaries
        ground_truth (List[Dict]): List of ground truth dictionaries
        num_classes (int): Number of classes
        iou_threshold (float): IoU threshold for positive match
    
    Returns:
        Dict: mAP metrics
    """
    # Ensure inputs are on the same device and convert to list if needed
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    
    # Containers for AP of each class
    class_aps = []
    
    # Iterate through each class
    for cls in range(1, num_classes + 1):  # Assuming class indices start from 1
        class_preds = []
        class_gts = []
        
        # Collect predictions and ground truth for this class
        for pred, gt in zip(predictions, ground_truth):
            # Filter predictions and ground truth for current class
            pred_mask = pred['labels'] == cls
            gt_mask = gt['labels'] == cls
            
            class_preds.append({
                'boxes': pred['boxes'][pred_mask],
                'scores': pred['scores'][pred_mask] if 'scores' in pred else torch.ones_like(pred['labels'][pred_mask], dtype=torch.float)
            })
            
            class_gts.append({
                'boxes': gt['boxes'][gt_mask],
                'labels': gt['labels'][gt_mask]
            })
        
        # Compute AP for this class
        class_ap = compute_class_ap(class_preds, class_gts, iou_threshold)
        class_aps.append(class_ap)
    
    # Compute mAP metrics
    valid_aps = [ap for ap in class_aps if not np.isnan(ap)]
    map_metrics = {
        'map': np.mean(valid_aps) if valid_aps else 0.0,
        'map_50': np.mean(valid_aps) if valid_aps else 0.0,  # Same as standard mAP@0.5
        'map_75': np.mean([ap for ap in valid_aps if ap >= 0.75]) if valid_aps else 0.0  # mAP at stricter IoU
    }
    
    return map_metrics

def compute_class_ap(class_preds, class_gts, iou_threshold=0.5):
    """
    Compute Average Precision for a single class
    
    Args:
        class_preds (List[Dict]): Predictions for a specific class
        class_gts (List[Dict]): Ground truth for a specific class
        iou_threshold (float): IoU threshold for positive match
    
    Returns:
        float: Average Precision for the class
    """
    # Collect all predictions across images
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    
    for preds, gts in zip(class_preds, class_gts):
        all_pred_boxes.append(preds['boxes'])
        all_pred_scores.append(preds['scores'])
        all_gt_boxes.append(gts['boxes'])
    
    # Concatenate predictions and ground truth
    all_pred_boxes = torch.cat(all_pred_boxes) if len(all_pred_boxes) > 0 else torch.empty((0, 4))
    all_pred_scores = torch.cat(all_pred_scores) if len(all_pred_scores) > 0 else torch.empty(0)
    all_gt_boxes = torch.cat(all_gt_boxes) if len(all_gt_boxes) > 0 else torch.empty((0, 4))
    
    # If no predictions or ground truth, return 0
    if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
        return 0.0
    
    # Sort predictions by confidence score
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    sorted_pred_boxes = all_pred_boxes[sorted_indices]
    
    # Track matched ground truth
    gt_matched = torch.zeros(len(all_gt_boxes), dtype=torch.bool)
    
    # Initialize precision and recall lists
    precisions = []
    recalls = []
    
    tp = 0  # True positives
    fp = 0  # False positives
    
    # Process each prediction
    for pred_box in sorted_pred_boxes:
        # Compute IoU with all ground truth boxes
        ious = calculate_iou(pred_box.unsqueeze(0), all_gt_boxes)
        ious = ious.squeeze(0)
        
        # Find best match
        max_iou, max_idx = torch.max(ious, dim=0)
        
        # Check if match is valid
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            gt_matched[max_idx] = True
            tp += 1
        else:
            fp += 1
        
        # Compute precision and recall
        precision = tp / (tp + fp)
        recall = tp / len(all_gt_boxes)
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Convert to numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Compute Average Precision
    if len(precisions) > 0:
        ap = compute_average_precision(recalls, precisions)
    else:
        ap = 0.0
    
    return ap