import torch
import numpy as np
from typing import Dict, List

def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes
    """
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    return intersection / (union + 1e-8)

def compute_average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using PASCAL VOC method
    """
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))
    
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    
    i = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])

def calculate_map(
    predictions: List[Dict],
    ground_truth: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5  # PASCAL VOC standard
) -> Dict[str, float]:
    """
    Calculate PASCAL VOC-style mAP@0.5 (no multi-threshold calculations)
    """
    class_aps = []
    
    for cls in range(1, num_classes + 1):
        class_preds = []
        class_gts = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_mask = pred['labels'] == cls
            gt_mask = gt['labels'] == cls
            
            class_preds.append({
                'boxes': pred['boxes'][pred_mask],
                'scores': pred['scores'][pred_mask] if 'scores' in pred else torch.ones_like(pred['labels'][pred_mask])
            })
            
            class_gts.append({
                'boxes': gt['boxes'][gt_mask],
                'labels': gt['labels'][gt_mask]
            })
        
        # Compute AP for this class
        all_pred_boxes = torch.cat([p['boxes'] for p in class_preds]) if any(len(p['boxes']) > 0 for p in class_preds) else torch.empty((0, 4))
        all_pred_scores = torch.cat([p['scores'] for p in class_preds]) if any(len(p['scores']) > 0 for p in class_preds) else torch.empty(0)
        all_gt_boxes = torch.cat([g['boxes'] for g in class_gts]) if any(len(g['boxes']) > 0 for g in class_gts) else torch.empty((0, 4))

        if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
            class_aps.append(0.0)
            continue

        sorted_indices = torch.argsort(all_pred_scores, descending=True)
        sorted_pred_boxes = all_pred_boxes[sorted_indices]
        gt_matched = torch.zeros(len(all_gt_boxes), dtype=torch.bool)
        
        tp, fp = 0, 0
        precisions, recalls = [], []
        
        for pred_box in sorted_pred_boxes:
            ious = calculate_iou(pred_box.unsqueeze(0), all_gt_boxes).squeeze(0)
            max_iou, max_idx = torch.max(ious, dim=0)
            
            if max_iou >= iou_threshold and not gt_matched[max_idx]:
                gt_matched[max_idx] = True
                tp += 1
            else:
                fp += 1
                
            precisions.append(tp / (tp + fp + 1e-8))
            recalls.append(tp / (len(all_gt_boxes) + 1e-8))
        
        ap = compute_average_precision(np.array(recalls), np.array(precisions))
        class_aps.append(ap)
    
    valid_aps = [ap for ap in class_aps if not np.isnan(ap)]
    return {'map': np.mean(valid_aps) if valid_aps else 0.0}