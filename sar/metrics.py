"""
Jal-Rakshak — Segmentation Evaluation Metrics
===============================================
Compute IoU, Dice, Precision, Recall, F1, and boundary metrics
for comparing segmentation masks.

Used for:
1. Comparing YOLO vs classical method masks (validation)
2. Comparing any mask vs ground-truth (when available)
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass
class SegmentationMetrics:
    """Complete set of segmentation evaluation metrics."""
    iou: float            # Intersection over Union (Jaccard)
    dice: float           # Sørensen-Dice coefficient
    precision: float      # TP / (TP + FP)
    recall: float         # TP / (TP + FN)
    f1: float             # 2 * precision * recall / (precision + recall)
    boundary_f_score: float  # Boundary-based F-score
    true_positive_px: int
    false_positive_px: int
    false_negative_px: int
    true_negative_px: int

    def to_dict(self) -> dict:
        return {
            "iou": round(self.iou, 4),
            "dice": round(self.dice, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "boundary_f_score": round(self.boundary_f_score, 4),
            "true_positive_px": self.true_positive_px,
            "false_positive_px": self.false_positive_px,
            "false_negative_px": self.false_negative_px,
            "true_negative_px": self.true_negative_px,
        }


def compute_metrics(prediction: np.ndarray, reference: np.ndarray,
                    boundary_tolerance_px: int = 3) -> SegmentationMetrics:
    """
    Compute segmentation metrics between a prediction mask and a reference mask.

    Both masks should be binary (0 or non-zero).
    Neither mask is assumed to be ground truth — this computes agreement.

    Args:
        prediction: Binary prediction mask
        reference: Binary reference mask
        boundary_tolerance_px: Tolerance in pixels for boundary F-score

    Returns:
        SegmentationMetrics with all computed values
    """
    pred = (prediction > 0).astype(np.uint8)
    ref = (reference > 0).astype(np.uint8)

    # Ensure same shape
    if pred.shape != ref.shape:
        ref = cv2.resize(ref, (pred.shape[1], pred.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    tp = int(np.sum((pred == 1) & (ref == 1)))
    fp = int(np.sum((pred == 1) & (ref == 0)))
    fn = int(np.sum((pred == 0) & (ref == 1)))
    tn = int(np.sum((pred == 0) & (ref == 0)))

    # IoU / Jaccard
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0

    # Dice / Sørensen-Dice
    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp) / dice_denom if dice_denom > 0 else 0.0

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Boundary F-score
    boundary_f = _boundary_f_score(pred, ref, tolerance=boundary_tolerance_px)

    return SegmentationMetrics(
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        f1=f1,
        boundary_f_score=boundary_f,
        true_positive_px=tp,
        false_positive_px=fp,
        false_negative_px=fn,
        true_negative_px=tn,
    )


def _boundary_f_score(pred: np.ndarray, ref: np.ndarray,
                      tolerance: int = 3) -> float:
    """
    Compute boundary-based F-score.

    Extracts contours from both masks and measures how closely the
    prediction boundary matches the reference boundary.
    """
    # Extract boundaries using morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_boundary = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel)
    ref_boundary = cv2.morphologyEx(ref, cv2.MORPH_GRADIENT, kernel)

    if np.sum(pred_boundary) == 0 and np.sum(ref_boundary) == 0:
        return 1.0  # Both empty — perfect agreement
    if np.sum(pred_boundary) == 0 or np.sum(ref_boundary) == 0:
        return 0.0  # One empty, one not

    # Dilate boundaries by tolerance
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (2 * tolerance + 1, 2 * tolerance + 1))
    ref_dilated = cv2.dilate(ref_boundary, dilate_kernel)
    pred_dilated = cv2.dilate(pred_boundary, dilate_kernel)

    # Precision: fraction of prediction boundary within tolerance of reference
    pred_match = np.sum((pred_boundary > 0) & (ref_dilated > 0))
    pred_total = np.sum(pred_boundary > 0)
    b_precision = pred_match / pred_total if pred_total > 0 else 0.0

    # Recall: fraction of reference boundary within tolerance of prediction
    ref_match = np.sum((ref_boundary > 0) & (pred_dilated > 0))
    ref_total = np.sum(ref_boundary > 0)
    b_recall = ref_match / ref_total if ref_total > 0 else 0.0

    if b_precision + b_recall == 0:
        return 0.0

    return 2 * b_precision * b_recall / (b_precision + b_recall)


def create_overlay_visualization(image: np.ndarray,
                                 prediction: np.ndarray,
                                 reference: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create a color-coded overlay visualization.

    Colors:
    - Green: True Positive (both agree)
    - Red: False Positive (prediction only)
    - Blue: False Negative (reference only)
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    pred = (prediction > 0).astype(np.uint8)
    overlay = vis.copy()

    if reference is not None:
        ref = (reference > 0).astype(np.uint8)
        if ref.shape != pred.shape:
            ref = cv2.resize(ref, (pred.shape[1], pred.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        tp_mask = (pred == 1) & (ref == 1)
        fp_mask = (pred == 1) & (ref == 0)
        fn_mask = (pred == 0) & (ref == 1)

        overlay[tp_mask] = [0, 255, 0]    # Green = agreement
        overlay[fp_mask] = [0, 0, 255]    # Red = prediction only
        overlay[fn_mask] = [255, 0, 0]    # Blue = reference only
    else:
        overlay[pred == 1] = [0, 0, 255]  # Red = detection

    result = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
    return result
