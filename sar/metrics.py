"""
Jal-Rakshak — Segmentation Evaluation & Validation Metrics
===========================================================
Computes Jaccard / IoU, Sørensen-Dice, Precision, Recall, F1, and
Boundary F-score (BF score) for comparing segmentation masks.

Inspired by the reference repository (Dhanush024/Oil-Spill-Detection-in-SAR-images):
- segmentation_evaluation.m (jaccard, dice, bfscore)

IMPORTANT ARCHITECTURAL DISTINCTION:
- Prediction vs classical threshold mask is VALIDATION / CONSENSUS AGREEMENT.
- Prediction vs labeled ground-truth mask is EVALUATION / BENCHMARK ACCURACY.
Never conflate classical consensus with ground truth.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SegmentationMetrics:
    """Complete set of segmentation evaluation metrics."""
    iou: float               # Intersection over Union (Jaccard similarity)
    dice: float              # Sørensen-Dice coefficient: 2*TP / (2*TP + FP + FN)
    precision: float         # TP / (TP + FP)
    recall: float            # TP / (TP + FN)
    f1: float                # 2 * precision * recall / (precision + recall)
    boundary_f_score: float  # Boundary contour precision/recall F-score (BF score)
    true_positive_px: int
    false_positive_px: int
    false_negative_px: int
    true_negative_px: int
    evaluation_type: str = "general"  # "ground_truth" or "classical_validation"

    def to_dict(self) -> Dict[str, Any]:
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
            "evaluation_type": self.evaluation_type,
        }


def compute_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    boundary_tolerance_px: int = 3,
    evaluation_type: str = "general",
) -> SegmentationMetrics:
    """
    Compute segmentation metrics between a prediction mask and a reference mask.

    Both masks should be binary (0 or non-zero).

    Args:
        prediction: Binary prediction mask
        reference: Binary reference or ground-truth mask
        boundary_tolerance_px: Distance error threshold in pixels for BF score
        evaluation_type: Tag indicating 'ground_truth', 'classical_validation', or 'general'

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

    # Jaccard Index / IoU: TP / (TP + FP + FN)
    union = tp + fp + fn
    iou = tp / union if union > 0 else (1.0 if np.sum(ref) == 0 and np.sum(pred) == 0 else 0.0)

    # Sørensen-Dice Index: 2*TP / (2*TP + FP + FN)
    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp) / dice_denom if dice_denom > 0 else (1.0 if np.sum(ref) == 0 and np.sum(pred) == 0 else 0.0)

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if np.sum(pred) == 0 and np.sum(ref) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if np.sum(ref) == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Boundary F-score (BF score)
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
        evaluation_type=evaluation_type,
    )


def evaluate_against_ground_truth(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    boundary_tolerance_px: int = 3,
) -> SegmentationMetrics:
    """
    EVALUATION AGAINST GROUND TRUTH:
    Evaluates detector segmentation performance strictly against an authenticated
    expert-labeled ground-truth mask.

    Returns true benchmark accuracy (IoU, Dice, BF score).
    Do NOT call this with classical algorithmic outputs.
    """
    return compute_metrics(
        prediction=prediction,
        reference=ground_truth,
        boundary_tolerance_px=boundary_tolerance_px,
        evaluation_type="ground_truth",
    )


def evaluate_against_classical_reference(
    prediction: np.ndarray,
    classical_mask: np.ndarray,
    boundary_tolerance_px: int = 3,
) -> SegmentationMetrics:
    """
    VALIDATION AGAINST CLASSICAL REFERENCE:
    Measures algorithmic cross-validation agreement between the primary detector
    (YOLO) and classical physical SAR thresholding.

    This represents multi-signal consensus and look-alike verification,
    NOT ground-truth benchmark accuracy.
    """
    return compute_metrics(
        prediction=prediction,
        reference=classical_mask,
        boundary_tolerance_px=boundary_tolerance_px,
        evaluation_type="classical_validation",
    )


def _boundary_f_score(
    pred: np.ndarray,
    ref: np.ndarray,
    tolerance: int = 3,
) -> float:
    """
    Compute Boundary F-score (BF score) between two binary masks.

    Reference: Csurka et al., "What is a good evaluation measure for semantic segmentation?"
    Extracts the 1-pixel boundary of both masks using morphological gradients,
    dilates each by `tolerance` pixels, and calculates precision and recall
    of the boundary points.

    BF = 2 * (Precision * Recall) / (Precision + Recall)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_boundary = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel)
    ref_boundary = cv2.morphologyEx(ref, cv2.MORPH_GRADIENT, kernel)

    pred_cnt = int(np.sum(pred_boundary > 0))
    ref_cnt = int(np.sum(ref_boundary > 0))

    if pred_cnt == 0 and ref_cnt == 0:
        return 1.0  # Both masks empty — perfect boundary agreement
    if pred_cnt == 0 or ref_cnt == 0:
        return 0.0  # One empty, one has boundary — complete mismatch

    # Dilate boundaries by tolerance
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    ref_dilated = cv2.dilate(ref_boundary, dilate_kernel)
    pred_dilated = cv2.dilate(pred_boundary, dilate_kernel)

    # Precision: fraction of predicted boundary within tolerance of true boundary
    pred_match = float(np.sum((pred_boundary > 0) & (ref_dilated > 0)))
    b_precision = pred_match / pred_cnt if pred_cnt > 0 else 0.0

    # Recall: fraction of true boundary within tolerance of predicted boundary
    ref_match = float(np.sum((ref_boundary > 0) & (pred_dilated > 0)))
    b_recall = ref_match / ref_cnt if ref_cnt > 0 else 0.0

    if b_precision + b_recall == 0:
        return 0.0

    return float(2.0 * b_precision * b_recall / (b_precision + b_recall))


def create_overlay_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a color-coded overlay visualization.

    Colors:
    - Green: Agreement (True Positive)
    - Red: Prediction only (Potential false positive or detector-unique)
    - Blue: Reference only (Potential false negative or missed classical spot)
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
