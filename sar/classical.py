"""
Jal-Rakshak — Classical SAR Validation Methods
================================================
Independent oil spill detection using classical image processing,
for cross-validating YOLO segmentation results.

Inspired by Oil-Spill-Detection-in-SAR-images reference repository.
Implements adaptive thresholding, K-means segmentation, and dark-spot
extraction natively in Python/OpenCV.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class ClassicalMethod(Enum):
    """Available classical segmentation methods."""
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    KMEANS = "kmeans"
    OTSU = "otsu"
    DARK_SPOT = "dark_spot"


@dataclass
class ClassicalResult:
    """Result from a single classical segmentation method."""
    method: ClassicalMethod
    mask: np.ndarray        # Binary mask (H, W), 255 = detected dark region
    num_regions: int
    total_area_px: float
    processing_time_ms: float
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method.value,
            "num_regions": self.num_regions,
            "total_area_px": round(self.total_area_px, 1),
            "processing_time_ms": round(self.processing_time_ms, 1),
            "parameters": self.parameters,
        }


@dataclass
class ValidationResult:
    """Result of comparing YOLO mask with classical methods."""
    overlap_score: float          # 0-1, IoU between YOLO and classical
    agreement_pct: float          # 0-100, percentage of pixel agreement
    yolo_only_area_px: float      # pixels in YOLO but not classical
    classical_only_area_px: float # pixels in classical but not YOLO
    both_area_px: float           # pixels in both
    validation_confidence: float  # 0-1, confidence from agreement analysis
    look_alike_indicators: List[str]
    method_results: List[ClassicalResult]

    def to_dict(self) -> dict:
        return {
            "overlap_score": round(self.overlap_score, 4),
            "agreement_pct": round(self.agreement_pct, 1),
            "yolo_only_area_px": round(self.yolo_only_area_px, 1),
            "classical_only_area_px": round(self.classical_only_area_px, 1),
            "both_area_px": round(self.both_area_px, 1),
            "validation_confidence": round(self.validation_confidence, 4),
            "look_alike_indicators": self.look_alike_indicators,
            "method_results": [r.to_dict() for r in self.method_results],
        }


def adaptive_threshold_segment(image: np.ndarray,
                                block_size: int = 51,
                                C: int = 10) -> ClassicalResult:
    """
    Detect dark regions using local adaptive thresholding.

    Oil spills appear as dark regions in SAR images (low radar backscatter).
    Adaptive thresholding adjusts to local intensity variations, making it
    robust to uneven illumination across the SAR scene.
    """
    import time
    start = time.time()

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert: we want dark regions as foreground
    inverted = cv2.bitwise_not(gray)

    mask = cv2.adaptiveThreshold(
        inverted, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(c) for c in contours)

    elapsed = (time.time() - start) * 1000
    return ClassicalResult(
        method=ClassicalMethod.ADAPTIVE_THRESHOLD,
        mask=mask,
        num_regions=len(contours),
        total_area_px=total_area,
        processing_time_ms=elapsed,
        parameters={"block_size": block_size, "C": C},
    )


def kmeans_segment(image: np.ndarray, k: int = 3,
                   dark_cluster_threshold: float = 0.33) -> ClassicalResult:
    """
    K-means segmentation to identify dark clusters (potential oil spills).

    Groups pixels into K clusters by intensity. The darkest cluster(s)
    are potential oil spill regions.
    """
    import time
    start = time.time()

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixel_values = gray.reshape((-1, 1)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    centers = centers.flatten()
    labels = labels.flatten()

    # Identify the darkest cluster(s)
    sorted_indices = np.argsort(centers)
    # Take clusters whose center is below the threshold fraction of max
    dark_threshold = centers.max() * dark_cluster_threshold
    dark_clusters = [i for i in sorted_indices if centers[i] <= dark_threshold]

    if not dark_clusters:
        dark_clusters = [sorted_indices[0]]  # at least the darkest one

    # Create mask for dark clusters
    mask = np.zeros_like(gray)
    label_map = labels.reshape(gray.shape)
    for cluster_idx in dark_clusters:
        mask[label_map == cluster_idx] = 255

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(c) for c in contours)

    elapsed = (time.time() - start) * 1000
    return ClassicalResult(
        method=ClassicalMethod.KMEANS,
        mask=mask,
        num_regions=len(contours),
        total_area_px=total_area,
        processing_time_ms=elapsed,
        parameters={"k": k, "dark_threshold": dark_cluster_threshold,
                     "cluster_centers": centers.tolist()},
    )


def otsu_threshold_segment(image: np.ndarray) -> ClassicalResult:
    """
    Otsu's automatic thresholding for dark region detection.

    Automatically determines the optimal threshold to separate dark
    (potential spill) regions from the background.
    """
    import time
    start = time.time()

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu on inverted image (dark regions become bright)
    inverted = cv2.bitwise_not(gray)
    threshold, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(c) for c in contours)

    elapsed = (time.time() - start) * 1000
    return ClassicalResult(
        method=ClassicalMethod.OTSU,
        mask=mask,
        num_regions=len(contours),
        total_area_px=total_area,
        processing_time_ms=elapsed,
        parameters={"otsu_threshold": int(threshold)},
    )


def dark_spot_extraction(image: np.ndarray,
                          percentile: float = 15.0,
                          min_area_px: int = 200) -> ClassicalResult:
    """
    Extract dark spots from SAR image using percentile-based thresholding.

    Dark spots in SAR images are potential oil spills. This method
    extracts regions below a specified intensity percentile and filters
    by minimum area.
    """
    import time
    start = time.time()

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold at the given percentile
    threshold_val = np.percentile(gray, percentile)
    mask = np.zeros_like(gray)
    mask[gray <= threshold_val] = 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Filter by minimum area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    cv2.drawContours(filtered_mask, filtered_contours, -1, 255, -1)

    total_area = sum(cv2.contourArea(c) for c in filtered_contours)

    elapsed = (time.time() - start) * 1000
    return ClassicalResult(
        method=ClassicalMethod.DARK_SPOT,
        mask=filtered_mask,
        num_regions=len(filtered_contours),
        total_area_px=total_area,
        processing_time_ms=elapsed,
        parameters={"percentile": percentile, "min_area_px": min_area_px,
                     "threshold_value": float(threshold_val)},
    )


def validate_detection(yolo_mask: np.ndarray, image: np.ndarray,
                       methods: Optional[List[ClassicalMethod]] = None) -> ValidationResult:
    """
    Cross-validate YOLO detection against classical methods.

    Runs selected classical methods and compares their output with
    the YOLO segmentation mask to produce a validation confidence score.

    This is NOT ground truth comparison — it's independent method agreement.
    """
    if methods is None:
        methods = [ClassicalMethod.ADAPTIVE_THRESHOLD, ClassicalMethod.KMEANS]

    method_results = []
    combined_classical_mask = np.zeros_like(yolo_mask)

    for method in methods:
        if method == ClassicalMethod.ADAPTIVE_THRESHOLD:
            result = adaptive_threshold_segment(image)
        elif method == ClassicalMethod.KMEANS:
            result = kmeans_segment(image)
        elif method == ClassicalMethod.OTSU:
            result = otsu_threshold_segment(image)
        elif method == ClassicalMethod.DARK_SPOT:
            result = dark_spot_extraction(image)
        else:
            continue

        method_results.append(result)

        # Resize method mask to match YOLO mask if needed
        if result.mask.shape != yolo_mask.shape:
            resized = cv2.resize(result.mask, (yolo_mask.shape[1], yolo_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        else:
            resized = result.mask

        combined_classical_mask = cv2.bitwise_or(combined_classical_mask, resized)

    # Calculate agreement metrics
    yolo_binary = (yolo_mask > 0).astype(np.uint8)
    classical_binary = (combined_classical_mask > 0).astype(np.uint8)

    intersection = cv2.bitwise_and(yolo_binary, classical_binary)
    union = cv2.bitwise_or(yolo_binary, classical_binary)

    intersection_area = float(np.sum(intersection))
    union_area = float(np.sum(union))
    yolo_area = float(np.sum(yolo_binary))
    classical_area = float(np.sum(classical_binary))

    # IoU
    overlap_score = intersection_area / union_area if union_area > 0 else 0.0

    # Agreement percentage (over total image area)
    total_pixels = float(yolo_mask.shape[0] * yolo_mask.shape[1])
    agreement = float(np.sum(yolo_binary == classical_binary)) / total_pixels * 100 if total_pixels > 0 else 0.0

    # Look-alike indicators
    indicators = []
    if overlap_score < 0.2 and yolo_area > 0:
        indicators.append("Low YOLO-classical agreement — potential look-alike")
    if classical_area > yolo_area * 3:
        indicators.append("Classical method detects much larger region — possible low-wind zone")
    if classical_area < yolo_area * 0.1 and yolo_area > 0:
        indicators.append("Classical method finds much less — YOLO may be overdetecting")

    # Validation confidence
    # Higher when both methods agree (high IoU) and both detect something
    if yolo_area == 0 and classical_area == 0:
        validation_confidence = 0.0
    elif yolo_area == 0 or classical_area == 0:
        validation_confidence = 0.2
    else:
        validation_confidence = min(1.0, 0.3 + overlap_score * 0.7)

    return ValidationResult(
        overlap_score=overlap_score,
        agreement_pct=agreement,
        yolo_only_area_px=max(0, yolo_area - intersection_area),
        classical_only_area_px=max(0, classical_area - intersection_area),
        both_area_px=intersection_area,
        validation_confidence=validation_confidence,
        look_alike_indicators=indicators,
        method_results=method_results,
    )
