"""
Jal-Rakshak — Classical SAR Validation & Consensus Engine
==========================================================
Independent oil spill detection using classical image processing,
serving as an independent validation / evidence layer for YOLOv8.

Inspired by the reference repository (Dhanush024/Oil-Spill-Detection-in-SAR-images):
- Local adaptive thresholding (local_threshold.m)
- K-means intensity clustering (kmeansSegment.m & kmeansSegment_for_land.m)
- Automatic thresholding (automatic_threshold.m)
- Dark-spot extraction & centroid analysis (superpixel.m & automatic_threshold_for_land.m)
- Fuzzy gradient edge detection (fuzzy_edgeDetect.m)
- Superpixel oversegmentation (superpixel.m)
- Multi-signal consensus validation and 6-panel diagnostic visualization
"""

import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

from sar.landmask import extract_land_mask, intersect_with_ocean, LandMaskResult


class ClassicalMethod(Enum):
    """Available classical segmentation methods."""
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    KMEANS = "kmeans"
    OTSU = "otsu"
    DARK_SPOT = "dark_spot"
    FUZZY_EDGE = "fuzzy_edge"
    SUPERPIXEL = "superpixel"


class ValidationStatus(Enum):
    """Standardized validation consensus statuses."""
    CONFIRMED = "CONFIRMED BY MULTIPLE SIGNALS"
    PROBABLE = "PROBABLE"
    INCONCLUSIVE = "INCONCLUSIVE"
    LIKELY_LOOK_ALIKE = "LIKELY LOOK-ALIKE"
    REJECTED = "REJECTED"


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
class ConsensusValidationResult:
    """Complete multi-signal consensus validation result."""
    yolo_confidence: float
    classical_agreement: float          # 0.0 - 1.0 composite classical agreement score
    look_alike_risk: float              # 0.0 - 1.0 risk of biogenic/wind look-alike
    land_sea_consistency: float         # 0.0 - 1.0 fraction of detection inside ocean
    morphology_consistency: float       # 0.0 - 1.0 physical slick consistency
    contrast_ratio: float               # mean(slick) / mean(ocean)
    final_validation_status: str        # One of ValidationStatus values
    validated_mask: np.ndarray          # Refined consensus mask (H, W) uint8
    explanation: str                    # Human-readable rationale
    overlap_score: float                # IoU between YOLO and classical
    method_results: List[ClassicalResult] = field(default_factory=list)
    land_mask: Optional[np.ndarray] = None
    sea_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yolo_confidence": round(self.yolo_confidence, 4),
            "classical_agreement": round(self.classical_agreement, 4),
            "look_alike_risk": round(self.look_alike_risk, 4),
            "land_sea_consistency": round(self.land_sea_consistency, 4),
            "morphology_consistency": round(self.morphology_consistency, 4),
            "contrast_ratio": round(self.contrast_ratio, 3),
            "final_validation_status": self.final_validation_status,
            "overlap_score": round(self.overlap_score, 4),
            "explanation": self.explanation,
            "method_results": [r.to_dict() for r in self.method_results],
        }


# Maintain backward compatibility alias
ValidationResult = ConsensusValidationResult


def normalize_validation_result(raw_val: Any) -> Dict[str, Any]:
    """
    Normalize validation results into a predictable, safe presentation dictionary.

    Guarantees a stable dictionary schema regardless of whether the input is:
    - None
    - An empty list []
    - A dictionary
    - A list of dictionaries (multi-detection or sequential validation)
    - Any unexpected object

    Guarantees:
    - `method_results` is a dictionary keyed by method name, containing at least:
        `"land_mask": {"overlap_fraction": float, "consistency": float}`
    - `raw_method_results` contains the original list of method results.
    - All telemetry numeric fields (`yolo_confidence`, `classical_agreement`,
      `contrast_ratio`, `look_alike_risk`, `land_sea_consistency`) are floats.
    - Safe for expressions like:
        `val_res.get("method_results", {}).get("land_mask", {}).get("overlap_fraction", 0.0)`
    """
    normalized: Dict[str, Any] = {
        "yolo_confidence": 0.0,
        "classical_agreement": 0.0,
        "look_alike_risk": 0.0,
        "land_sea_consistency": 1.0,
        "morphology_consistency": 0.0,
        "contrast_ratio": 1.0,
        "final_validation_status": "STANDBY",
        "overlap_score": 0.0,
        "explanation": "Awaiting consensus evaluation.",
        "method_results": {},
        "raw_method_results": [],
    }

    if not raw_val:
        normalized["method_results"]["land_mask"] = {
            "overlap_fraction": 0.0,
            "consistency": 1.0,
            "method": "land_mask",
        }
        return normalized

    primary_dict: Dict[str, Any] = {}
    if isinstance(raw_val, (list, tuple)):
        normalized["raw_method_results"] = [item for item in raw_val]
        for item in raw_val:
            if isinstance(item, dict):
                primary_dict = item
                break
    elif isinstance(raw_val, dict):
        primary_dict = raw_val

    for k in [
        "yolo_confidence", "classical_agreement", "look_alike_risk",
        "land_sea_consistency", "morphology_consistency", "contrast_ratio",
        "final_validation_status", "overlap_score", "explanation"
    ]:
        if k in primary_dict and primary_dict[k] is not None:
            normalized[k] = primary_dict[k]

    raw_methods = primary_dict.get("method_results", [])
    methods_dict: Dict[str, Any] = {}

    if isinstance(raw_methods, list):
        if not normalized["raw_method_results"]:
            normalized["raw_method_results"] = raw_methods
        for m in raw_methods:
            if isinstance(m, dict):
                m_name = m.get("method", "unknown")
                methods_dict[str(m_name)] = m
    elif isinstance(raw_methods, dict):
        methods_dict = dict(raw_methods)
        if not normalized["raw_method_results"]:
            normalized["raw_method_results"] = list(raw_methods.values())

    # Ensure land_mask entry exists in method_results with overlap_fraction
    consistency = normalized.get("land_sea_consistency", 1.0)
    try:
        derived_overlap = max(0.0, min(1.0, 1.0 - float(consistency)))
    except (ValueError, TypeError):
        derived_overlap = 0.0

    if "land_mask" not in methods_dict or not isinstance(methods_dict["land_mask"], dict):
        methods_dict["land_mask"] = {
            "overlap_fraction": round(derived_overlap, 4),
            "consistency": round(float(consistency) if isinstance(consistency, (int, float)) else 1.0, 4),
            "method": "land_mask",
        }
    else:
        lm_entry = dict(methods_dict["land_mask"])
        if "overlap_fraction" not in lm_entry:
            lm_entry["overlap_fraction"] = round(derived_overlap, 4)
        methods_dict["land_mask"] = lm_entry

    normalized["method_results"] = methods_dict
    return normalized



# =========================================================================
# 1. Classical Segmentation Methods (Reference Repository Implementations)
# =========================================================================

def adaptive_threshold_segment(
    image: np.ndarray,
    block_size: int = 51,
    C: int = 10,
) -> ClassicalResult:
    """
    Local adaptive thresholding for dark region detection (local_threshold.m).

    Inverts grayscale SAR image so low-backscatter oil slicks become foreground,
    applying Gaussian-weighted local adaptive thresholding followed by morphological
    opening and closing with disk structuring elements.
    """
    start = time.time()
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert: dark regions become foreground
    inverted = cv2.bitwise_not(gray)

    # Gaussian adaptive threshold
    b_size = block_size if block_size % 2 == 1 else block_size + 1
    mask = cv2.adaptiveThreshold(
        inverted, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        b_size, C,
    )

    # Morphological opening and closing with disk element
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=2)

    # Hole filling (imfill holes)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= 50:
            cv2.drawContours(filled_mask, [c], -1, 255, thickness=cv2.FILLED)

    elapsed = (time.time() - start) * 1000
    total_area = float(np.sum(filled_mask > 0))

    return ClassicalResult(
        method=ClassicalMethod.ADAPTIVE_THRESHOLD,
        mask=filled_mask,
        num_regions=len(contours),
        total_area_px=total_area,
        processing_time_ms=elapsed,
        parameters={"block_size": b_size, "C": C},
    )


def kmeans_segment(
    image: np.ndarray,
    k: int = 4,
    min_area_px: int = 100,
) -> ClassicalResult:
    """
    K-means clustering segmentation (kmeansSegment.m & kmeansSegment_for_land.m).

    In SAR images, radar backscatter naturally clusters into:
    - Darkest cluster: oil slick / specular calm
    - Intermediate clusters: sea clutter / ocean background
    - Brightest cluster: terrestrial landmass / ships
    Extracts the darkest cluster, fills holes, and filters by minimum area.
    """
    start = time.time()
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian smoothing to reduce speckle before clustering
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    pixel_values = smoothed.reshape((-1, 1)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    centers = centers.flatten()
    labels = labels.flatten()

    # Identify the darkest cluster center
    sorted_order = np.argsort(centers)
    darkest_cluster_idx = sorted_order[0]
    darkest_center = float(centers[darkest_cluster_idx])

    # Also extract second darkest if it is close to the minimum (multi-thickness slick)
    clusters_to_take = [darkest_cluster_idx]
    if k > 2:
        second_idx = sorted_order[1]
        second_center = float(centers[second_idx])
        overall_mean = float(np.mean(gray))
        if second_center < overall_mean * 0.70:
            clusters_to_take.append(second_idx)

    label_map = labels.reshape(gray.shape)
    raw_mask = np.zeros_like(gray, dtype=np.uint8)
    for c_idx in clusters_to_take:
        raw_mask[label_map == c_idx] = 255

    # Morphological hole filling & area filtering (imfill & bwareafilt)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, se)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(cleaned)
    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) >= min_area_px:
            cv2.drawContours(filled_mask, [c], -1, 255, thickness=cv2.FILLED)
            valid_contours.append(c)

    elapsed = (time.time() - start) * 1000
    total_area = float(np.sum(filled_mask > 0))

    return ClassicalResult(
        method=ClassicalMethod.KMEANS,
        mask=filled_mask,
        num_regions=len(valid_contours),
        total_area_px=total_area,
        processing_time_ms=elapsed,
        parameters={"k": k, "darkest_center": darkest_center, "clusters_used": len(clusters_to_take)},
    )


def otsu_threshold_segment(image: np.ndarray) -> ClassicalResult:
    """
    Automatic thresholding using Otsu's method (automatic_threshold.m).
    """
    start = time.time()
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted = cv2.bitwise_not(gray)
    threshold, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=2)

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


def dark_spot_extraction(
    image: np.ndarray,
    percentile: float = 15.0,
    min_area_px: int = 150,
) -> ClassicalResult:
    """
    Dark-spot feature extraction (automatic_threshold_for_land.m & superpixel.m).

    Extracts pixels below an intensity quantile, isolates connected components,
    and filters by geometric area.
    """
    start = time.time()
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh_val = float(np.percentile(gray, percentile))
    mask = np.zeros_like(gray)
    mask[gray <= thresh_val] = 255

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=1)

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
        parameters={"percentile": percentile, "threshold_value": thresh_val},
    )


def fuzzy_edge_detect(
    image: np.ndarray,
    gradient_sigma: float = 15.0,
    edge_threshold: float = 0.60,
) -> ClassicalResult:
    """
    Fuzzy-logic boundary & edge detection (fuzzy_edgeDetect.m).

    Uses Sobel horizontal and vertical gradients to locate damping boundaries,
    applying a continuous sigmoidal fuzzy membership function to highlight
    the sharp perimeter transitions characteristic of mineral oil slicks.
    """
    start = time.time()
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Bilateral or Gaussian smoothing
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Sobel gradients
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # 3. Fuzzy membership function (sigmoid response to gradient strength)
    fuzzy_edge = 1.0 - np.exp(-(mag / (gradient_sigma + 1e-5)) ** 2)

    # 4. Binary edge mask
    edge_mask = np.zeros_like(gray, dtype=np.uint8)
    edge_mask[fuzzy_edge >= edge_threshold] = 255

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, se)

    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = float(np.sum(edge_mask > 0))
    elapsed = (time.time() - start) * 1000

    return ClassicalResult(
        method=ClassicalMethod.FUZZY_EDGE,
        mask=edge_mask,
        num_regions=len(contours),
        total_area_px=total_area,
        processing_time_ms=elapsed,
        parameters={"gradient_sigma": gradient_sigma, "edge_threshold": edge_threshold},
    )


def superpixel_segment(
    image: np.ndarray,
    grid_size: int = 16,
    dark_quantile: float = 0.20,
) -> ClassicalResult:
    """
    Superpixel oversegmentation & dark patch grouping (superpixel.m).

    Partitions the SAR image into regular local superpixel patches,
    computes the mean backscatter per superpixel, applies Otsu/quantile
    thresholding on patch means, and returns grouped candidate spill patches.
    """
    start = time.time()
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Block-wise superpixel aggregation
    step_y = max(4, grid_size)
    step_x = max(4, grid_size)

    superpixel_mask = np.zeros_like(gray)
    patch_means = []

    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            patch = gray[y:min(y + step_y, h), x:min(x + step_x, w)]
            patch_means.append(np.mean(patch))

    if patch_means:
        thresh = float(np.percentile(patch_means, dark_quantile * 100))
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                patch = gray[y:min(y + step_y, h), x:min(x + step_x, w)]
                if np.mean(patch) <= thresh:
                    superpixel_mask[y:min(y + step_y, h), x:min(x + step_x, w)] = 255

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    superpixel_mask = cv2.morphologyEx(superpixel_mask, cv2.MORPH_CLOSE, se)

    contours, _ = cv2.findContours(superpixel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = float(np.sum(superpixel_mask > 0))
    elapsed = (time.time() - start) * 1000

    return ClassicalResult(
        method=ClassicalMethod.SUPERPIXEL,
        mask=superpixel_mask,
        num_regions=len(contours),
        total_area_px=total_area,
        processing_time_ms=elapsed,
        parameters={"grid_size": grid_size, "dark_quantile": dark_quantile},
    )


# =========================================================================
# 2. Consensus Validation Framework
# =========================================================================

def validate_consensus(
    yolo_mask: np.ndarray,
    image: np.ndarray,
    yolo_confidence: float = 0.50,
    land_mask: Optional[np.ndarray] = None,
    methods: Optional[List[ClassicalMethod]] = None,
) -> ConsensusValidationResult:
    """
    Cross-validate YOLO detection against an independent classical consensus layer.

    Evaluates:
    1. Land/sea consistency: Spill MUST reside in the marine ocean domain.
    2. Intensity contrast: True spills have lower backscatter than surrounding ocean.
    3. Multi-algorithm classical agreement: Adaptive threshold, K-Means, Dark Spot.
    4. Morphology consistency: Aspect ratio, solidity, perimeter smoothness.
    5. Look-alike risk: Identifies natural low-wind calm zones or biogenic films.

    Produces one of the 5 standardized validation statuses:
    - CONFIRMED BY MULTIPLE SIGNALS
    - PROBABLE
    - INCONCLUSIVE
    - LIKELY LOOK-ALIKE
    - REJECTED
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Resize yolo_mask if necessary
    if yolo_mask.shape[:2] != (h, w):
        yolo_bin = cv2.resize((yolo_mask > 0).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        yolo_bin = (yolo_mask > 0).astype(np.uint8) * 255

    # 1. Land/sea separation
    if land_mask is None:
        land_res = extract_land_mask(gray)
        l_mask = land_res.land_mask
        s_mask = land_res.sea_mask
    else:
        if land_mask.shape[:2] != (h, w):
            l_mask = cv2.resize((land_mask > 0).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            l_mask = (land_mask > 0).astype(np.uint8) * 255
        s_mask = cv2.bitwise_not(l_mask)

    yolo_px = float(np.sum(yolo_bin > 0))
    total_px = float(h * w)

    # 2. Run Classical Methods
    if methods is None:
        methods = [
            ClassicalMethod.ADAPTIVE_THRESHOLD,
            ClassicalMethod.KMEANS,
            ClassicalMethod.DARK_SPOT,
        ]

    method_results: List[ClassicalResult] = []
    classical_votes = np.zeros((h, w), dtype=np.uint8)

    for m in methods:
        if m == ClassicalMethod.ADAPTIVE_THRESHOLD:
            res = adaptive_threshold_segment(gray)
        elif m == ClassicalMethod.KMEANS:
            res = kmeans_segment(gray)
        elif m == ClassicalMethod.OTSU:
            res = otsu_threshold_segment(gray)
        elif m == ClassicalMethod.DARK_SPOT:
            res = dark_spot_extraction(gray)
        elif m == ClassicalMethod.FUZZY_EDGE:
            res = fuzzy_edge_detect(gray)
        elif m == ClassicalMethod.SUPERPIXEL:
            res = superpixel_segment(gray)
        else:
            continue

        method_results.append(res)
        # Rescale if needed
        r_mask = res.mask
        if r_mask.shape[:2] != (h, w):
            r_mask = cv2.resize(r_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # Constrain classical evidence to sea
        r_ocean = intersect_with_ocean(r_mask, s_mask)
        classical_votes[r_ocean > 0] += 1

    # Consensus classical mask: at least 2 methods agree, or 1 method if only 1 tested
    vote_thresh = 2 if len(methods) >= 2 else 1
    combined_classical_mask = (classical_votes >= vote_thresh).astype(np.uint8) * 255

    # 3. Calculate Land/Sea Consistency
    if yolo_px > 0:
        land_overlap_px = float(np.sum((yolo_bin > 0) & (l_mask > 0)))
        land_overlap_ratio = land_overlap_px / yolo_px
        land_sea_consistency = max(0.0, 1.0 - land_overlap_ratio)
    else:
        land_sea_consistency = 1.0
        land_overlap_ratio = 0.0

    # 4. Intensity Contrast Analysis: mu(slick) vs mu(surrounding ocean)
    sea_only = (s_mask > 0)
    ocean_mean = float(np.mean(gray[sea_only])) if np.sum(sea_only) > 0 else float(np.mean(gray))

    if yolo_px > 0:
        slick_mean = float(np.mean(gray[yolo_bin > 0]))
        contrast_ratio = slick_mean / max(1.0, ocean_mean)
    else:
        slick_mean = ocean_mean
        contrast_ratio = 1.0

    # 5. Overlap Score (IoU between YOLO and classical in ocean domain)
    yolo_ocean = intersect_with_ocean(yolo_bin, s_mask)
    intersection = cv2.bitwise_and(yolo_ocean, combined_classical_mask)
    union = cv2.bitwise_or(yolo_ocean, combined_classical_mask)
    inter_px = float(np.sum(intersection > 0))
    union_px = float(np.sum(union > 0))

    overlap_score = inter_px / union_px if union_px > 0 else 0.0

    # Agreement score: weighted combination of IoU and vote support inside YOLO mask
    if yolo_px > 0:
        yolo_ocean_px = float(np.sum(yolo_ocean > 0))
        if yolo_ocean_px > 0:
            support_ratio = float(np.sum((yolo_ocean > 0) & (combined_classical_mask > 0))) / yolo_ocean_px
            classical_agreement = 0.5 * overlap_score + 0.5 * support_ratio
        else:
            classical_agreement = 0.0
    else:
        classical_agreement = 0.0

    # 6. Morphology Consistency
    morphology_consistency = 0.5
    if yolo_px > 0:
        contours, _ = cv2.findContours(yolo_ocean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_c)
            perimeter = cv2.arcLength(main_c, True)
            hull = cv2.convexHull(main_c)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(1.0, hull_area)
            # True slicks are often elongated filaments (moderate to low solidity, high perimeter/area)
            if 0.15 <= solidity <= 0.85:
                morphology_consistency = 0.8
            else:
                morphology_consistency = 0.4
            # Add bonus if contrast is convincingly dark
            if contrast_ratio < 0.75:
                morphology_consistency = min(1.0, morphology_consistency + 0.2)

    # 7. Look-Alike Risk Estimation
    look_alike_risk = 0.10
    if yolo_px > 0:
        # Weak contrast (slick not much darker than ocean) suggests look-alike or low wind
        if contrast_ratio > 0.85:
            look_alike_risk += 0.35
        # Very huge area (> 20% of scene) suggests natural low wind zone
        if yolo_px / total_px > 0.20:
            look_alike_risk += 0.30
        # High classical area without YOLO localization
        classical_px = float(np.sum(combined_classical_mask > 0))
        if classical_px > yolo_px * 3.5:
            look_alike_risk += 0.25
    look_alike_risk = min(1.0, max(0.0, look_alike_risk))

    # 8. Decision Logic for Final Validation Status
    explanation_parts = []

    # Hard rejection criteria:
    # A. Majority on land (land overlap > 40%)
    # B. Detected region is BRIGHTER than ocean (contrast_ratio >= 1.0)
    # C. Huge scene coverage (> 35% of total scene)
    if yolo_px > 0 and (land_overlap_ratio > 0.40 or contrast_ratio >= 1.0 or yolo_px / total_px > 0.35):
        status = ValidationStatus.REJECTED.value
        validated_mask = np.zeros((h, w), dtype=np.uint8)
        if land_overlap_ratio > 0.40:
            explanation_parts.append(
                f"Rejected: {land_overlap_ratio:.1%} terrestrial land overlap violates maritime domain constraint."
            )
        if contrast_ratio >= 1.0:
            explanation_parts.append(
                f"Rejected: Region has high backscatter (contrast ratio {contrast_ratio:.2f} >= 1.0), characteristic of land/waves rather than oil damping."
            )
        if yolo_px / total_px > 0.35:
            explanation_parts.append(
                f"Rejected: Coverage ({yolo_px / total_px:.1%}) exceeds maximum marine spill threshold."
            )
    elif yolo_px == 0:
        # Zero YOLO detections
        classical_px = float(np.sum(combined_classical_mask > 0))
        if classical_px > 1000:
            status = ValidationStatus.INCONCLUSIVE.value
            validated_mask = combined_classical_mask.copy()
            explanation_parts.append(
                "Inconclusive: YOLO detected no spill, but classical methods detected potential dark spots in open water."
            )
        else:
            status = ValidationStatus.CONFIRMED.value
            validated_mask = np.zeros((h, w), dtype=np.uint8)
            explanation_parts.append(
                "Confirmed: Both YOLO and classical consensus agree on clear ocean with no oil spill."
            )
    else:
        # YOLO detected a marine region with acceptable land overlap (< 40%) and dark contrast (< 1.0)
        # Refine candidate mask by intersecting strictly with ocean domain
        validated_candidate = yolo_ocean.copy()

        if yolo_confidence >= 0.40 and classical_agreement >= 0.30 and contrast_ratio < 0.85:
            status = ValidationStatus.CONFIRMED.value
            validated_mask = validated_candidate
            explanation_parts.append(
                f"Confirmed: Strong multi-signal agreement (YOLO {yolo_confidence:.0%}, "
                f"classical agreement {classical_agreement:.0%}, dark damping ratio {contrast_ratio:.2f})."
            )
        elif yolo_confidence >= 0.30 and (classical_agreement >= 0.15 or contrast_ratio < 0.75):
            status = ValidationStatus.PROBABLE.value
            validated_mask = validated_candidate
            explanation_parts.append(
                f"Probable: Moderate evidence (YOLO {yolo_confidence:.0%}, "
                f"classical agreement {classical_agreement:.0%}, contrast ratio {contrast_ratio:.2f})."
            )
        elif look_alike_risk >= 0.65 or contrast_ratio >= 0.88:
            status = ValidationStatus.LIKELY_LOOK_ALIKE.value
            validated_mask = validated_candidate
            explanation_parts.append(
                f"Likely Look-Alike: High risk ({look_alike_risk:.0%}) — weak backscatter damping "
                f"({contrast_ratio:.2f}) indicates potential low-wind calm or biogenic film."
            )
        else:
            status = ValidationStatus.INCONCLUSIVE.value
            validated_mask = validated_candidate
            explanation_parts.append(
                f"Inconclusive: Discrepancy between YOLO prediction and classical consensus "
                f"(agreement {classical_agreement:.0%}, IoU {overlap_score:.2f})."
            )

    explanation = " ".join(explanation_parts)

    return ConsensusValidationResult(
        yolo_confidence=float(yolo_confidence),
        classical_agreement=float(classical_agreement),
        look_alike_risk=float(look_alike_risk),
        land_sea_consistency=float(land_sea_consistency),
        morphology_consistency=float(morphology_consistency),
        contrast_ratio=float(contrast_ratio),
        final_validation_status=status,
        validated_mask=validated_mask,
        explanation=explanation,
        overlap_score=float(overlap_score),
        method_results=method_results,
        land_mask=l_mask,
        sea_mask=s_mask,
    )


# Backward-compatible wrapper
def validate_detection(
    yolo_mask: np.ndarray,
    image: np.ndarray,
    methods: Optional[List[ClassicalMethod]] = None,
) -> ConsensusValidationResult:
    """Wrapper ensuring backward compatibility with previous validate_detection calls."""
    return validate_consensus(yolo_mask=yolo_mask, image=image, methods=methods)


# =========================================================================
# 3. Qualitative Visual Validation: 6-Panel Diagnostic Generator
# =========================================================================

def generate_diagnostic_panels(
    image: np.ndarray,
    yolo_mask: np.ndarray,
    classical_mask: np.ndarray,
    land_mask: np.ndarray,
    validated_mask: np.ndarray,
    validation_status: str,
    title: str = "SAR Oil Spill Validation Diagnostics",
) -> np.ndarray:
    """
    Generate the mandatory 6-panel visual diagnostic image:

    Panel 1: ORIGINAL SAR
    Panel 2: YOLO MASK
    Panel 3: CLASSICAL MASK
    Panel 4: LAND/SEA MASK
    Panel 5: FINAL VALIDATED MASK
    Panel 6: OVERLAY

    This diagnostic visualization is essential for qualitative validation
    and debugging segmentation performance across complex coastal/marine scenes.
    """
    if len(image.shape) == 2:
        gray = image.copy()
        orig_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    # Normalize panel size to 400x500 for crisp uniform grid
    target_w = 480
    target_h = int(h * (target_w / max(1, w)))
    target_h = max(320, min(target_h, 480))

    def _prep_panel(img_bgr: np.ndarray, label: str, badge_color=(240, 240, 240)) -> np.ndarray:
        p = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # Header banner
        header_h = 42
        panel = np.zeros((target_h + header_h, target_w, 3), dtype=np.uint8)
        panel[header_h:, :] = p
        # Draw header bar
        cv2.rectangle(panel, (0, 0), (target_w, header_h), (25, 28, 36), -1)
        cv2.putText(panel, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2)
        # Thin boundary line
        cv2.rectangle(panel, (0, 0), (target_w - 1, target_h + header_h - 1), (60, 65, 75), 1)
        return panel

    # 1. ORIGINAL SAR
    p1 = _prep_panel(orig_bgr, "1. ORIGINAL SAR", (220, 220, 220))

    # 2. YOLO MASK (Magenta/Red foreground on dark base)
    yolo_res = cv2.resize((yolo_mask > 0).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    p2_bgr = cv2.cvtColor(gray // 2, cv2.COLOR_GRAY2BGR)
    p2_bgr[yolo_res > 0] = [255, 50, 80]  # Magenta/Red
    contours_y, _ = cv2.findContours(yolo_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p2_bgr, contours_y, -1, (255, 220, 100), 2)
    p2 = _prep_panel(p2_bgr, "2. YOLO MASK", (100, 160, 255))

    # 3. CLASSICAL MASK (Cyan on dark base)
    class_res = cv2.resize((classical_mask > 0).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    p3_bgr = cv2.cvtColor(gray // 2, cv2.COLOR_GRAY2BGR)
    p3_bgr[class_res > 0] = [0, 220, 255]  # Cyan
    contours_c, _ = cv2.findContours(class_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p3_bgr, contours_c, -1, (255, 255, 255), 1)
    p3 = _prep_panel(p3_bgr, "3. CLASSICAL MASK", (255, 220, 0))

    # 4. LAND/SEA MASK (Brown/Ochre for land, Navy for ocean)
    land_res = cv2.resize((land_mask > 0).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    p4_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    p4_bgr[:] = [60, 25, 15]  # Deep navy ocean
    p4_bgr[land_res > 0] = [40, 140, 190]  # Ochre/brown landmass
    contours_l, _ = cv2.findContours(land_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p4_bgr, contours_l, -1, (100, 230, 255), 2)  # Coastline
    p4 = _prep_panel(p4_bgr, "4. LAND/SEA MASK", (180, 210, 150))

    # 5. FINAL VALIDATED MASK
    val_res = cv2.resize((validated_mask > 0).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    p5_bgr = cv2.cvtColor(gray // 3, cv2.COLOR_GRAY2BGR)
    if np.sum(val_res > 0) > 0:
        p5_bgr[val_res > 0] = [50, 220, 50]  # Bright Green for validated spill
        contours_v, _ = cv2.findContours(val_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p5_bgr, contours_v, -1, (255, 255, 255), 2)
    else:
        # Clear or rejected
        cv2.putText(p5_bgr, "NO VALIDATED SPILL", (w // 6, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
    p5 = _prep_panel(p5_bgr, "5. FINAL VALIDATED MASK", (100, 255, 120))

    # 6. OVERLAY ON ORIGINAL
    p6_bgr = orig_bgr.copy()
    if np.sum(val_res > 0) > 0:
        overlay_layer = p6_bgr.copy()
        overlay_layer[val_res > 0] = [0, 30, 255]  # Red overlay
        p6_bgr = cv2.addWeighted(overlay_layer, 0.45, p6_bgr, 0.55, 0)
        contours_v, _ = cv2.findContours(val_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p6_bgr, contours_v, -1, (0, 255, 255), 2)
    # Status badge on overlay
    badge_bg = (0, 180, 0) if "CONFIRMED" in validation_status else \
               ((0, 140, 255) if "PROBABLE" in validation_status else (40, 40, 200))
    cv2.rectangle(p6_bgr, (10, 10), (min(w - 10, 380), 55), (20, 20, 20), -1)
    cv2.rectangle(p6_bgr, (10, 10), (min(w - 10, 380), 55), badge_bg, 2)
    cv2.putText(p6_bgr, validation_status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    p6 = _prep_panel(p6_bgr, "6. VALIDATED OVERLAY", (255, 255, 255))

    # Assemble 2 rows x 3 columns grid
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    grid = np.vstack([row1, row2])

    # Super-header title
    banner_h = 56
    composite = np.zeros((grid.shape[0] + banner_h, grid.shape[1], 3), dtype=np.uint8)
    composite[:banner_h, :] = (15, 18, 24)
    composite[banner_h:, :] = grid

    cv2.putText(composite, f"{title} | STATUS: {validation_status}",
                (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 245, 255), 2)

    return composite
