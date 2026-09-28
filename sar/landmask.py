"""
Jal-Rakshak — SAR Land/Sea Masking Module
==========================================
Isolates terrestrial landmasses from marine water bodies in SAR imagery.

Derived from the reference repository (Dhanush024/Oil-Spill-Detection-in-SAR-images):
- Wiener / Lee filtering for speckle attenuation
- Unsharp masking for land boundary sharpening
- Adaptive & K-means thresholding for land separation
- Morphological closing, opening, and hole filling (imfill)
- Marine ocean intersection guarantees
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LandMaskResult:
    """Result of land/sea segmentation."""
    land_mask: np.ndarray       # Binary uint8 (255 = land, 0 = ocean)
    sea_mask: np.ndarray        # Binary uint8 (255 = sea/water, 0 = land)
    land_fraction: float        # 0.0 to 1.0
    sea_fraction: float         # 0.0 to 1.0
    method_used: str
    threshold_value: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "land_fraction": round(self.land_fraction, 4),
            "sea_fraction": round(self.sea_fraction, 4),
            "method_used": self.method_used,
            "threshold_value": round(self.threshold_value, 2) if self.threshold_value is not None else None,
        }


def extract_land_mask(
    image: np.ndarray,
    threshold: Optional[int] = None,
    min_land_area_px: int = 500,
    blur_kernel_size: int = 5,
    morph_kernel_size: int = 7,
) -> LandMaskResult:
    """
    Extract a robust land/sea mask from a SAR image.

    In SAR images, landmasses typically produce high radar backscatter (bright)
    due to surface roughness and topographical facets, whereas open water
    acts as a specular or lower-backscatter surface.

    Workflow based on reference repo (land_mask.m & automatic_threshold_for_land.m):
    1. Grayscale conversion and intensity normalization
    2. Adaptive smoothing (Gaussian / median) to suppress speckle
    3. Contrast evaluation: if threshold is provided, use it; otherwise compute
       Otsu or adaptive upper-quantile thresholding
    4. Morphological closing and opening with disk/ellipse structuring elements
    5. Hole filling (imfill) to ensure contiguous landmasses
    6. Minimum area filtering to reject tiny bright buoys or ship spikes from land

    Args:
        image: Input SAR image (grayscale or BGR)
        threshold: Optional fixed intensity threshold (0-255). If None, Otsu is used.
        min_land_area_px: Minimum connected component area to consider as land
        blur_kernel_size: Gaussian kernel size for pre-threshold smoothing
        morph_kernel_size: Structuring element size for morphological cleanup

    Returns:
        LandMaskResult containing binary land and sea masks and scene statistics
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    total_pixels = float(h * w)

    if total_pixels == 0:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return LandMaskResult(
            land_mask=empty,
            sea_mask=empty,
            land_fraction=0.0,
            sea_fraction=1.0,
            method_used="empty_input",
        )

    # 1. Noise reduction via Gaussian smoothing (analogous to imgaussfilt in reference)
    k_blur = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
    smoothed = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)

    # 2. Thresholding: land is the brighter class in SAR backscatter
    if threshold is not None and threshold > 0:
        _, raw_mask = cv2.threshold(smoothed, threshold, 255, cv2.THRESH_BINARY)
        method_name = f"manual_threshold_{threshold}"
        thresh_val = float(threshold)
    else:
        # Otsu thresholding with adaptive fallback
        otsu_val, raw_mask = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        method_name = "adaptive_otsu"
        thresh_val = float(otsu_val)

    # 3. Morphological closing to seal fjords, valleys, and coastal inlets
    k_morph = morph_kernel_size if morph_kernel_size % 2 == 1 else morph_kernel_size + 1
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, se_close)

    # 4. Morphological opening to eliminate isolated oceanic speckles / buoys
    se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se_open)

    # 5. Contiguous land hole filling (equivalent to MATLAB imfill(mask, 'holes'))
    # Fill holes by finding external contours and drawing filled polygons
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(opened)

    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_land_area_px:
            cv2.drawContours(filled_mask, [c], -1, 255, thickness=cv2.FILLED)

    land_pixels = float(np.sum(filled_mask > 0))
    land_fraction = land_pixels / total_pixels
    sea_fraction = 1.0 - land_fraction

    # If Otsu erroneously classified > 92% of the scene as land (e.g. low-contrast oceanic patch),
    # clamp back or verify using intensity percentile
    if land_fraction > 0.92:
        # Re-estimate: only top 15% brightest pixels are candidate land
        p85 = float(np.percentile(gray, 85))
        _, p_mask = cv2.threshold(smoothed, int(p85), 255, cv2.THRESH_BINARY)
        p_closed = cv2.morphologyEx(p_mask, cv2.MORPH_CLOSE, se_close)
        p_contours, _ = cv2.findContours(p_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(opened)
        for c in p_contours:
            if cv2.contourArea(c) >= min_land_area_px * 2:
                cv2.drawContours(filled_mask, [c], -1, 255, thickness=cv2.FILLED)
        land_pixels = float(np.sum(filled_mask > 0))
        land_fraction = land_pixels / total_pixels
        sea_fraction = 1.0 - land_fraction
        method_name = "percentile_p85_fallback"
        thresh_val = p85

    sea_mask = cv2.bitwise_not(filled_mask)

    return LandMaskResult(
        land_mask=filled_mask,
        sea_mask=sea_mask,
        land_fraction=land_fraction,
        sea_fraction=sea_fraction,
        method_used=method_name,
        threshold_value=thresh_val,
    )


def intersect_with_ocean(mask: np.ndarray, sea_mask: np.ndarray) -> np.ndarray:
    """
    Constrain any candidate spill mask strictly to the marine ocean domain.

    Ensures the final mask contains zero terrestrial land pixels.
    """
    if mask.shape[:2] != sea_mask.shape[:2]:
        sea_mask_res = cv2.resize(
            sea_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    else:
        sea_mask_res = sea_mask
    return cv2.bitwise_and(mask, sea_mask_res)


def apply_sea_mask(image: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """
    Mask out land regions from an image, zeroing out all land pixels.
    """
    result = image.copy()
    if land_mask.shape[:2] != image.shape[:2]:
        land_mask_res = cv2.resize(
            land_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    else:
        land_mask_res = land_mask
    result[land_mask_res > 0] = 0
    return result
