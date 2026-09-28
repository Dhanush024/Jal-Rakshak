"""
Jal-Rakshak — SAR Preprocessing Pipeline
==========================================
Configurable SAR image preprocessing with speckle reduction,
contrast enhancement, normalization, and land masking.

Inspired by Oil-Spill-Detection-in-SAR-images reference repository:
- Lee filtering for speckle reduction
- Median/Wiener filtering
- Histogram equalization
- Land masking
"""

import numpy as np
import cv2
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


class PreprocessingLevel(Enum):
    """User-selectable preprocessing intensity."""
    RAW = "raw"
    LIGHT = "light"
    ENHANCED = "enhanced"


@dataclass
class TileInfo:
    """Metadata for a tile extracted from a larger SAR scene."""
    tile_id: int
    row: int
    col: int
    x_offset: int
    y_offset: int
    width: int
    height: int
    source_width: int
    source_height: int
    # Geospatial metadata (when available)
    geo_transform: Optional[Tuple[float, ...]] = None
    crs: Optional[str] = None
    acquisition_time: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "row": self.row,
            "col": self.col,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "geo_transform": self.geo_transform,
            "crs": self.crs,
            "acquisition_time": self.acquisition_time,
        }


@dataclass
class PreprocessingResult:
    """Result of SAR preprocessing."""
    image: np.ndarray
    level: PreprocessingLevel
    steps_applied: List[str] = field(default_factory=list)
    tiles: Optional[List[Tuple[np.ndarray, TileInfo]]] = None


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 uint8 range."""
    if image.dtype != np.uint8:
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min > 0:
            normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        return normalized
    return image.copy()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if multi-channel."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def lee_filter(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Lee speckle filter for SAR images.

    The Lee filter reduces multiplicative speckle noise while preserving
    edges better than simple averaging. It uses local statistics (mean and
    variance) to adaptively weight between the pixel value and the local mean.

    Reference: Lee, J.S. (1980). "Digital Image Enhancement and Noise
    Filtering by Use of Local Statistics"
    """
    img = image.astype(np.float64)

    # Local mean
    local_mean = cv2.blur(img, (kernel_size, kernel_size))

    # Local variance
    local_sq_mean = cv2.blur(img ** 2, (kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # numerical stability

    # Estimate noise variance from the overall image
    overall_var = np.var(img)
    if overall_var == 0:
        return image.copy()

    # Noise variance estimate (for multiplicative noise model)
    noise_var = overall_var * 0.25  # heuristic for typical SAR

    # Weight factor
    weight = local_var / (local_var + noise_var + 1e-10)

    # Filtered output: weighted combination of pixel and local mean
    filtered = local_mean + weight * (img - local_mean)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter for salt-and-pepper noise reduction."""
    return cv2.medianBlur(image, kernel_size)


def wiener_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Simplified Wiener-like filter using local statistics.

    A full Wiener filter requires knowledge of the noise power spectrum.
    This approximation uses local mean and variance for adaptive smoothing.
    """
    img = image.astype(np.float64)
    local_mean = cv2.blur(img, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(img ** 2, (kernel_size, kernel_size))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)

    noise_var = np.mean(local_var)
    if noise_var == 0:
        return image.copy()

    weight = np.maximum(local_var - noise_var, 0) / (local_var + 1e-10)
    filtered = local_mean + weight * (img - local_mean)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def histogram_equalization(image: np.ndarray, clip_limit: float = 2.0,
                           tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Better than global histogram equalization for SAR images because it
    preserves local contrast while enhancing dark regions (where oil appears).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def create_land_mask(image: np.ndarray, threshold: Optional[int] = None) -> np.ndarray:
    """
    Create a land mask based on adaptive Otsu or intensity thresholding.

    In SAR images, land generally appears bright (high radar backscatter) compared
    to calm water. When threshold is None, Otsu thresholding with Gaussian blurring
    adaptively segments terrestrial landmasses across scenes of arbitrary dynamic range.
    """
    gray = to_grayscale(image) if len(image.shape) == 3 else image
    if threshold is None or threshold <= 0:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Morphological closing to fill small gaps and unify land features
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def apply_sea_mask(image: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """Apply land mask to isolate sea regions. Land pixels set to 0."""
    result = image.copy()
    result[land_mask > 0] = 0
    return result


def tile_image(image: np.ndarray, tile_size: int = 512,
               overlap: int = 32,
               geo_transform: Optional[Tuple[float, ...]] = None,
               crs: Optional[str] = None,
               acquisition_time: Optional[str] = None) -> List[Tuple[np.ndarray, TileInfo]]:
    """
    Split a large SAR image into overlapping tiles for processing.

    Each tile retains metadata for reconstruction. Overlap ensures
    detections near tile boundaries are not missed.
    """
    h, w = image.shape[:2]
    tiles = []
    tile_id = 0
    step = tile_size - overlap

    for row_idx, y in enumerate(range(0, h, step)):
        for col_idx, x in enumerate(range(0, w, step)):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]

            # Pad if tile is smaller than tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size) + tile.shape[2:],
                                 dtype=tile.dtype) if len(tile.shape) == 3 else \
                         np.zeros((tile_size, tile_size), dtype=tile.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            info = TileInfo(
                tile_id=tile_id,
                row=row_idx,
                col=col_idx,
                x_offset=x,
                y_offset=y,
                width=x_end - x,
                height=y_end - y,
                source_width=w,
                source_height=h,
                geo_transform=geo_transform,
                crs=crs,
                acquisition_time=acquisition_time,
            )
            tiles.append((tile, info))
            tile_id += 1

    return tiles


def preprocess_sar(image: np.ndarray,
                   level: PreprocessingLevel = PreprocessingLevel.LIGHT,
                   apply_land_mask: bool = True) -> PreprocessingResult:
    """
    Apply configurable SAR preprocessing pipeline.

    Levels:
    - RAW: No processing, just normalization
    - LIGHT: Normalization + median filter + CLAHE
    - ENHANCED: Normalization + Lee filter + median + CLAHE + land mask
    """
    steps = []
    result = normalize_image(image)
    steps.append("normalize")

    gray = to_grayscale(result)
    steps.append("grayscale")

    if level == PreprocessingLevel.RAW:
        return PreprocessingResult(image=gray, level=level, steps_applied=steps)

    if level in (PreprocessingLevel.LIGHT, PreprocessingLevel.ENHANCED):
        gray = median_filter(gray, kernel_size=3)
        steps.append("median_filter_3x3")

    if level == PreprocessingLevel.ENHANCED:
        gray = lee_filter(gray, kernel_size=7)
        steps.append("lee_filter_7x7")

    # Contrast enhancement
    gray = histogram_equalization(gray)
    steps.append("CLAHE")

    # Land masking
    if apply_land_mask and level == PreprocessingLevel.ENHANCED:
        land = create_land_mask(image)
        gray = apply_sea_mask(gray, land)
        steps.append("land_mask")

    return PreprocessingResult(image=gray, level=level, steps_applied=steps)
