"""
Jal-Rakshak — Spill Geometry & Characterization
=================================================
Calculate geometric properties of detected oil spill regions.
"""

import numpy as np
import cv2
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from config.settings import DEFAULT_PIXEL_RESOLUTION_M


@dataclass
class SpillCharacterization:
    """Complete geometric characterization of an oil spill."""
    # Pixel-space measurements
    centroid_px: Tuple[float, float]
    perimeter_px: float
    area_px: float
    length_px: float           # major axis of fitted ellipse
    width_px: float            # minor axis of fitted ellipse
    aspect_ratio: float
    orientation_deg: float     # angle of major axis from horizontal
    num_connected_regions: int
    is_fragmented: bool
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h
    convexity: float           # area / convex hull area
    solidity: float            # contour area / convex hull area

    # Geographic measurements (when resolution available)
    pixel_resolution_m: float
    area_sq_km: float
    perimeter_km: float
    length_km: float
    width_km: float
    estimated_volume_tons: Optional[float]  # only if scientifically justified
    volume_estimation_method: str           # "empirical" or "unavailable"

    # Geographic position (when georeferenced)
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None

    def to_dict(self) -> dict:
        result = {
            "centroid_px": [round(c, 1) for c in self.centroid_px],
            "perimeter_px": round(self.perimeter_px, 1),
            "area_px": round(self.area_px, 1),
            "length_px": round(self.length_px, 1),
            "width_px": round(self.width_px, 1),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "orientation_deg": round(self.orientation_deg, 1),
            "num_connected_regions": self.num_connected_regions,
            "is_fragmented": self.is_fragmented,
            "convexity": round(self.convexity, 3),
            "solidity": round(self.solidity, 3),
            "pixel_resolution_m": self.pixel_resolution_m,
            "area_sq_km": round(self.area_sq_km, 6),
            "perimeter_km": round(self.perimeter_km, 4),
            "length_km": round(self.length_km, 4),
            "width_km": round(self.width_km, 4),
            "volume_estimation_method": self.volume_estimation_method,
        }
        if self.estimated_volume_tons is not None:
            result["estimated_volume_tons"] = round(self.estimated_volume_tons, 2)
            result["volume_note"] = (
                "Estimated using empirical thickness assumption (0.1-1.0 mm). "
                "This is NOT a measured quantity. Actual volume depends on oil type, "
                "weathering, and spill dynamics."
            )
        if self.centroid_lat is not None:
            result["centroid_lat"] = round(self.centroid_lat, 6)
            result["centroid_lon"] = round(self.centroid_lon, 6)
        return result


def characterize_spill(mask: np.ndarray,
                       polygon: Optional[List[Tuple[float, float]]] = None,
                       pixel_resolution_m: float = None,
                       centroid_geo: Optional[Tuple[float, float]] = None
                       ) -> SpillCharacterization:
    """
    Compute comprehensive geometric characterization of a spill region.

    Args:
        mask: Binary mask of the spill (H, W), 255 = spill
        polygon: Optional polygon coordinates [(x, y), ...]
        pixel_resolution_m: Pixel size in meters
        centroid_geo: Optional (lat, lon) of the centroid

    Returns:
        SpillCharacterization with all computed properties
    """
    if pixel_resolution_m is None:
        pixel_resolution_m = DEFAULT_PIXEL_RESOLUTION_M

    binary = (mask > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_regions = len(contours)

    if num_regions == 0:
        return _empty_characterization(pixel_resolution_m)

    # Use the largest contour for primary measurements
    largest = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(largest)
    perimeter_px = cv2.arcLength(largest, True)

    # Centroid
    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = 0.0, 0.0

    # Fitted ellipse for length/width/orientation
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        (_, (minor_axis, major_axis), angle) = ellipse
        length_px = max(major_axis, minor_axis)
        width_px = min(major_axis, minor_axis)
        orientation = angle
    else:
        rect = cv2.minAreaRect(largest)
        (_, (w, h), angle) = rect
        length_px = max(w, h)
        width_px = min(w, h)
        orientation = angle

    aspect_ratio = length_px / width_px if width_px > 0 else 1.0

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(largest)

    # Convexity and solidity
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area_px / hull_area if hull_area > 0 else 0.0
    convexity = solidity  # same metric for our purposes

    # Fragmentation: more than one connected region
    is_fragmented = num_regions > 1

    # Geographic measurements
    m_per_px = pixel_resolution_m
    area_sq_m = area_px * (m_per_px ** 2)
    area_sq_km = area_sq_m / 1_000_000.0
    perimeter_km = perimeter_px * m_per_px / 1000.0
    length_km = length_px * m_per_px / 1000.0
    width_km = width_px * m_per_px / 1000.0

    # Volume estimation (empirical, with heavy caveats)
    # Typical oil slick thickness: 0.1 mm (sheen) to 1.0 mm (thick)
    # We use a mid-range assumption of 0.5 mm for rough estimation
    # Oil density ~0.85 tons/m³
    thickness_m = 0.0005  # 0.5 mm
    volume_m3 = area_sq_m * thickness_m
    oil_density_tons_per_m3 = 0.85
    estimated_volume_tons = volume_m3 * oil_density_tons_per_m3

    return SpillCharacterization(
        centroid_px=(cx, cy),
        perimeter_px=perimeter_px,
        area_px=area_px,
        length_px=length_px,
        width_px=width_px,
        aspect_ratio=aspect_ratio,
        orientation_deg=orientation,
        num_connected_regions=num_regions,
        is_fragmented=is_fragmented,
        bounding_rect=(x, y, w, h),
        convexity=convexity,
        solidity=solidity,
        pixel_resolution_m=pixel_resolution_m,
        area_sq_km=area_sq_km,
        perimeter_km=perimeter_km,
        length_km=length_km,
        width_km=width_km,
        estimated_volume_tons=estimated_volume_tons,
        volume_estimation_method="empirical",
        centroid_lat=centroid_geo[0] if centroid_geo else None,
        centroid_lon=centroid_geo[1] if centroid_geo else None,
    )


def _empty_characterization(pixel_resolution_m: float) -> SpillCharacterization:
    """Return a zeroed-out characterization when no spill is found."""
    return SpillCharacterization(
        centroid_px=(0, 0), perimeter_px=0, area_px=0,
        length_px=0, width_px=0, aspect_ratio=0, orientation_deg=0,
        num_connected_regions=0, is_fragmented=False,
        bounding_rect=(0, 0, 0, 0), convexity=0, solidity=0,
        pixel_resolution_m=pixel_resolution_m,
        area_sq_km=0, perimeter_km=0, length_km=0, width_km=0,
        estimated_volume_tons=None, volume_estimation_method="unavailable",
    )
