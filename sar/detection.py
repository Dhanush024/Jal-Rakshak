"""
Jal-Rakshak — YOLO Oil Spill Detection Manager
=================================================
Manages YOLO model lifecycle and inference for oil spill segmentation.
Supports multiple detections, confidence thresholds, and proper mask handling.
"""

import os
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from config.settings import MODEL_PATH, DEFAULT_PIXEL_RESOLUTION_M


@dataclass
class SpillDetection:
    """A single detected oil spill region."""
    detection_id: int
    confidence: float
    mask: np.ndarray                    # binary mask (H, W)
    polygon: List[Tuple[float, float]]  # [(x, y), ...] pixel coordinates
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2)
    centroid_px: Tuple[float, float]    # pixel centroid (x, y)
    pixel_area: float                   # area in pixels
    geo_area_sq_km: Optional[float] = None  # area in sq km (when georeferenced)
    centroid_geo: Optional[Tuple[float, float]] = None  # (lat, lon)
    detection_timestamp: Optional[str] = None
    tile_id: Optional[int] = None
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "confidence": round(self.confidence, 4),
            "polygon_points": len(self.polygon),
            "bbox": list(self.bbox),
            "centroid_px": [round(c, 1) for c in self.centroid_px],
            "pixel_area": round(self.pixel_area, 1),
            "geo_area_sq_km": round(self.geo_area_sq_km, 4) if self.geo_area_sq_km else None,
            "centroid_geo": list(self.centroid_geo) if self.centroid_geo else None,
            "detection_timestamp": self.detection_timestamp,
            "tile_id": self.tile_id,
            "source_file": self.source_file,
        }


@dataclass
class DetectionResult:
    """Complete result from a detection run."""
    detections: List[SpillDetection]
    image_shape: Tuple[int, ...]
    model_name: str
    inference_time_ms: float
    source_file: Optional[str] = None
    preprocessing_level: Optional[str] = None

    @property
    def spill_detected(self) -> bool:
        return len(self.detections) > 0

    @property
    def total_pixel_area(self) -> float:
        return sum(d.pixel_area for d in self.detections)

    @property
    def primary_detection(self) -> Optional[SpillDetection]:
        """Return the largest detection by area."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.pixel_area)

    def to_dict(self) -> dict:
        return {
            "spill_detected": self.spill_detected,
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "image_shape": list(self.image_shape),
            "model_name": self.model_name,
            "inference_time_ms": round(self.inference_time_ms, 1),
            "source_file": self.source_file,
        }


class YOLODetector:
    """
    YOLO-based oil spill segmentation detector.

    Manages model lifecycle (load once, reuse) and handles:
    - Zero detections
    - Single detection
    - Multiple spills
    - Malformed masks
    - Very small detections (configurable minimum area)
    """

    def __init__(self, model_path: str = None, min_confidence: float = 0.25,
                 min_area_px: int = 100):
        self.model_path = model_path or MODEL_PATH
        self.min_confidence = min_confidence
        self.min_area_px = min_area_px
        self._model = None

    @property
    def model(self):
        """Lazy-load and cache the YOLO model."""
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"YOLO model not found: {self.model_path}. "
                    f"Ensure best.pt is in the project directory."
                )
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image_path: str,
               pixel_resolution_m: float = None,
               detection_timestamp: str = None) -> DetectionResult:
        """
        Run YOLO inference on an image and extract all spill detections.

        Args:
            image_path: Path to SAR image
            pixel_resolution_m: Pixel resolution in meters (for area calculation)
            detection_timestamp: ISO timestamp of detection

        Returns:
            DetectionResult with all valid detections
        """
        import time

        if pixel_resolution_m is None:
            pixel_resolution_m = DEFAULT_PIXEL_RESOLUTION_M

        if not os.path.exists(image_path):
            return DetectionResult(
                detections=[],
                image_shape=(0, 0),
                model_name=os.path.basename(self.model_path),
                inference_time_ms=0.0,
                source_file=image_path,
            )

        start = time.time()
        results = self.model(image_path, verbose=False)
        elapsed_ms = (time.time() - start) * 1000

        img = cv2.imread(image_path)
        img_shape = img.shape if img is not None else (0, 0)

        detections = []
        result = results[0]

        if result.masks is not None:
            for idx in range(len(result.masks.xy)):
                try:
                    detection = self._process_mask(
                        mask_xy=result.masks.xy[idx],
                        mask_data=result.masks.data[idx] if result.masks.data is not None else None,
                        confidence=float(result.boxes.conf[idx]) if result.boxes is not None else 0.0,
                        detection_id=idx,
                        image_shape=img_shape,
                        pixel_resolution_m=pixel_resolution_m,
                        detection_timestamp=detection_timestamp,
                        source_file=image_path,
                    )
                    if detection is not None:
                        detections.append(detection)
                except (IndexError, ValueError, cv2.error) as e:
                    # Skip malformed masks
                    continue

        return DetectionResult(
            detections=detections,
            image_shape=img_shape,
            model_name=os.path.basename(self.model_path),
            inference_time_ms=elapsed_ms,
            source_file=image_path,
        )

    def _process_mask(self, mask_xy, mask_data, confidence: float,
                      detection_id: int, image_shape: Tuple,
                      pixel_resolution_m: float,
                      detection_timestamp: Optional[str],
                      source_file: Optional[str]) -> Optional[SpillDetection]:
        """Process a single detection mask into a SpillDetection object."""
        # Filter by confidence
        if confidence < self.min_confidence:
            return None

        # Extract polygon coordinates
        polygon = mask_xy.tolist()
        if len(polygon) < 3:
            return None

        # Calculate pixel area
        pts = np.array(polygon, dtype=np.int32)
        pixel_area = cv2.contourArea(pts)
        if pixel_area < self.min_area_px:
            return None

        # Calculate centroid
        M = cv2.moments(pts)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])

        # Bounding box
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

        # Create binary mask
        h, w = image_shape[:2] if len(image_shape) >= 2 else (0, 0)
        if h > 0 and w > 0:
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(binary_mask, [pts.reshape(-1, 1, 2)], 255)
        else:
            binary_mask = np.zeros((1, 1), dtype=np.uint8)

        # Calculate geographic area
        area_sq_m = pixel_area * (pixel_resolution_m ** 2)
        geo_area_sq_km = area_sq_m / 1_000_000.0

        return SpillDetection(
            detection_id=detection_id,
            confidence=confidence,
            mask=binary_mask,
            polygon=[(float(p[0]), float(p[1])) for p in polygon],
            bbox=bbox,
            centroid_px=(float(cx), float(cy)),
            pixel_area=pixel_area,
            geo_area_sq_km=geo_area_sq_km,
            detection_timestamp=detection_timestamp,
            source_file=source_file,
        )
