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
    is_valid_marine: bool = True
    rejection_reason: Optional[str] = None
    mean_intensity: Optional[float] = None

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
            "is_valid_marine": self.is_valid_marine,
            "rejection_reason": self.rejection_reason,
            "mean_intensity": round(self.mean_intensity, 2) if self.mean_intensity is not None else None,
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
    def valid_detections(self) -> List[SpillDetection]:
        """Return only detections that meet physical maritime constraints."""
        return [d for d in self.detections if d.is_valid_marine]

    @property
    def spill_detected(self) -> bool:
        """True if at least one valid marine spill detection exists."""
        return len(self.valid_detections) > 0

    @property
    def total_pixel_area(self) -> float:
        """Total area of all valid marine detections."""
        return sum(d.pixel_area for d in self.valid_detections)

    @property
    def primary_detection(self) -> Optional[SpillDetection]:
        """Return the most confident valid marine spill detection."""
        valids = self.valid_detections
        if not valids:
            return None
        return max(valids, key=lambda d: (d.confidence, d.pixel_area))

    def to_dict(self) -> dict:
        return {
            "spill_detected": self.spill_detected,
            "num_detections": len(self.valid_detections),
            "total_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.valid_detections],
            "all_detections": [d.to_dict() for d in self.detections],
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
    - Marine physical constraints (rejects huge landmasses or scene artifacts)
    """

    def __init__(self, model_path: str = None, min_confidence: float = 0.25,
                 min_area_px: int = 100, max_scene_coverage: float = 0.35,
                 max_land_overlap: float = 0.40):
        self.model_path = model_path or MODEL_PATH
        self.min_confidence = min_confidence
        self.min_area_px = min_area_px
        self.max_scene_coverage = max_scene_coverage
        self.max_land_overlap = max_land_overlap
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
               detection_timestamp: str = None,
               land_mask: Optional[np.ndarray] = None) -> DetectionResult:
        """
        Run YOLO inference on an image and extract all spill detections.

        Args:
            image_path: Path to SAR image
            pixel_resolution_m: Pixel resolution in meters (for area calculation)
            detection_timestamp: ISO timestamp of detection
            land_mask: Optional binary land mask (255 = land) for physical constraint

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
        h, w = img_shape[:2] if len(img_shape) >= 2 else (0, 0)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if (img is not None and len(img.shape) == 3) else img

        # Derive adaptive land/sea mask if not explicitly passed
        if land_mask is None and gray_img is not None and h > 0 and w > 0:
            from sar.preprocessing import create_land_mask
            land_mask = create_land_mask(gray_img)

        detections = []
        result = results[0]

        if result.masks is not None:
            num_masks = len(result.masks)
            for idx in range(num_masks):
                try:
                    cls_id = int(result.boxes.cls[idx]) if (result.boxes is not None and len(result.boxes.cls) > idx) else 0
                    if cls_id != 0:
                        continue  # Class 0 is 'oill'

                    conf = float(result.boxes.conf[idx]) if (result.boxes is not None and len(result.boxes.conf) > idx) else 0.0

                    m_xy = result.masks.xy[idx] if (result.masks.xy is not None and len(result.masks.xy) > idx) else None
                    m_data = result.masks.data[idx] if (result.masks.data is not None and len(result.masks.data) > idx) else None

                    detection = self._process_mask(
                        mask_xy=m_xy,
                        mask_data=m_data,
                        confidence=conf,
                        detection_id=idx,
                        image_shape=img_shape,
                        pixel_resolution_m=pixel_resolution_m,
                        detection_timestamp=detection_timestamp,
                        source_file=image_path,
                        gray_img=gray_img,
                        land_mask=land_mask,
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
                      source_file: Optional[str],
                      gray_img: Optional[np.ndarray] = None,
                      land_mask: Optional[np.ndarray] = None) -> Optional[SpillDetection]:
        """Process a single detection mask into a SpillDetection object."""
        # Filter by confidence
        if confidence < self.min_confidence:
            return None

        h, w = image_shape[:2] if len(image_shape) >= 2 else (0, 0)
        if h <= 0 or w <= 0:
            return None

        # 1. Native mask extraction from Ultralytics prototype masks
        if mask_data is not None:
            if hasattr(mask_data, "cpu"):
                m_arr = mask_data.cpu().numpy()
            else:
                m_arr = np.asarray(mask_data)
            if m_arr.shape[:2] != (h, w):
                binary_mask = cv2.resize((m_arr > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                binary_mask = (m_arr > 0.5).astype(np.uint8) * 255
        elif mask_xy is not None and len(mask_xy) >= 3:
            pts = np.array(mask_xy, dtype=np.int32)
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(binary_mask, [pts.reshape(-1, 1, 2)], 255)
        else:
            return None

        pixel_area = float(np.sum(binary_mask > 0))
        if pixel_area < self.min_area_px:
            return None

        # 2. Extract clean boundary contours without zero-width bridge seam lines
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            if len(main_contour) >= 3:
                polygon = [(float(p[0][0]), float(p[0][1])) for p in main_contour]
            else:
                polygon = [(float(p[0]), float(p[1])) for p in mask_xy] if (mask_xy is not None and len(mask_xy) >= 3) else []
        elif mask_xy is not None and len(mask_xy) >= 3:
            polygon = [(float(p[0]), float(p[1])) for p in mask_xy]
        else:
            polygon = []

        if len(polygon) < 3:
            return None

        # 3. Calculate centroid and bounding box
        M = cv2.moments(binary_mask)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            pts = np.array(polygon, dtype=np.float32)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

        x_min, y_min, bw, bh = cv2.boundingRect(binary_mask)
        bbox = (int(x_min), int(y_min), int(x_min + bw), int(y_min + bh))

        # 4. Marine physical validation checks
        scene_coverage = pixel_area / max(1, (h * w))
        is_valid_marine = True
        rejection_reason = None
        mean_intensity = None

        if gray_img is not None:
            mean_intensity = float(cv2.mean(gray_img, mask=binary_mask)[0])

        if scene_coverage > self.max_scene_coverage:
            is_valid_marine = False
            rejection_reason = (
                f"Scene coverage ({scene_coverage:.1%}) exceeds maximum marine threshold "
                f"({self.max_scene_coverage:.0%}) — flagged as landmass or full-scene artifact."
            )
        elif land_mask is not None:
            land_overlap = np.sum((binary_mask > 0) & (land_mask > 0)) / max(1, np.sum(binary_mask > 0))
            if land_overlap > self.max_land_overlap:
                is_valid_marine = False
                rejection_reason = (
                    f"Terrestrial land overlap ({land_overlap:.1%}) exceeds marine boundary threshold "
                    f"({self.max_land_overlap:.0%}) — flagged as coastal/terrestrial artifact."
                )

        # Geographic area
        area_sq_m = pixel_area * (pixel_resolution_m ** 2)
        geo_area_sq_km = area_sq_m / 1_000_000.0

        return SpillDetection(
            detection_id=detection_id,
            confidence=confidence,
            mask=binary_mask,
            polygon=polygon,
            bbox=bbox,
            centroid_px=(float(cx), float(cy)),
            pixel_area=pixel_area,
            geo_area_sq_km=geo_area_sq_km,
            detection_timestamp=detection_timestamp,
            source_file=source_file,
            is_valid_marine=is_valid_marine,
            rejection_reason=rejection_reason,
            mean_intensity=mean_intensity,
        )


def render_detection_overlay(
    image: np.ndarray,
    detections: List[SpillDetection],
    alpha: float = 0.40,
    draw_contours: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """
    Render clean detection overlay directly from native masks and contours.

    Preserves original image dimensions.
    Avoids polygon self-intersection and artificial seam artifacts.
    Handles multiple detections cleanly.
    """
    if image is None or len(image.shape) < 2:
        return image

    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    elif vis.shape[2] == 4:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGRA2RGB)
    h, w = vis.shape[:2]

    overlay = np.zeros_like(vis)
    has_mask = False

    for det in detections:
        mask = getattr(det, "mask", None)
        if mask is not None and mask.shape[:2] == (h, w) and np.sum(mask > 0) > 0:
            is_valid = getattr(det, "is_valid_marine", True)
            color = [255, 30, 30] if is_valid else [255, 165, 0]
            overlay[mask > 0] = color
            has_mask = True

            if draw_contours:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                line_color = (255, 230, 50) if is_valid else (200, 200, 200)
                cv2.drawContours(vis, contours, -1, line_color, 2)

            if draw_labels:
                cx, cy = getattr(det, "centroid_px", (w // 2, h // 2))
                conf = getattr(det, "confidence", 0.0)
                det_id = getattr(det, "detection_id", 0)
                label = f"Slick #{det_id} ({conf:.0%})" if is_valid else f"Artifact #{det_id} [REJECTED]"
                cv2.putText(vis, label, (max(10, int(cx) - 40), max(20, int(cy) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis, label, (max(10, int(cx) - 40), max(20, int(cy) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if has_mask:
        cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)

    return vis
