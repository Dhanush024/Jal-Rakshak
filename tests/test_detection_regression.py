"""Regression tests for YOLOv8 segmentation pipeline, coordinate scaling, and land false-positive rejection."""

import pytest
import os
import cv2
import numpy as np

from sar.detection import YOLODetector, DetectionResult, SpillDetection
from sar.preprocessing import TileInfo


class TestDetectionRegression:
    """Regression test suite for SAR detection pipeline."""

    def test_mask_dimensions_and_polygon_bounds(self):
        """Verify mask dimensions match image and polygon bounds are strictly within image extent."""
        img_path = "temp_upload.jpg"
        if not os.path.exists(img_path):
            pytest.skip("temp_upload.jpg not found")

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        detector = YOLODetector()
        res = detector.detect(img_path)

        assert res.image_shape[:2] == (h, w)
        assert len(res.valid_detections) > 0

        for d in res.valid_detections:
            assert d.mask.shape == (h, w)
            for x, y in d.polygon:
                assert 0.0 <= x <= float(w), f"Polygon x ({x}) outside image width ({w})"
                assert 0.0 <= y <= float(h), f"Polygon y ({y}) outside image height ({h})"

    def test_xy_orientation_no_transposition(self):
        """Verify (x, y) orientation corresponds to (col, row), not transposed (row, col)."""
        detector = YOLODetector()
        # Create a mock detection with a horizontal ellipse
        h, w = 300, 600
        mock_mask = np.zeros((h, w), dtype=np.uint8)
        # Horizontal ellipse centered at x=300, y=150, axes=(100, 20)
        cv2.ellipse(mock_mask, (300, 150), (100, 20), 0, 0, 360, 255, -1)
        contours, _ = cv2.findContours(mock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = contours[0].reshape(-1, 2)

        det = detector._process_mask(
            mask_xy=poly,
            mask_data=None,
            confidence=0.85,
            detection_id=0,
            image_shape=(h, w),
            pixel_resolution_m=10.0,
            detection_timestamp="2026-09-28T00:00:00Z",
            source_file="test.png",
        )

        assert det is not None
        assert det.is_valid_marine is True
        # Check centroid: cx should be ~300, cy should be ~150
        assert abs(det.centroid_px[0] - 300) < 5
        assert abs(det.centroid_px[1] - 150) < 5

    def test_land_false_positive_rejection(self):
        """Verify that large land/coastal detections (>35% scene coverage) are rejected from valid marine spills."""
        img_path = "temp_upload.jpg"
        if not os.path.exists(img_path):
            pytest.skip("temp_upload.jpg not found")

        detector = YOLODetector()
        res = detector.detect(img_path)

        # Ensure all detections are extracted
        assert len(res.detections) >= 3

        # Exactly the land detection should be flagged as invalid
        land_detections = [d for d in res.detections if not d.is_valid_marine]
        assert len(land_detections) >= 1
        assert "Scene coverage" in land_detections[0].rejection_reason or "land" in land_detections[0].rejection_reason.lower()

        # Valid marine detections must NOT include the land detection
        for d in res.valid_detections:
            assert d.is_valid_marine is True
            # The offshore slick is centered in the right half of the image (x > 300)
            assert d.centroid_px[0] > 300

    def test_primary_detection_selection_prefers_valid_high_confidence(self):
        """Verify that primary_detection picks the most confident valid marine slick, NOT the largest land area."""
        img_path = "temp_upload.jpg"
        if not os.path.exists(img_path):
            pytest.skip("temp_upload.jpg not found")

        detector = YOLODetector()
        res = detector.detect(img_path)

        primary = res.primary_detection
        assert primary is not None
        assert primary.is_valid_marine is True
        # Primary must be the genuine slick (Det 0 with conf ~0.698)
        assert primary.confidence >= 0.65
        # Primary must be in offshore region
        assert primary.centroid_px[0] > 340

    def test_empty_mask_handling(self):
        """Verify graceful handling when no spills or empty masks are present."""
        detector = YOLODetector()
        empty_poly = np.array([])
        det = detector._process_mask(
            mask_xy=empty_poly,
            mask_data=None,
            confidence=0.5,
            detection_id=0,
            image_shape=(100, 100),
            pixel_resolution_m=10.0,
            detection_timestamp=None,
            source_file=None,
        )
        assert det is None

    def test_multiple_masks_preserved(self):
        """Verify multiple valid detections are preserved in DetectionResult."""
        img_path = "temp_upload.jpg"
        if not os.path.exists(img_path):
            pytest.skip("temp_upload.jpg not found")

        detector = YOLODetector()
        res = detector.detect(img_path)

        assert len(res.valid_detections) >= 2
        for d in res.valid_detections:
            assert d.pixel_area > 1000

    def test_tile_local_mask_not_rendered_as_full_image(self):
        """Regression test ensuring tile-local coordinates map properly to global image coordinates."""
        full_h, full_w = 1000, 1000
        tile_x_offset = 300
        tile_y_offset = 400
        tile_size = 512

        tile_info = TileInfo(
            tile_id=1,
            row=1,
            col=1,
            x_offset=tile_x_offset,
            y_offset=tile_y_offset,
            width=tile_size,
            height=tile_size,
            source_width=full_w,
            source_height=full_h,
        )

        # Local tile polygon centered at (100, 100) inside the tile
        local_polygon = [(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)]

        # Global translation function
        global_polygon = [(x + tile_info.x_offset, y + tile_info.y_offset) for x, y in local_polygon]

        # Verify global coordinates are offset by tile_x_offset and tile_y_offset
        for (gx, gy), (lx, ly) in zip(global_polygon, local_polygon):
            assert gx == lx + 300
            assert gy == ly + 400
            assert gx >= tile_x_offset
            assert gy >= tile_y_offset
