"""
Tests for Phase X SAR Enhancements: Land Masking, Classical Consensus,
Boundary F-score (BF Score), and 6-Panel Diagnostic Visualizations.
"""

import sys
import os
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sar.landmask import extract_land_mask, intersect_with_ocean, apply_sea_mask, LandMaskResult
from sar.classical import (
    ClassicalMethod,
    ValidationStatus,
    adaptive_threshold_segment,
    kmeans_segment,
    otsu_threshold_segment,
    dark_spot_extraction,
    fuzzy_edge_detect,
    superpixel_segment,
    validate_consensus,
    generate_diagnostic_panels,
    normalize_validation_result,
)
from sar.metrics import (
    compute_metrics,
    evaluate_against_ground_truth,
    evaluate_against_classical_reference,
    _boundary_f_score,
)


@pytest.fixture
def mock_sar_coastal_scene():
    """Create a synthetic 256x256 SAR image with bright land (right) and dark sea (left)."""
    img = np.full((256, 256), 80, dtype=np.uint8)  # Sea background
    # Add land on the right half (bright backscatter)
    img[:, 150:] = 190
    # Add a genuine dark oil slick in the sea (low backscatter)
    cv2.ellipse(img, (75, 120), (35, 15), 30, 0, 360, 30, -1)
    # Add speckle noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


class TestLandMasking:
    """Verify land/sea separation and ocean intersection."""

    def test_extract_land_mask(self, mock_sar_coastal_scene):
        res = extract_land_mask(mock_sar_coastal_scene)
        assert isinstance(res, LandMaskResult)
        assert res.land_mask.shape == (256, 256)
        assert res.sea_mask.shape == (256, 256)
        # Land should cover roughly the right region
        assert res.land_mask[:, 200].mean() > 200
        # Sea should cover roughly the left region
        assert res.sea_mask[:, 50].mean() > 200
        assert 0.20 < res.land_fraction < 0.60
        assert 0.40 < res.sea_fraction < 0.80

    def test_intersect_with_ocean(self):
        candidate_mask = np.full((100, 100), 255, dtype=np.uint8)
        sea_mask = np.zeros((100, 100), dtype=np.uint8)
        sea_mask[:, :50] = 255  # Left half is ocean

        ocean_spill = intersect_with_ocean(candidate_mask, sea_mask)
        assert np.sum(ocean_spill[:, :50] > 0) == 50 * 100
        assert np.sum(ocean_spill[:, 50:] > 0) == 0

    def test_apply_sea_mask(self, mock_sar_coastal_scene):
        res = extract_land_mask(mock_sar_coastal_scene)
        sea_img = apply_sea_mask(mock_sar_coastal_scene, res.land_mask)
        assert sea_img[res.land_mask > 0].sum() == 0


class TestClassicalAlgorithms:
    """Verify individual classical segmentation methods inspired by reference repository."""

    def test_adaptive_threshold_segment(self, mock_sar_coastal_scene):
        res = adaptive_threshold_segment(mock_sar_coastal_scene)
        assert res.method == ClassicalMethod.ADAPTIVE_THRESHOLD
        assert res.mask.shape == (256, 256)
        assert res.total_area_px > 0

    def test_kmeans_segment(self, mock_sar_coastal_scene):
        res = kmeans_segment(mock_sar_coastal_scene, k=4)
        assert res.method == ClassicalMethod.KMEANS
        assert res.mask.shape == (256, 256)
        assert res.total_area_px > 0

    def test_otsu_threshold_segment(self, mock_sar_coastal_scene):
        res = otsu_threshold_segment(mock_sar_coastal_scene)
        assert res.method == ClassicalMethod.OTSU
        assert res.mask.shape == (256, 256)

    def test_dark_spot_extraction(self, mock_sar_coastal_scene):
        res = dark_spot_extraction(mock_sar_coastal_scene, percentile=15.0)
        assert res.method == ClassicalMethod.DARK_SPOT
        assert res.mask.shape == (256, 256)

    def test_fuzzy_edge_detect(self, mock_sar_coastal_scene):
        res = fuzzy_edge_detect(mock_sar_coastal_scene)
        assert res.method == ClassicalMethod.FUZZY_EDGE
        assert res.mask.shape == (256, 256)

    def test_superpixel_segment(self, mock_sar_coastal_scene):
        res = superpixel_segment(mock_sar_coastal_scene, grid_size=16)
        assert res.method == ClassicalMethod.SUPERPIXEL
        assert res.mask.shape == (256, 256)


class TestConsensusValidation:
    """Verify interpretable multi-signal consensus scoring and 5 statuses."""

    def test_confirmed_spill(self, mock_sar_coastal_scene):
        # Genuine slick in the sea
        slick_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(slick_mask, (75, 120), (35, 15), 30, 0, 360, 255, -1)

        consensus = validate_consensus(
            yolo_mask=slick_mask,
            image=mock_sar_coastal_scene,
            yolo_confidence=0.75,
        )
        assert consensus.final_validation_status in (
            ValidationStatus.CONFIRMED.value,
            ValidationStatus.PROBABLE.value,
        )
        assert consensus.land_sea_consistency > 0.85
        assert consensus.contrast_ratio < 0.85
        assert np.sum(consensus.validated_mask > 0) > 0

    def test_rejected_land_false_positive(self, mock_sar_coastal_scene):
        # YOLO erroneously detecting bright land on the right
        land_false_mask = np.zeros((256, 256), dtype=np.uint8)
        land_false_mask[:, 180:] = 255

        consensus = validate_consensus(
            yolo_mask=land_false_mask,
            image=mock_sar_coastal_scene,
            yolo_confidence=0.45,
        )
        assert consensus.final_validation_status == ValidationStatus.REJECTED.value
        assert np.sum(consensus.validated_mask > 0) == 0
        assert "land" in consensus.explanation.lower() or "contrast" in consensus.explanation.lower()

    def test_zero_detection_agreement(self, mock_sar_coastal_scene):
        empty_mask = np.zeros((256, 256), dtype=np.uint8)
        consensus = validate_consensus(
            yolo_mask=empty_mask,
            image=mock_sar_coastal_scene,
            yolo_confidence=0.0,
        )
        assert consensus.final_validation_status in (
            ValidationStatus.CONFIRMED.value,
            ValidationStatus.INCONCLUSIVE.value,
        )


class TestMetricsAndEvaluation:
    """Verify BF score and separation between ground-truth and classical validation."""

    def test_bf_score_identical_masks(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 20, 255, -1)
        bf = _boundary_f_score(mask, mask, tolerance=3)
        assert bf == pytest.approx(1.0, abs=0.01)

    def test_bf_score_empty_masks(self):
        empty = np.zeros((100, 100), dtype=np.uint8)
        assert _boundary_f_score(empty, empty) == 1.0

    def test_evaluate_against_ground_truth(self):
        pred = np.zeros((100, 100), dtype=np.uint8)
        gt = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(pred, (50, 50), 20, 255, -1)
        cv2.circle(gt, (50, 50), 20, 255, -1)

        metrics = evaluate_against_ground_truth(pred, gt)
        assert metrics.evaluation_type == "ground_truth"
        assert metrics.iou == pytest.approx(1.0, abs=0.01)
        assert metrics.dice == pytest.approx(1.0, abs=0.01)
        assert metrics.boundary_f_score == pytest.approx(1.0, abs=0.01)

    def test_evaluate_against_classical_reference(self):
        pred = np.zeros((100, 100), dtype=np.uint8)
        classical = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(pred, (50, 50), 20, 255, -1)
        cv2.circle(classical, (52, 50), 20, 255, -1)

        metrics = evaluate_against_classical_reference(pred, classical)
        assert metrics.evaluation_type == "classical_validation"
        assert metrics.iou > 0.70
        assert metrics.dice > 0.80


class TestDiagnosticPanels:
    """Verify generation of the 6-panel diagnostic visualization."""

    def test_generate_diagnostic_panels(self, mock_sar_coastal_scene):
        yolo_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(yolo_mask, (75, 120), 20, 255, -1)
        class_mask = yolo_mask.copy()
        land_mask = np.zeros((256, 256), dtype=np.uint8)
        land_mask[:, 150:] = 255

        composite = generate_diagnostic_panels(
            image=mock_sar_coastal_scene,
            yolo_mask=yolo_mask,
            classical_mask=class_mask,
            land_mask=land_mask,
            validated_mask=yolo_mask,
            validation_status=ValidationStatus.CONFIRMED.value,
        )
        assert composite is not None
        assert len(composite.shape) == 3
        # Should be a wide 2-row x 3-col composite with header
        assert composite.shape[1] > 1000
        assert composite.shape[0] > 600


class TestValidationResultNormalization:
    """
    Regression tests for validation_result normalization.
    Guarantees no AttributeError: 'list' object has no attribute 'get'.
    Covers the 6 required edge cases:
    1. val_res dict
    2. val_res list
    3. val_res empty list
    4. val_res None
    5. missing land_mask
    6. missing overlap_fraction
    """

    def test_val_res_none(self):
        """Case 4: val_res is None."""
        norm = normalize_validation_result(None)
        assert isinstance(norm, dict)
        assert isinstance(norm.get("method_results"), dict)
        assert isinstance(norm.get("method_results", {}).get("land_mask"), dict)
        overlap = norm.get("method_results", {}).get("land_mask", {}).get("overlap_fraction", 0.0)
        assert isinstance(overlap, (int, float))
        assert overlap == 0.0

    def test_val_res_empty_list(self):
        """Case 3: val_res is an empty list []."""
        norm = normalize_validation_result([])
        assert isinstance(norm, dict)
        assert isinstance(norm.get("method_results"), dict)
        overlap = norm.get("method_results", {}).get("land_mask", {}).get("overlap_fraction", 0.0)
        assert isinstance(overlap, (int, float))

    def test_val_res_list_of_dicts(self):
        """Case 2: val_res is a list of dictionaries (multi-detection/pipeline pass)."""
        raw_list = [
            {
                "final_validation_status": "CONFIRMED",
                "classical_agreement": 0.88,
                "land_sea_consistency": 0.95,
                "method_results": [
                    {"method": "adaptive_threshold", "total_area_px": 500},
                    {"method": "kmeans", "total_area_px": 480},
                ],
            }
        ]
        norm = normalize_validation_result(raw_list)
        assert isinstance(norm, dict)
        assert norm["final_validation_status"] == "CONFIRMED"
        assert norm["classical_agreement"] == 0.88
        assert isinstance(norm["method_results"], dict)
        # Check that method_results was indexed by name
        assert "adaptive_threshold" in norm["method_results"]
        # Exact problematic expression from prompt:
        overlap = norm.get("method_results", {}).get("land_mask", {}).get("overlap_fraction", 0.0)
        assert isinstance(overlap, (int, float))
        assert abs(overlap - 0.05) < 1e-3  # 1.0 - 0.95 = 0.05

    def test_val_res_dict_with_list_method_results(self):
        """Case 1: val_res is standard dict produced by ConsensusValidationResult.to_dict()."""
        raw_dict = {
            "yolo_confidence": 0.85,
            "classical_agreement": 0.92,
            "look_alike_risk": 0.12,
            "land_sea_consistency": 0.98,
            "contrast_ratio": 0.55,
            "final_validation_status": "CONFIRMED",
            "explanation": "High multi-signal consensus",
            "method_results": [
                {"method": "adaptive_threshold", "total_area_px": 1200},
                {"method": "kmeans", "total_area_px": 1150},
            ],
        }
        norm = normalize_validation_result(raw_dict)
        assert isinstance(norm, dict)
        assert isinstance(norm.get("method_results"), dict)
        # Must not raise AttributeError: 'list' object has no attribute 'get'
        overlap = norm.get("method_results", {}).get("land_mask", {}).get("overlap_fraction", 0.0)
        assert isinstance(overlap, (int, float))
        assert overlap >= 0.0

    def test_val_res_missing_land_mask(self):
        """Case 5: method_results dictionary is present but missing land_mask key."""
        raw_dict = {
            "classical_agreement": 0.75,
            "land_sea_consistency": 0.90,
            "method_results": {
                "otsu": {"total_area_px": 300},
            },
        }
        norm = normalize_validation_result(raw_dict)
        assert "land_mask" in norm["method_results"]
        overlap = norm.get("method_results", {}).get("land_mask", {}).get("overlap_fraction", 0.0)
        assert isinstance(overlap, (int, float))
        assert abs(overlap - 0.10) < 1e-3

    def test_val_res_missing_overlap_fraction(self):
        """Case 6: land_mask is present but missing overlap_fraction key."""
        raw_dict = {
            "land_sea_consistency": 0.85,
            "method_results": {
                "land_mask": {"status": "checked"},
            },
        }
        norm = normalize_validation_result(raw_dict)
        overlap = norm.get("method_results", {}).get("land_mask", {}).get("overlap_fraction", 0.0)
        assert isinstance(overlap, (int, float))
        assert abs(overlap - 0.15) < 1e-3

