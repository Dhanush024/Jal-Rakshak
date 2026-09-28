"""
Jal-Rakshak — Advanced Pipeline Unit Tests
===========================================
Tests for:
- Spill Age Estimation & Weathering (sar/weathering.py)
- Multi-temporal SAR Tracking (sar/weathering.py)
- Source Probability Likelihood Surface (ocean/probability.py)
- Report PDF Export (reporting/pdf.py)
"""

import os
import sys
import pytest
from datetime import datetime, timezone, timedelta
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sar.geometry import SpillCharacterization, characterize_spill
from sar.weathering import (
    SpillAgeEstimator,
    TemporalSARObservation,
    MultiTemporalSARTracker,
)
from ocean.probability import SourceProbabilityModel
from reporting.incident import generate_report
from reporting.pdf import generate_pdf_report


class TestSpillAgeEstimation:
    """Test suite for SpillAgeEstimator."""

    def test_age_estimation_compact_fresh(self):
        """A compact, high-solidity, low-aspect-ratio spill should estimate fresh release."""
        # Create synthetic circular mask (50x50 circle in 200x200)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 30, 255, -1)
        char = characterize_spill(mask, pixel_resolution_m=10.0)

        res = SpillAgeEstimator.estimate_age(char, wind_speed_ms=5.0)
        assert res.status == "ESTIMATED"
        assert res.estimated_age_hours_min is not None
        assert res.estimated_age_hours_max is not None
        assert res.estimated_age_hours_min < res.estimated_age_hours_max
        assert res.best_estimate_hours <= 3.0
        assert res.confidence >= 0.50
        assert len(res.supporting_evidence) > 0

    def test_age_estimation_elongated_weathered(self):
        """An elongated, high-aspect-ratio slick should estimate mature weathering."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.ellipse(mask, (150, 150), (120, 15), 30, 0, 360, 255, -1)
        char = characterize_spill(mask, pixel_resolution_m=10.0)

        res = SpillAgeEstimator.estimate_age(char, wind_speed_ms=7.0)
        assert res.status == "ESTIMATED"
        assert res.best_estimate_hours >= 3.0
        assert res.fay_regime == "surface_tension_viscous"

    def test_age_estimation_tiny_fallback_unknown(self):
        """Spills smaller than 50 pixels must return AGE: UNKNOWN."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[48:52, 48:52] = 255  # 16 pixels
        char = characterize_spill(mask, pixel_resolution_m=10.0)

        res = SpillAgeEstimator.estimate_age(char)
        assert res.status == "UNKNOWN"
        assert res.best_estimate_hours is None
        assert "insufficient_data" in res.evidence_method

    def test_multitemporal_sar_growth(self):
        """Multi-temporal SAR passes should project growth rate and age."""
        t0 = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 9, 14, 14, 0, tzinfo=timezone.utc)  # +4 hours

        obs = [
            TemporalSARObservation(
                timestamp=t0,
                satellite_name="Sentinel-1A",
                area_sq_km=0.85,
                centroid_lat=13.10,
                centroid_lon=80.35,
                length_km=2.2,
                width_km=0.6,
                num_connected_regions=1,
                solidity=0.88,
                aspect_ratio=3.6,
            ),
            TemporalSARObservation(
                timestamp=t1,
                satellite_name="Sentinel-1B",
                area_sq_km=1.45,
                centroid_lat=13.08,
                centroid_lon=80.39,
                length_km=3.5,
                width_km=0.7,
                num_connected_regions=2,
                solidity=0.76,
                aspect_ratio=5.0,
            ),
        ]

        dummy_char = SpillCharacterization(
            centroid_px=(100, 100), perimeter_px=200, area_px=1000,
            length_px=80, width_px=20, aspect_ratio=4.0, orientation_deg=45,
            num_connected_regions=2, is_fragmented=True, bounding_rect=(0, 0, 100, 100),
            convexity=0.8, solidity=0.75, pixel_resolution_m=10.0,
            area_sq_km=1.45, perimeter_km=5.0, length_km=3.5, width_km=0.7,
            estimated_volume_tons=145.0, volume_estimation_method="empirical",
        )

        res = SpillAgeEstimator.estimate_age(dummy_char, multi_temporal_obs=obs)
        assert res.status == "ESTIMATED"
        assert res.evidence_method == "multi_temporal_growth"
        assert res.confidence >= 0.80
        assert res.best_estimate_hours > 4.0

    def test_multitemporal_tracker(self):
        """Test MultiTemporalSARTracker calculates speed and growth rate."""
        t0 = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 9, 14, 16, 0, tzinfo=timezone.utc)  # +6 hours

        obs = [
            TemporalSARObservation(
                timestamp=t0,
                satellite_name="Sentinel-1A",
                area_sq_km=1.0,
                centroid_lat=13.10,
                centroid_lon=80.35,
                length_km=2.0,
                width_km=0.5,
                num_connected_regions=1,
                solidity=0.85,
                aspect_ratio=4.0,
            ),
            TemporalSARObservation(
                timestamp=t1,
                satellite_name="RISAT-1",
                area_sq_km=1.9,
                centroid_lat=13.06,
                centroid_lon=80.42,
                length_km=3.8,
                width_km=0.8,
                num_connected_regions=2,
                solidity=0.70,
                aspect_ratio=4.75,
            ),
        ]

        track_res = MultiTemporalSARTracker.analyze_sequence(obs)
        assert track_res is not None
        assert track_res.num_observations == 2
        assert track_res.time_span_hours == 6.0
        assert pytest.approx(track_res.area_change_sq_km, 0.01) == 0.9
        assert track_res.net_drift_distance_km > 0
        assert track_res.fragmentation_trend == "increasing"


class TestSourceProbabilityMap:
    """Test suite for SourceProbabilityModel."""

    def test_source_probability_generation(self):
        """Test probability surface generation and quantile rings."""
        origin_time = datetime(2026, 9, 14, 14, 18, tzinfo=timezone.utc)
        res = SourceProbabilityModel.generate_surface(
            origin_lat=13.102,
            origin_lon=80.354,
            origin_time=origin_time,
            final_uncertainty_km=3.2,
        )

        assert res.origin_lat == 13.102
        assert res.origin_lon == 80.354
        assert res.p50_radius_km < res.p75_radius_km < res.p95_radius_km
        assert res.high_probability_area_sq_km > 0
        assert len(res.credible_zones) == 3
        assert len(res.heatmap_grid_points) > 20
        assert "dispersion" in res.uncertainty_statement.lower()

    def test_source_probability_with_particles(self):
        """Test probability surface centered on empirical particle cloud."""
        origin_time = datetime(2026, 9, 14, 14, 18, tzinfo=timezone.utc)
        particles = [
            (13.105 + np.sin(i) * 0.01, 80.358 + np.cos(i) * 0.01)
            for i in range(50)
        ]

        res = SourceProbabilityModel.generate_surface(
            origin_lat=13.10,
            origin_lon=80.35,
            origin_time=origin_time,
            final_uncertainty_km=4.0,
            particle_endpoints=particles,
        )

        assert abs(res.peak_lat - 13.105) < 0.02
        assert abs(res.peak_lon - 80.358) < 0.02
        assert res.p95_radius_km > 0


class TestReportPDFExport:
    """Test suite for Report PDF generation."""

    def test_pdf_report_compilation(self, tmp_path):
        """Test full incident report compilation to PDF."""
        report = generate_report(
            incident_id="JR-TEST-PDF",
            detection_result={"spill_detected": True, "confidence": 0.88, "num_spills": 1},
            characterization={"area_sq_km": 1.45, "perimeter_km": 6.2, "length_km": 3.1, "aspect_ratio": 3.8},
            age_estimation={"status": "ESTIMATED", "best_estimate_hours": 3.2, "confidence": 0.65},
            source_probability={"peak_lat": 13.10, "peak_lon": 80.35, "p50_radius_km": 1.8, "p95_radius_km": 3.8},
            vessel_scores=[{
                "mmsi": "419001234",
                "name": "MT ARCTIC STAR",
                "score": 87.4,
                "breakdown": {"spatial": 95, "temporal": 90, "trajectory": 85, "behavioral": 70},
                "features": {"min_distance_to_source_km": 1.2, "time_diff_to_event_min": 14.0},
            }],
            coastal_impact={
                "nearest_shoreline_point": {"name": "Ennore Creek"},
                "shortest_distance_to_coast_km": 8.4,
                "landfall_projected": True,
                "eta_to_coast_hours": 14.5,
                "coastal_vulnerability_score": 78.5,
                "threatened_assets_count": 2,
            },
            is_demo=True,
        )

        pdf_path = str(tmp_path / "test_incident_dossier.pdf")
        out = report.export_pdf(pdf_path)

        assert os.path.exists(out)
        assert os.path.getsize(out) > 5000  # Non-trivial PDF generated

        # Also test direct generate_pdf_report function
        direct_pdf = str(tmp_path / "test_direct.pdf")
        direct_out = generate_pdf_report(report.to_dict(), direct_pdf)
        assert os.path.exists(direct_out)
        assert os.path.getsize(direct_out) > 5000
