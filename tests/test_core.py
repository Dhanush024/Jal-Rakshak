"""
Jal-Rakshak — Core Module Tests
==================================
Tests for geospatial, ocean, risk, ais, and pipeline modules.
Run with: python -m pytest tests/ -v
"""

import sys
import os
import math
import pytest
from datetime import datetime, timezone, timedelta

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# Geospatial
# ═══════════════════════════════════════════════════════════════════════════

class TestGeospatial:
    def test_haversine_zero_distance(self):
        from geospatial.distance import haversine_km
        assert haversine_km(10.0, 80.0, 10.0, 80.0) == 0.0

    def test_haversine_known_distance(self):
        from geospatial.distance import haversine_km
        # Chennai to Mumbai ~ 1,034 km
        dist = haversine_km(13.08, 80.27, 19.08, 72.88)
        assert 1000 < dist < 1100

    def test_bearing_north(self):
        from geospatial.distance import bearing_deg
        b = bearing_deg(10.0, 80.0, 11.0, 80.0)
        assert abs(b - 0.0) < 1.0  # nearly due north

    def test_bearing_east(self):
        from geospatial.distance import bearing_deg
        b = bearing_deg(10.0, 80.0, 10.0, 81.0)
        assert abs(b - 90.0) < 1.0  # nearly due east

    def test_destination_round_trip(self):
        from geospatial.distance import destination_point, haversine_km
        lat, lon = 12.45, 80.23
        lat2, lon2 = destination_point(lat, lon, 90.0, 10.0)
        dist = haversine_km(lat, lon, lat2, lon2)
        assert abs(dist - 10.0) < 0.01

    def test_bearing_difference(self):
        from geospatial.distance import bearing_difference
        assert bearing_difference(10, 350) == 20
        assert bearing_difference(0, 180) == 180
        assert bearing_difference(90, 90) == 0

    def test_polygon_area_km2(self):
        from geospatial.distance import polygon_area_km2
        # Empty and degenerate cases
        assert polygon_area_km2([]) == 0.0
        assert polygon_area_km2([[80.0, 12.0]]) == 0.0
        assert polygon_area_km2([[80.0, 12.0], [80.1, 12.0]]) == 0.0
        # 0.1 deg square near equator/tropics: ~120 km2
        square = [[80.0, 12.0], [80.1, 12.0], [80.1, 12.1], [80.0, 12.1], [80.0, 12.0]]
        area = polygon_area_km2(square)
        assert 115.0 < area < 125.0



# ═══════════════════════════════════════════════════════════════════════════
# Ocean Hindcast
# ═══════════════════════════════════════════════════════════════════════════

class TestOceanHindcast:
    def test_hindcast_produces_trajectory(self):
        from ocean.hindcast import DriftModel
        dm = DriftModel()
        t = datetime(2026, 9, 14, 15, 30, tzinfo=timezone.utc)
        result = dm.hindcast(12.45, 80.23, t, duration_minutes=30)
        assert len(result.trajectory) > 0
        assert result.direction == "hindcast"

    def test_hindcast_origin_differs_from_detection(self):
        from ocean.hindcast import DriftModel
        dm = DriftModel(current_speed_ms=0.5)
        t = datetime(2026, 9, 14, 15, 30, tzinfo=timezone.utc)
        result = dm.hindcast(12.45, 80.23, t, duration_minutes=60)
        # Origin should be different from detection point
        assert result.origin_lat != 12.45 or result.origin_lon != 80.23

    def test_hindcast_uncertainty_grows(self):
        from ocean.hindcast import DriftModel
        dm = DriftModel()
        t = datetime(2026, 9, 14, 15, 30, tzinfo=timezone.utc)
        result = dm.hindcast(12.45, 80.23, t, duration_minutes=60)
        # Trajectory is origin→detection after reversal.
        # The origin (first point) has highest uncertainty since it's
        # the farthest extrapolation from the detection point.
        uncertainties = [p.uncertainty_km for p in result.trajectory]
        assert uncertainties[0] >= uncertainties[-1]
        assert result.final_uncertainty_km > 0

    def test_forecast_produces_results(self):
        from ocean.hindcast import DriftModel
        dm = DriftModel()
        t = datetime(2026, 9, 14, 15, 30, tzinfo=timezone.utc)
        results = dm.forecast(12.45, 80.23, t, forecast_hours=[6, 12])
        assert len(results) == 2

    def test_particle_ensemble(self):
        from ocean.hindcast import DriftModel
        dm = DriftModel()
        t = datetime(2026, 9, 14, 15, 30, tzinfo=timezone.utc)
        particles = dm.particle_ensemble(12.45, 80.23, t,
                                          duration_minutes=60, num_particles=20)
        assert len(particles) == 20
        for lat, lon in particles:
            assert -90 <= lat <= 90
            assert -180 <= lon <= 360


# ═══════════════════════════════════════════════════════════════════════════
# Risk Engine
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskEngine:
    def test_low_risk(self):
        from risk.engine import assess_risk, RiskLevel
        r = assess_risk(spill_area_sq_km=0.01, distance_to_coast_km=200)
        assert r.level == RiskLevel.LOW

    def test_high_risk(self):
        from risk.engine import assess_risk, RiskLevel
        r = assess_risk(
            spill_area_sq_km=5.0,
            detection_confidence=0.9,
            distance_to_coast_km=10,
            sensitive_areas_nearby=2,
        )
        assert r.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_risk_has_factors(self):
        from risk.engine import assess_risk
        r = assess_risk(spill_area_sq_km=1.0)
        assert len(r.factors) >= 5

    def test_risk_score_range(self):
        from risk.engine import assess_risk
        r = assess_risk()
        assert 0 <= r.overall_score <= 100


# ═══════════════════════════════════════════════════════════════════════════
# AIS Filtering
# ═══════════════════════════════════════════════════════════════════════════

class TestAISFiltering:
    def _make_record(self, mmsi, lat, lon, speed=10.0, heading=180.0, t=None):
        from ais.provider import AISRecord
        if t is None:
            t = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
        return AISRecord(mmsi=mmsi, name=f"V{mmsi}", lat=lat, lon=lon,
                         heading=heading, speed_knots=speed, timestamp=t)

    def test_spatial_filter(self):
        from ais.filtering import filter_by_spatial_distance
        tracks = {
            "001": [self._make_record("001", 12.45, 80.23)],      # near
            "002": [self._make_record("002", 20.0, 90.0)],        # far
        }
        kept, stage = filter_by_spatial_distance(tracks, 12.45, 80.23, max_distance_km=50)
        assert "001" in kept
        assert "002" not in kept
        assert stage.vessels_before == 2
        assert stage.vessels_after == 1

    def test_temporal_filter(self):
        from ais.filtering import filter_by_temporal_window
        t1 = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)  # 4 days earlier
        tracks = {
            "001": [self._make_record("001", 12.0, 80.0, t=t1)],
            "002": [self._make_record("002", 12.0, 80.0, t=t2)],
        }
        kept, stage = filter_by_temporal_window(tracks, t1, window_hours=6)
        assert "001" in kept
        assert "002" not in kept


# ═══════════════════════════════════════════════════════════════════════════
# Demo Scenario
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoScenario:
    def test_demo_tracks_generated(self):
        from demo.scenario import generate_demo_ais_tracks
        tracks = generate_demo_ais_tracks()
        assert len(tracks) == 4
        for mmsi, track in tracks.items():
            assert len(track) > 0

    def test_demo_ocean_data(self):
        from demo.scenario import generate_demo_ocean_data
        currents, winds = generate_demo_ocean_data()
        assert len(currents) == 25  # 5x5 grid
        assert len(winds) == 25

    def test_demo_tracks_are_deterministic(self):
        from demo.scenario import generate_demo_ais_tracks
        t1 = generate_demo_ais_tracks()
        t2 = generate_demo_ais_tracks()
        for mmsi in t1:
            assert t1[mmsi][0].lat == t2[mmsi][0].lat


# ═══════════════════════════════════════════════════════════════════════════
# Alert Manager
# ═══════════════════════════════════════════════════════════════════════════

class TestAlertManager:
    def test_simulation_alert(self):
        from alerts.manager import AlertManager
        mgr = AlertManager(simulation_mode=True)
        mgr.set_incident("TEST-001")
        alert = mgr.generate_community_alert(
            location="Test Location", risk_level="HIGH",
            lat=12.45, lon=80.23,
        )
        assert alert.is_simulated
        assert alert.delivered
        assert "SIMULATED" in alert.format_community_alert()

    def test_authority_alert(self):
        from alerts.manager import AlertManager
        mgr = AlertManager(simulation_mode=True)
        mgr.set_incident("TEST-002")
        alert = mgr.generate_authority_alert(
            location="Test", risk_level="CRITICAL",
            spill_area_sq_km=5.0,
            candidate_vessels=["MT Falcon", "MV Star"],
        )
        assert "MT Falcon" in alert.message


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

class TestReporting:
    def test_report_generation(self):
        from reporting.incident import generate_report, DataClassification
        r = generate_report(
            incident_id="TEST-001",
            detection_result={"spill_detected": True},
            risk_assessment={"level": "HIGH"},
            is_demo=True,
        )
        assert r.incident_id == "TEST-001"
        assert len(r.sections) >= 2
        # Should have demo limitation
        assert any("SIMULATED" in l or "DEMO" in l for l in r.limitations)

    def test_report_json(self):
        from reporting.incident import generate_report
        r = generate_report(incident_id="TEST-002", is_demo=True)
        j = r.to_json()
        assert "TEST-002" in j


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_demo_mode_default(self):
        from config.settings import is_demo_mode
        # Default APP_MODE is "demo"
        assert is_demo_mode() is True

    def test_validate_config(self):
        from config.settings import validate_config
        warnings = validate_config()
        assert isinstance(warnings, list)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Graph
# ═══════════════════════════════════════════════════════════════════════════

class TestPipelineGraph:
    def test_pipeline_builds(self):
        from pipeline.graph import build_pipeline
        g = build_pipeline()
        assert "preprocess" in g.nodes
        assert "detect" in g.nodes
        assert "report" in g.nodes
        assert len(g.nodes) == 11

    def test_pipeline_compiles(self):
        from pipeline.graph import compile_pipeline
        c = compile_pipeline()
        assert c is not None

    def test_pipeline_execution_short_circuit(self, tmp_path):
        import cv2
        import numpy as np
        from pipeline.graph import run_pipeline

        # Empty black image has no spill -> should short-circuit to report
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        img_file = str(tmp_path / "blank.png")
        cv2.imwrite(img_file, blank)

        res = run_pipeline(img_file, spill_lat=12.45, spill_lon=80.23)
        assert res.get("spill_detected") is False
        assert res.get("report_done") is True
        assert res.get("hindcast_done") is None or res.get("hindcast_done") is False

    def test_pipeline_execution_full_demo(self):
        import os
        from pipeline.graph import run_pipeline
        from demo.scenario import CHENNAI_SCENARIO

        demo_img = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo", "demo_sar_patch.png")
        if os.path.exists(demo_img):
            res = run_pipeline(
                demo_img,
                spill_lat=CHENNAI_SCENARIO.spill_lat,
                spill_lon=CHENNAI_SCENARIO.spill_lon,
                detection_timestamp=CHENNAI_SCENARIO.detection_time.isoformat(),
            )
            assert res.get("spill_detected") is True
            assert res.get("hindcast_done") is True
            assert res.get("ais_done") is True
            assert res.get("attribution_done") is True
            assert res.get("risk_done") is True
            assert res.get("report_done") is True
            assert len(res.get("candidate_scores", [])) > 0
