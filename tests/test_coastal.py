"""
Jal-Rakshak — Coastal Impact Assessment Tests
==============================================
Tests for shoreline proximity, landfall trajectory intersection,
threatened sensitive assets, and environmental containment strategies.
"""

import pytest
from coastal.zones import (
    EnvironmentalSensitivityIndex,
    CoastalPoint,
    SensitiveArea,
    CHENNAI_CORRIDOR_SHORELINE,
    CHENNAI_SENSITIVE_AREAS,
    get_nearest_shoreline_point,
)
from coastal.impact import (
    assess_coastal_impact,
    CoastalImpactResult,
)


class TestCoastalZones:
    def test_nearest_shoreline_point(self):
        # Spill off Chennai coast: (12.45 N, 80.23 E)
        pt, dist_km = get_nearest_shoreline_point(12.45, 80.23)
        assert pt is not None
        assert isinstance(pt, CoastalPoint)
        assert 10.0 < dist_km < 25.0
        assert pt.esi in list(EnvironmentalSensitivityIndex)

    def test_sensitive_areas_defined(self):
        assert len(CHENNAI_SENSITIVE_AREAS) >= 4
        for sa in CHENNAI_SENSITIVE_AREAS:
            assert isinstance(sa, SensitiveArea)
            assert sa.vulnerability_score > 0
            assert sa.priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            assert len(sa.recommended_strategy) > 0


class TestCoastalImpactEngine:
    def test_coastal_impact_calculation(self):
        result = assess_coastal_impact(
            spill_lat=12.45,
            spill_lon=80.23,
            current_speed_ms=0.45,
            current_bearing_deg=270.0,  # Moving due west directly toward coast
            wind_speed_ms=6.0,
            wind_bearing_deg=270.0,
        )
        assert isinstance(result, CoastalImpactResult)
        assert result.shortest_distance_to_coast_km > 0
        assert result.landfall_projected is True
        assert result.eta_to_coast_hours is not None
        assert result.eta_to_coast_hours > 0
        assert result.eta_uncertainty_range_hours is not None
        assert result.eta_uncertainty_range_hours[0] <= result.eta_to_coast_hours <= result.eta_uncertainty_range_hours[1]
        assert result.coastal_vulnerability_score >= 0.0

    def test_threatened_assets_detected(self):
        result = assess_coastal_impact(
            spill_lat=12.75,
            spill_lon=80.30,  # Close to Covelong
            current_speed_ms=0.4,
            current_bearing_deg=250.0,
            wind_speed_ms=5.0,
            wind_bearing_deg=250.0,
        )
        assert len(result.threatened_assets) > 0
        assert result.threatened_assets[0].distance_from_spill_km < 35.0

    def test_countermeasures_and_restrictions(self):
        result = assess_coastal_impact(
            spill_lat=12.80,
            spill_lon=80.30,
            current_speed_ms=0.5,
            current_bearing_deg=270.0,
        )
        assert len(result.containment_recommendations) > 0
        assert len(result.dispersant_restrictions) > 0
        # Dispersant restriction must mention shoreline or depth
        assert any("shoreline" in r.lower() or "dispersant" in r.lower() for r in result.dispersant_restrictions)

    def test_pipeline_integration(self):
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
            assert res.get("coastal_done") is True
            coastal = res.get("coastal_impact", {})
            assert "shortest_distance_to_coast_km" in coastal
            assert "threatened_assets" in coastal
            assert "coastal_vulnerability_score" in coastal
