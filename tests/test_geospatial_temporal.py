"""
Unit and Regression Tests for Geospatial Temporal Engine
=========================================================
Validates deterministic temporal advection, time-slice geometry divergence,
track sorting and geographic boundary calculations according to Phase 20 specifications.
"""

import pytest
from datetime import datetime, timezone, timedelta
from geospatial.temporal import (
    validate_lat_lon,
    parse_time_slice_hours,
    calculate_drift_vector,
    compute_temporal_position,
    compute_drift_trajectory_points,
    compute_all_time_slice_geometries,
    sort_and_validate_track,
    find_vessel_position_at_epoch,
    calculate_map_bounds,
    STANDARD_TIME_SLICES,
)


class TestGeospatialTemporalEngine:
    """Test suite covering Phase 20 regression requirements."""

    def test_lat_lon_validation(self):
        """Validates geographic latitude and longitude range constraints."""
        assert validate_lat_lon(13.0827, 80.2707) is True
        assert validate_lat_lon(-90.0, 180.0) is True
        assert validate_lat_lon(0.0, 0.0) is True
        assert validate_lat_lon(91.0, 80.0) is False
        assert validate_lat_lon(-90.1, 0.0) is False
        assert validate_lat_lon(13.0, 181.0) is False
        assert validate_lat_lon(13.0, -180.5) is False
        assert validate_lat_lon("not_a_number", 80.0) is False
        assert validate_lat_lon(None, None) is False

    def test_parse_time_slice_hours(self):
        """Verify time-slice parsing for standard forensic milestones."""
        assert parse_time_slice_hours("T-3H") == -3.0
        assert parse_time_slice_hours("T-3") == -3.0
        assert parse_time_slice_hours("T0") == 0.0
        assert parse_time_slice_hours("T+6H") == 6.0
        assert parse_time_slice_hours("T+6") == 6.0
        assert parse_time_slice_hours("T+12H") == 12.0
        assert parse_time_slice_hours("T+24H") == 24.0
        assert parse_time_slice_hours("UNKNOWN") == 0.0

    def test_deterministic_demo_drift_calculation(self):
        """Exact same inputs must produce exact same deterministic drift vector."""
        vec1 = calculate_drift_vector(0.48, 118.0, 6.2, 135.0, 0.03)
        vec2 = calculate_drift_vector(0.48, 118.0, 6.2, 135.0, 0.03)
        assert vec1["net_drift_speed_ms"] == vec2["net_drift_speed_ms"]
        assert vec1["net_drift_bearing_deg"] == vec2["net_drift_bearing_deg"]
        assert vec1["reverse_bearing_deg"] == vec2["reverse_bearing_deg"]
        assert vec1["net_drift_speed_ms"] > 0.40

    def test_time_slice_geometry_divergence_invariants(self):
        """
        Hard Phase 20 invariant:
        geometry(T0) != geometry(T+6)
        geometry(T+6) != geometry(T+12)
        geometry(T+12) != geometry(T+24)
        when a non-zero current vector exists.
        """
        spill_lat, spill_lon = 12.4500, 80.2300
        speed_ms = 0.52
        bearing_deg = 118.0

        geo_t_minus_3 = compute_temporal_position(spill_lat, spill_lon, speed_ms, bearing_deg, -3.0)
        geo_t0 = compute_temporal_position(spill_lat, spill_lon, speed_ms, bearing_deg, 0.0)
        geo_t_plus_6 = compute_temporal_position(spill_lat, spill_lon, speed_ms, bearing_deg, 6.0)
        geo_t_plus_12 = compute_temporal_position(spill_lat, spill_lon, speed_ms, bearing_deg, 12.0)
        geo_t_plus_24 = compute_temporal_position(spill_lat, spill_lon, speed_ms, bearing_deg, 24.0)

        # Centroid at T0 must equal the observed spill coordinates
        assert abs(geo_t0[0] - spill_lat) < 1e-6
        assert abs(geo_t0[1] - spill_lon) < 1e-6

        # Invariant 1: T0 != T+6H
        assert (geo_t0[0], geo_t0[1]) != (geo_t_plus_6[0], geo_t_plus_6[1])

        # Invariant 2: T+6H != T+12H
        assert (geo_t_plus_6[0], geo_t_plus_6[1]) != (geo_t_plus_12[0], geo_t_plus_12[1])

        # Invariant 3: T+12H != T+24H
        assert (geo_t_plus_12[0], geo_t_plus_12[1]) != (geo_t_plus_24[0], geo_t_plus_24[1])

        # Invariant 4: T-3H (origin) != T0
        assert (geo_t_minus_3[0], geo_t_minus_3[1]) != (geo_t0[0], geo_t0[1])

        # Forward movement must move monotonically along the trajectory direction
        assert geo_t_plus_6[0] < geo_t0[0]  # bearing 118 is South-East, so latitude decreases
        assert geo_t_plus_12[0] < geo_t_plus_6[0]
        assert geo_t_plus_24[0] < geo_t_plus_12[0]

        # Uncertainty must grow with time horizon
        assert geo_t0[2] < geo_t_plus_6[2] < geo_t_plus_12[2] < geo_t_plus_24[2]

    def test_compute_all_time_slice_geometries(self):
        """Verify bulk dictionary generation for all milestone time slices."""
        slices_geo = compute_all_time_slice_geometries(12.45, 80.23, 0.48, 118.0)
        assert len(slices_geo) == len(STANDARD_TIME_SLICES)
        for s in STANDARD_TIME_SLICES:
            assert s in slices_geo
            assert validate_lat_lon(slices_geo[s]["lat"], slices_geo[s]["lon"]) is True
            assert slices_geo[s]["uncertainty_km"] > 0

    def test_drift_trajectory_points_generation(self):
        """Ensure smooth, ordered coordinates along trajectory."""
        pts = compute_drift_trajectory_points(12.45, 80.23, 0.48, 118.0, 0.0, 24.0, steps=10)
        assert len(pts) == 11
        for lat, lon in pts:
            assert validate_lat_lon(lat, lon) is True

    def test_vessel_track_timestamp_ordering_and_validation(self):
        """Vessel tracks must be sorted ascending by timestamp and coordinates validated."""
        unordered_raw = [
            {"lat": 12.50, "lon": 80.30, "timestamp": "2026-09-14T17:00:00Z"},
            {"lat": 12.40, "lon": 80.20, "timestamp": "2026-09-14T14:00:00Z"},
            {"lat": 95.00, "lon": 80.25, "timestamp": "2026-09-14T15:00:00Z"},  # Invalid lat > 90
            {"lat": 12.45, "lon": 80.25, "timestamp": "2026-09-14T15:30:00Z"},
            {"lat": 12.45, "lon": 80.25, "timestamp": "2026-09-14T15:30:00Z"},  # Duplicate
        ]
        cleaned = sort_and_validate_track(unordered_raw)
        assert len(cleaned) == 3
        # Strict ascending order
        assert cleaned[0]["timestamp"] == "2026-09-14T14:00:00Z"
        assert cleaned[1]["timestamp"] == "2026-09-14T15:30:00Z"
        assert cleaned[2]["timestamp"] == "2026-09-14T17:00:00Z"

    def test_find_vessel_position_at_epoch(self):
        """Finds closest track point for a target timestamp."""
        sorted_track = [
            {"lat": 12.40, "lon": 80.20, "timestamp": "2026-09-14T14:00:00+00:00"},
            {"lat": 12.45, "lon": 80.25, "timestamp": "2026-09-14T15:30:00+00:00"},
            {"lat": 12.50, "lon": 80.30, "timestamp": "2026-09-14T17:00:00+00:00"},
        ]
        t_target = datetime(2026, 9, 14, 15, 25, tzinfo=timezone.utc)
        closest = find_vessel_position_at_epoch(sorted_track, t_target)
        assert closest is not None
        assert closest["lat"] == 12.45

    def test_calculate_map_bounds(self):
        """Map bounding box must enclose all points with sensible padding."""
        geometries = [
            {"lat": 12.40, "lon": 80.20},
            {"lat": 12.60, "lon": 80.40},
            [12.50, 80.30],
        ]
        (min_lat, min_lon), (max_lat, max_lon) = calculate_map_bounds(geometries)
        assert min_lat < 12.40
        assert max_lat > 12.60
        assert min_lon < 80.20
        assert max_lon > 80.40
        assert validate_lat_lon(min_lat, min_lon) is True
        assert validate_lat_lon(max_lat, max_lon) is True

    def test_build_investigation_map_time_slices(self):
        """Verify that build_investigation_map renders distinct time slices without error."""
        from app import build_investigation_map

        demo_state = {
            "spill_detected": True,
            "spill_lat": 12.45,
            "spill_lon": 80.23,
            "source_lat": 12.48,
            "source_lon": 80.21,
            "source_uncertainty_km": 3.5,
            "hindcast_done": True,
            "ocean_current_u": 0.45,
            "ocean_current_v": -0.22,
        }

        # Build maps for each time slice
        rendered_htmls = {}
        for ts in ["T-3H", "T0", "T+6H", "T+12H", "T+24H"]:
            fmap = build_investigation_map(
                state=demo_state,
                mode="SOURCE",
                is_demo=True,
                time_slice=ts,
            )
            assert fmap is not None
            html = fmap.get_root().render()
            rendered_htmls[ts] = html
            # Check that the specific milestone marker is active in the rendered map
            assert f"Active Milestone: {ts}" in html

        # Verify that different time slice maps have different HTML content (geometries and active nodes)
        assert rendered_htmls["T-3H"] != rendered_htmls["T0"]
        assert rendered_htmls["T0"] != rendered_htmls["T+6H"]
        assert rendered_htmls["T+6H"] != rendered_htmls["T+12H"]
        assert rendered_htmls["T+12H"] != rendered_htmls["T+24H"]
