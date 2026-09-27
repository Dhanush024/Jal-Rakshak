"""
Jal-Rakshak — Demo Scenario Builder
=====================================
Generates self-contained demo data for offline demonstration.
All demo data is CLEARLY LABELED as simulated.
"""

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ais.provider import AISRecord, DataMode
from ocean.provider import OceanCurrentPoint, WindPoint
from geospatial.distance import destination_point


# ---------------------------------------------------------------------------
# Demo Scenario Configuration
# ---------------------------------------------------------------------------

@dataclass
class DemoScenario:
    """Complete demo incident scenario."""
    name: str
    description: str
    spill_lat: float
    spill_lon: float
    detection_time: datetime
    drift_duration_minutes: float
    current_speed_ms: float
    current_bearing_deg: float
    wind_speed_ms: float
    wind_bearing_deg: float
    # Coastline reference point for impact calculation
    coast_lat: float
    coast_lon: float
    coast_name: str
    # Sensitive areas
    sensitive_areas: List[Dict]


# Default demo scenario: Chennai coast spill
CHENNAI_SCENARIO = DemoScenario(
    name="Chennai Coast Incident (Demo)",
    description=(
        "Simulated oil spill detection off the coast of Chennai, India. "
        "SAR imagery reveals a dark slick in the Bay of Bengal. "
        "All data in this scenario is SIMULATED for demonstration."
    ),
    spill_lat=12.4500,
    spill_lon=80.2300,
    detection_time=datetime(2026, 9, 14, 15, 30, 0, tzinfo=timezone.utc),
    drift_duration_minutes=72.0,
    current_speed_ms=0.48,
    current_bearing_deg=118.0,
    wind_speed_ms=6.2,
    wind_bearing_deg=135.0,
    coast_lat=12.85,
    coast_lon=80.27,
    coast_name="Chennai Coast / Marina Beach",
    sensitive_areas=[
        {"name": "Pulicat Lake Bird Sanctuary", "lat": 13.42, "lon": 80.32, "type": "wildlife"},
        {"name": "Covelong Fishing Harbor", "lat": 12.79, "lon": 80.25, "type": "fishing"},
        {"name": "Ennore Port", "lat": 13.22, "lon": 80.32, "type": "port"},
    ],
)


# ---------------------------------------------------------------------------
# Demo AIS Track Generation
# ---------------------------------------------------------------------------

def _generate_track(mmsi: str, name: str,
                    start_lat: float, start_lon: float,
                    heading: float, speed_knots: float,
                    start_time: datetime,
                    duration_minutes: int = 300,
                    heading_drift: float = 0.0,
                    speed_variation: float = 0.0,
                    vessel_type: str = "Cargo") -> List[AISRecord]:
    """Generate a minute-by-minute AIS track for one vessel."""
    track = []
    lat, lon = start_lat, start_lon
    current_heading = heading
    rng = random.Random(hash(mmsi))  # deterministic per vessel

    for minute in range(duration_minutes + 1):
        t = start_time + timedelta(minutes=minute)
        spd = max(0.0, speed_knots + rng.uniform(-speed_variation, speed_variation))
        current_heading = (current_heading + heading_drift) % 360

        track.append(AISRecord(
            mmsi=mmsi,
            name=name,
            lat=round(lat, 6),
            lon=round(lon, 6),
            heading=round(current_heading, 1),
            speed_knots=round(spd, 1),
            timestamp=t,
            vessel_type=vessel_type,
        ))

        dist_km = (spd * 1.852) / 60.0
        lat, lon = destination_point(lat, lon, current_heading, dist_km)

    return track


def generate_demo_ais_tracks(scenario: DemoScenario = None) -> Dict[str, List[AISRecord]]:
    """
    Generate 4 simulated vessel tracks for the demo scenario.

    MT Sea Falcon: Primary candidate — passes near source region
    MV Ocean Star: Secondary — nearby but farther
    MV Eastern Wind: Tertiary — crosses region at distance
    FV Horizon: Unlikely — small fishing vessel, far away

    ALL TRACKS ARE SIMULATED.
    """
    if scenario is None:
        scenario = CHENNAI_SCENARIO

    origin_time = scenario.detection_time - timedelta(minutes=scenario.drift_duration_minutes)
    base_time = origin_time - timedelta(hours=3, minutes=18)

    # Estimate origin position (approximate)
    # The actual origin will be computed by the drift model
    origin_lat = scenario.spill_lat - 0.03
    origin_lon = scenario.spill_lon - 0.04

    # MT Sea Falcon (PRIMARY CANDIDATE)
    falcon_speed = 9.2
    falcon_heading = 248.0
    minutes_to_passage = 175
    dist_passage_km = (falcon_speed * 1.852 / 60.0) * minutes_to_passage
    falcon_start_lat, falcon_start_lon = destination_point(
        origin_lat + 0.001, origin_lon + 0.001,
        (falcon_heading + 180) % 360,
        dist_passage_km
    )

    # MV Ocean Star (SECONDARY)
    star_speed = 11.5
    star_heading = 210.0
    dist_star_km = (star_speed * 1.852 / 60.0) * 140
    star_start_lat, star_start_lon = destination_point(
        origin_lat + 0.05, origin_lon - 0.04,
        (star_heading + 180) % 360,
        dist_star_km
    )

    # MV Eastern Wind (TERTIARY)
    wind_speed = 13.8
    wind_heading = 162.0
    dist_wind_km = (wind_speed * 1.852 / 60.0) * 120
    wind_start_lat, wind_start_lon = destination_point(
        origin_lat + 0.12, origin_lon + 0.10,
        (wind_heading + 180) % 360,
        dist_wind_km
    )

    # FV Horizon (UNLIKELY)
    horizon_speed = 5.4
    horizon_heading = 310.0
    dist_hz_km = (horizon_speed * 1.852 / 60.0) * 100
    horizon_start_lat, horizon_start_lon = destination_point(
        origin_lat - 0.20, origin_lon + 0.25,
        (horizon_heading + 180) % 360,
        dist_hz_km
    )

    return {
        "419001234": _generate_track(
            "419001234", "MT Sea Falcon",
            falcon_start_lat, falcon_start_lon,
            falcon_heading, falcon_speed, base_time,
            duration_minutes=300, heading_drift=0.02, speed_variation=0.3,
            vessel_type="Oil Tanker",
        ),
        "419005678": _generate_track(
            "419005678", "MV Ocean Star",
            star_start_lat, star_start_lon,
            star_heading, star_speed, base_time,
            duration_minutes=300, heading_drift=-0.05, speed_variation=0.5,
            vessel_type="Cargo",
        ),
        "538003210": _generate_track(
            "538003210", "MV Eastern Wind",
            wind_start_lat, wind_start_lon,
            wind_heading, wind_speed, base_time,
            duration_minutes=300, heading_drift=0.08, speed_variation=0.8,
            vessel_type="Container",
        ),
        "412009876": _generate_track(
            "412009876", "FV Horizon",
            horizon_start_lat, horizon_start_lon,
            horizon_heading, horizon_speed, base_time,
            duration_minutes=300, heading_drift=-0.15, speed_variation=1.0,
            vessel_type="Fishing",
        ),
    }


def generate_demo_ocean_data(scenario: DemoScenario = None) -> Tuple[List[OceanCurrentPoint], List[WindPoint]]:
    """Generate simulated ocean current and wind data for the demo."""
    if scenario is None:
        scenario = CHENNAI_SCENARIO

    currents = []
    winds = []

    # Generate a 5x5 grid of current/wind points around the spill
    for dlat in [-0.1, -0.05, 0, 0.05, 0.1]:
        for dlon in [-0.1, -0.05, 0, 0.05, 0.1]:
            lat = scenario.spill_lat + dlat
            lon = scenario.spill_lon + dlon

            # Add some spatial variation
            rng = random.Random(hash((lat, lon)))
            speed_var = rng.uniform(-0.08, 0.08)
            bearing_var = rng.uniform(-15, 15)

            currents.append(OceanCurrentPoint(
                lat=lat, lon=lon,
                timestamp=scenario.detection_time,
                speed_ms=max(0.1, scenario.current_speed_ms + speed_var),
                direction_deg=(scenario.current_bearing_deg + bearing_var) % 360,
            ))

            winds.append(WindPoint(
                lat=lat, lon=lon,
                timestamp=scenario.detection_time,
                speed_ms=max(0.5, scenario.wind_speed_ms + rng.uniform(-1.5, 1.5)),
                direction_deg=(scenario.wind_bearing_deg + rng.uniform(-20, 20)) % 360,
            ))

    return currents, winds


def generate_demo_coastline() -> List[Dict]:
    """Generate simplified coastline points for the Chennai demo."""
    # Simplified east coast of India around Chennai
    return [
        {"name": "Pulicat", "lat": 13.42, "lon": 80.32},
        {"name": "Ennore", "lat": 13.22, "lon": 80.33},
        {"name": "Chennai North", "lat": 13.12, "lon": 80.30},
        {"name": "Marina Beach", "lat": 13.05, "lon": 80.28},
        {"name": "Adyar Estuary", "lat": 13.00, "lon": 80.27},
        {"name": "Thiruvanmiyur", "lat": 12.98, "lon": 80.26},
        {"name": "Covelong", "lat": 12.79, "lon": 80.25},
        {"name": "Mahabalipuram", "lat": 12.62, "lon": 80.19},
    ]


def get_or_create_demo_sar_patch(output_path: Optional[str] = None) -> str:
    """
    Get or create a synthetic Sentinel-1 SAR image patch for the Chennai demo scenario.
    Features:
    - Rayleigh-distributed oceanic sea-clutter speckle noise
    - Dark irregular oil slick region (damping of surface capillary waves)
    - High-reflectivity vessel target (point scatterer)
    """
    import os
    import numpy as np
    import cv2

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_sar_patch.png")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = 512, 512
    # Rayleigh-like speckle background
    rng = np.random.default_rng(42)
    base = rng.rayleigh(scale=65, size=(h, w))
    base = np.clip(base, 20, 240).astype(np.uint8)

    # Dark slick polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[180, 200], [220, 160], [310, 180], [360, 250], [330, 320], [240, 310], [190, 260]], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # Apply dampening in slick area (oil damps Bragg waves -> dark backscatter)
    slick = (base.astype(np.float32) * (1.0 - 0.7 * (mask.astype(np.float32) / 255.0))).astype(np.uint8)

    # Add vessel point target (bright pixel cluster)
    slick[150:154, 170:174] = 255

    cv2.imwrite(output_path, slick)
    return output_path
