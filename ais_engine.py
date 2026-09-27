"""
Jal-Rakshak AIS Correlation Engine
===================================
Simulated AIS tracks, oil spill drift model, and forensic correlation scoring.
Designed for deterministic hackathon demo with realistic maritime data.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AISRecord:
    """Single AIS position report for a vessel at a point in time."""
    mmsi: str
    name: str
    lat: float
    lon: float
    heading: float        # degrees, 0 = north, clockwise
    speed_knots: float
    timestamp: datetime   # UTC

    def to_dict(self) -> dict:
        return {
            "mmsi": self.mmsi,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "heading": self.heading,
            "speed_knots": self.speed_knots,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DriftResult:
    """Output of the backward-drift model."""
    origin_lat: float
    origin_lon: float
    origin_time: datetime
    temporal_uncertainty_min: float   # ± minutes
    spatial_uncertainty_km: float     # ± km
    current_speed_ms: float           # m/s
    current_bearing_deg: float        # degrees

    def to_dict(self) -> dict:
        return {
            "origin_lat": round(self.origin_lat, 4),
            "origin_lon": round(self.origin_lon, 4),
            "origin_time": self.origin_time.isoformat(),
            "origin_time_str": self.origin_time.strftime("%H:%M UTC"),
            "temporal_uncertainty_min": self.temporal_uncertainty_min,
            "spatial_uncertainty_km": self.spatial_uncertainty_km,
            "current_speed_ms": self.current_speed_ms,
            "current_bearing_deg": self.current_bearing_deg,
        }


@dataclass
class VesselCorrelation:
    """Correlation result for a single vessel."""
    mmsi: str
    name: str
    score: float                           # 0-100
    closest_distance_km: float
    closest_time: datetime
    time_diff_min: float                   # minutes from origin
    heading_at_closest: float
    speed_at_closest: float
    lat_at_origin_time: float
    lon_at_origin_time: float
    trajectory_match: bool
    time_match: bool
    spatial_match: bool
    drift_consistency: bool

    def to_dict(self) -> dict:
        return {
            "mmsi": self.mmsi,
            "name": self.name,
            "score": round(self.score, 1),
            "closest_distance_km": round(self.closest_distance_km, 2),
            "closest_time": self.closest_time.isoformat(),
            "closest_time_str": self.closest_time.strftime("%H:%M UTC"),
            "time_diff_min": round(self.time_diff_min, 0),
            "heading_at_closest": round(self.heading_at_closest, 1),
            "speed_at_closest": round(self.speed_at_closest, 1),
            "lat_at_origin_time": round(self.lat_at_origin_time, 4),
            "lon_at_origin_time": round(self.lon_at_origin_time, 4),
            "trajectory_match": self.trajectory_match,
            "time_match": self.time_match,
            "spatial_match": self.spatial_match,
            "drift_consistency": self.drift_consistency,
        }


# ---------------------------------------------------------------------------
# Geographic utilities
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r) -
         math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination_point(lat: float, lon: float, bearing_d: float,
                      distance_km: float) -> Tuple[float, float]:
    """Move from (lat, lon) along bearing by distance_km. Returns (lat, lon)."""
    R = 6371.0
    d = distance_km / R
    br = math.radians(bearing_d)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) +
                     math.cos(lat1) * math.sin(d) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


# ---------------------------------------------------------------------------
# Simulated AIS track generation
# ---------------------------------------------------------------------------

def _generate_track(mmsi: str, name: str,
                    start_lat: float, start_lon: float,
                    heading: float, speed_knots: float,
                    start_time: datetime,
                    duration_minutes: int = 300,
                    heading_drift: float = 0.0,
                    speed_variation: float = 0.0) -> List[AISRecord]:
    """
    Generate a minute-by-minute AIS track for one vessel.
    heading_drift: degrees per minute of gradual course change
    speed_variation: ± knots random variation
    """
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
        ))

        # Advance position: speed in knots → km/min, then move
        dist_km = (spd * 1.852) / 60.0  # knots to km/h, then per minute
        lat, lon = destination_point(lat, lon, current_heading, dist_km)

    return track


def generate_simulated_ais_tracks(
    origin_lat: float = 12.4000,
    origin_lon: float = 80.2000,
    origin_time: Optional[datetime] = None,
) -> Dict[str, List[AISRecord]]:
    """
    Generate 4 simulated vessel tracks around the spill origin.

    MT Sea Falcon: designed to pass through the origin zone ~23 min before origin_time.
    MV Ocean Star: passes nearby but farther out.
    MV Eastern Wind: crosses region but at a wider distance.
    FV Horizon: small fishing vessel, far from origin.
    """
    if origin_time is None:
        origin_time = datetime(2026, 9, 14, 14, 18, 0, tzinfo=timezone.utc)

    base_time = origin_time - timedelta(hours=3, minutes=18)  # ≈ 11:00 UTC

    # --- MT Sea Falcon (PRIMARY SUSPECT) ---
    # This vessel sails heading 248° and should pass within ~1 km of origin
    # at approximately minute 175 from base_time (≈ 13:55 UTC).
    # We place its start so that its track goes almost THROUGH the origin point.
    falcon_speed = 9.2
    falcon_heading = 248.0
    minutes_to_passage = 175
    dist_passage_km = (falcon_speed * 1.852 / 60.0) * minutes_to_passage
    # The vessel at minute 175 should be AT the origin (tiny offset for realism)
    falcon_start_lat, falcon_start_lon = destination_point(
        origin_lat + 0.001, origin_lon + 0.001,  # ~150m offset — nearly dead-on
        (falcon_heading + 180) % 360,             # reverse heading to find start
        dist_passage_km
    )

    # --- MV Ocean Star (SECONDARY) ---
    # Heading 210° (SW), passes ~5-7 km from origin. Close enough to be
    # suspicious but clearly farther than Falcon.
    star_speed = 11.5
    star_heading = 210.0  # angled away from origin zone
    dist_star_km = (star_speed * 1.852 / 60.0) * 140
    star_start_lat, star_start_lon = destination_point(
        origin_lat + 0.05, origin_lon - 0.04,
        (star_heading + 180) % 360,
        dist_star_km
    )

    # --- MV Eastern Wind (TERTIARY) ---
    # Heading 162° (SSE), passes ~12-15 km from origin. In the region but
    # no strong correlation.
    wind_speed = 13.8
    wind_heading = 162.0
    dist_wind_km = (wind_speed * 1.852 / 60.0) * 120
    wind_start_lat, wind_start_lon = destination_point(
        origin_lat + 0.12, origin_lon + 0.10,
        (wind_heading + 180) % 360,
        dist_wind_km
    )

    # --- FV Horizon (UNLIKELY) ---
    # Small fishing vessel heading 310° (NW), ~25 km from origin.
    # Background noise to show the system doesn't flag everything.
    horizon_speed = 5.4
    horizon_heading = 310.0
    dist_hz_km = (horizon_speed * 1.852 / 60.0) * 100
    horizon_start_lat, horizon_start_lon = destination_point(
        origin_lat - 0.20, origin_lon + 0.25,
        (horizon_heading + 180) % 360,
        dist_hz_km
    )

    tracks = {
        "419001234": _generate_track(
            "419001234", "MT Sea Falcon",
            falcon_start_lat, falcon_start_lon,
            falcon_heading, falcon_speed, base_time,
            duration_minutes=300, heading_drift=0.02, speed_variation=0.3
        ),
        "419005678": _generate_track(
            "419005678", "MV Ocean Star",
            star_start_lat, star_start_lon,
            star_heading, star_speed, base_time,
            duration_minutes=300, heading_drift=-0.05, speed_variation=0.5
        ),
        "538003210": _generate_track(
            "538003210", "MV Eastern Wind",
            wind_start_lat, wind_start_lon,
            wind_heading, wind_speed, base_time,
            duration_minutes=300, heading_drift=0.08, speed_variation=0.8
        ),
        "412009876": _generate_track(
            "412009876", "FV Horizon",
            horizon_start_lat, horizon_start_lon,
            horizon_heading, horizon_speed, base_time,
            duration_minutes=300, heading_drift=-0.15, speed_variation=1.0
        ),
    }
    return tracks


# ---------------------------------------------------------------------------
# Oil spill drift model (backward estimation)
# ---------------------------------------------------------------------------

class DriftModel:
    """
    Simplified backward-drift model.
    Given the current spill position and detection time, estimates origin.
    Uses surface current + wind factor to back-track the slick.
    """
    def __init__(self,
                 current_speed_ms: float = 0.48,
                 current_bearing_deg: float = 118.0,
                 wind_factor: float = 0.03,
                 wind_speed_ms: float = 6.2,
                 wind_bearing_deg: float = 135.0):
        self.current_speed_ms = current_speed_ms
        self.current_bearing_deg = current_bearing_deg
        self.wind_factor = wind_factor
        self.wind_speed_ms = wind_speed_ms
        self.wind_bearing_deg = wind_bearing_deg

    def estimate_origin(self,
                        spill_lat: float,
                        spill_lon: float,
                        detection_time: Optional[datetime] = None,
                        drift_duration_minutes: float = 72.0
                        ) -> DriftResult:
        """
        Back-track the spill to estimate where and when it originated.
        """
        if detection_time is None:
            detection_time = datetime(2026, 9, 14, 15, 30, 0, tzinfo=timezone.utc)

        # Combined drift vector (current + wind-induced)
        # Current component
        cx = self.current_speed_ms * math.sin(math.radians(self.current_bearing_deg))
        cy = self.current_speed_ms * math.cos(math.radians(self.current_bearing_deg))
        # Wind component
        wx = self.wind_factor * self.wind_speed_ms * math.sin(math.radians(self.wind_bearing_deg))
        wy = self.wind_factor * self.wind_speed_ms * math.cos(math.radians(self.wind_bearing_deg))
        # Total drift velocity
        vx = cx + wx
        vy = cy + wy
        drift_speed = math.sqrt(vx**2 + vy**2)
        drift_bearing = math.degrees(math.atan2(vx, vy)) % 360

        # Total drift distance over the given duration
        drift_distance_km = drift_speed * (drift_duration_minutes * 60) / 1000.0

        # Reverse the drift: origin is *opposite* the drift direction
        reverse_bearing = (drift_bearing + 180) % 360
        origin_lat, origin_lon = destination_point(
            spill_lat, spill_lon, reverse_bearing, drift_distance_km
        )

        origin_time = detection_time - timedelta(minutes=drift_duration_minutes)

        # Uncertainty grows with drift duration
        temporal_uncertainty = round(drift_duration_minutes * 0.3, 0)  # ~30% of duration
        spatial_uncertainty = round(drift_distance_km * 0.35, 1)       # ~35% of distance

        return DriftResult(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_time=origin_time,
            temporal_uncertainty_min=temporal_uncertainty,
            spatial_uncertainty_km=spatial_uncertainty,
            current_speed_ms=self.current_speed_ms,
            current_bearing_deg=self.current_bearing_deg,
        )


# ---------------------------------------------------------------------------
# Correlation scoring
# ---------------------------------------------------------------------------

def calculate_correlation_score(
    track: List[AISRecord],
    origin: DriftResult,
) -> VesselCorrelation:
    """
    Calculate how strongly a vessel's track correlates with the estimated spill origin.

    Score = weighted combination of:
      - temporal proximity   (30%)  — was the vessel near the origin at origin_time?
      - spatial proximity    (25%)  — minimum distance to origin point
      - trajectory intersect (20%)  — does the track cross the uncertainty ellipse?
      - heading consistency  (10%)  — heading relative to origin
      - speed plausibility   (5%)   — was it moving? (not anchored)
      - drift consistency    (10%)  — does spill drift direction match?
    """
    origin_time = origin.origin_time
    origin_lat = origin.origin_lat
    origin_lon = origin.origin_lon
    uncertainty_km = origin.spatial_uncertainty_km

    # Find the record closest in time to origin_time
    closest_record = min(track, key=lambda r: abs((r.timestamp - origin_time).total_seconds()))
    closest_dist = haversine_km(closest_record.lat, closest_record.lon, origin_lat, origin_lon)
    time_diff_sec = abs((closest_record.timestamp - origin_time).total_seconds())
    time_diff_min = time_diff_sec / 60.0

    # Find minimum distance record (spatial closest approach)
    min_dist_record = min(track, key=lambda r: haversine_km(r.lat, r.lon, origin_lat, origin_lon))
    min_dist_km = haversine_km(min_dist_record.lat, min_dist_record.lon, origin_lat, origin_lon)

    # Get vessel position at origin_time
    at_origin = _interpolate_at_time(track, origin_time)

    # --- Score components ---

    # 1. Temporal proximity (30%): how close in time was the vessel's nearest approach?
    if time_diff_min <= 15:
        temporal_score = 100
    elif time_diff_min <= 30:
        temporal_score = 100 - (time_diff_min - 15) * 3.33
    elif time_diff_min <= 60:
        temporal_score = 50 - (time_diff_min - 30) * 1.67
    else:
        temporal_score = max(0, 20 - (time_diff_min - 60) * 0.33)

    # 2. Spatial proximity (25%): minimum distance to origin
    if min_dist_km <= 1.0:
        spatial_score = 100
    elif min_dist_km <= 3.0:
        spatial_score = 100 - (min_dist_km - 1.0) * 25
    elif min_dist_km <= 10.0:
        spatial_score = 50 - (min_dist_km - 3.0) * 7.14
    else:
        spatial_score = max(0, 10 - (min_dist_km - 10.0) * 1.0)

    # 3. Trajectory intersection (20%): does any point cross the uncertainty ellipse?
    crosses_ellipse = any(
        haversine_km(r.lat, r.lon, origin_lat, origin_lon) <= uncertainty_km
        for r in track
    )
    trajectory_score = 100 if crosses_ellipse else 20

    # 4. Heading consistency (10%): is vessel heading roughly pointing at/through origin?
    if min_dist_km < 0.5:
        heading_score = 90  # too close for heading to matter much
    else:
        bearing_to_origin = bearing_deg(
            min_dist_record.lat, min_dist_record.lon, origin_lat, origin_lon
        )
        heading_diff = abs(min_dist_record.heading - bearing_to_origin)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff
        # Smaller heading diff → higher score. Allow ±45° as "consistent"
        if heading_diff <= 45:
            heading_score = 100 - heading_diff * 1.11
        elif heading_diff <= 90:
            heading_score = 50 - (heading_diff - 45) * 1.11
        else:
            heading_score = max(0, 10)

    # 5. Speed plausibility (5%): moving vessel more likely source than anchored
    avg_speed = sum(r.speed_knots for r in track) / len(track)
    if 3.0 <= avg_speed <= 15.0:
        speed_score = 80
    elif avg_speed > 15.0:
        speed_score = 60
    elif avg_speed > 0.5:
        speed_score = 40
    else:
        speed_score = 10

    # 6. Drift consistency (10%): does spill drift direction match origin→spill bearing?
    origin_to_spill_bearing = bearing_deg(origin_lat, origin_lon,
                                          origin_lat + 0.01, origin_lon + 0.01)  # approximate
    drift_bearing_diff = abs(origin.current_bearing_deg - origin_to_spill_bearing)
    if drift_bearing_diff > 180:
        drift_bearing_diff = 360 - drift_bearing_diff
    drift_score = max(0, 100 - drift_bearing_diff * 0.8) if min_dist_km < 5 else 30

    # --- Weighted total ---
    total = (
        temporal_score * 0.30 +
        spatial_score * 0.25 +
        trajectory_score * 0.20 +
        heading_score * 0.10 +
        speed_score * 0.05 +
        drift_score * 0.10
    )

    # Boolean match indicators
    trajectory_match = crosses_ellipse
    time_match = time_diff_min <= 30
    spatial_match = min_dist_km <= uncertainty_km * 1.5
    drift_ok = drift_score > 50

    return VesselCorrelation(
        mmsi=track[0].mmsi,
        name=track[0].name,
        score=total,
        closest_distance_km=min_dist_km,
        closest_time=min_dist_record.timestamp,
        time_diff_min=time_diff_min,
        heading_at_closest=min_dist_record.heading,
        speed_at_closest=min_dist_record.speed_knots,
        lat_at_origin_time=at_origin[0],
        lon_at_origin_time=at_origin[1],
        trajectory_match=trajectory_match,
        time_match=time_match,
        spatial_match=spatial_match,
        drift_consistency=drift_ok,
    )


def _interpolate_at_time(
    track: List[AISRecord], target_time: datetime
) -> Tuple[float, float, float, float]:
    """
    Linearly interpolate vessel position at target_time.
    Returns (lat, lon, heading, speed).
    """
    # Clamp to track bounds
    if target_time <= track[0].timestamp:
        r = track[0]
        return r.lat, r.lon, r.heading, r.speed_knots
    if target_time >= track[-1].timestamp:
        r = track[-1]
        return r.lat, r.lon, r.heading, r.speed_knots

    # Find the bounding records
    for i in range(len(track) - 1):
        if track[i].timestamp <= target_time <= track[i + 1].timestamp:
            t0 = track[i].timestamp
            t1 = track[i + 1].timestamp
            frac = (target_time - t0).total_seconds() / max(1, (t1 - t0).total_seconds())
            lat = track[i].lat + frac * (track[i + 1].lat - track[i].lat)
            lon = track[i].lon + frac * (track[i + 1].lon - track[i].lon)
            hdg = track[i].heading + frac * (track[i + 1].heading - track[i].heading)
            spd = track[i].speed_knots + frac * (track[i + 1].speed_knots - track[i].speed_knots)
            return lat, lon, hdg % 360, spd
    # Fallback
    r = track[-1]
    return r.lat, r.lon, r.heading, r.speed_knots


# ---------------------------------------------------------------------------
# Public API for the pipeline and UI
# ---------------------------------------------------------------------------

def rank_vessels(
    tracks: Dict[str, List[AISRecord]],
    origin: DriftResult,
) -> List[VesselCorrelation]:
    """Score and rank all vessels by correlation to the spill origin."""
    correlations = []
    for mmsi, track in tracks.items():
        corr = calculate_correlation_score(track, origin)
        correlations.append(corr)
    correlations.sort(key=lambda c: c.score, reverse=True)
    return correlations


def get_vessel_at_time(
    track: List[AISRecord], target_time: datetime
) -> dict:
    """
    Get interpolated vessel state at a specific time.
    Returns a dict suitable for display in the UI.
    """
    lat, lon, heading, speed = _interpolate_at_time(track, target_time)
    return {
        "mmsi": track[0].mmsi,
        "name": track[0].name,
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "heading": round(heading, 1),
        "speed_knots": round(speed, 1),
        "timestamp": target_time.isoformat(),
        "timestamp_str": target_time.strftime("%H:%M UTC"),
    }


def get_breadcrumb_trail(
    track: List[AISRecord],
    target_time: datetime,
    window_minutes: int = 60,
) -> List[dict]:
    """
    Get the last `window_minutes` of positions before `target_time`.
    Each entry has an opacity value (1.0 = most recent, fading to 0.2).
    """
    cutoff = target_time - timedelta(minutes=window_minutes)
    trail_records = [
        r for r in track
        if cutoff <= r.timestamp <= target_time
    ]
    if not trail_records:
        return []

    total = len(trail_records)
    trail = []
    for i, r in enumerate(trail_records):
        opacity = 0.2 + 0.8 * (i / max(1, total - 1))
        trail.append({
            "lat": r.lat,
            "lon": r.lon,
            "heading": r.heading,
            "speed_knots": r.speed_knots,
            "timestamp": r.timestamp.isoformat(),
            "opacity": round(opacity, 2),
        })
    return trail


def get_track_time_bounds(
    tracks: Dict[str, List[AISRecord]]
) -> Tuple[datetime, datetime]:
    """Get the overall time range covered by all tracks."""
    all_times = []
    for track in tracks.values():
        all_times.append(track[0].timestamp)
        all_times.append(track[-1].timestamp)
    return min(all_times), max(all_times)
