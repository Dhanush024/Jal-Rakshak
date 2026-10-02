"""
Jal-Rakshak — Deterministic Geospatial Temporal Geometry Engine
==============================================================
Pure mathematical, geographic, and temporal modeling for oil advection,
backtracking, forward forecast trajectories, and vessel track alignment.

Guarantees:
- Zero fabricated telemetry
- Exact deterministic calculations from physical inputs
- Geodesic spherical trigonometry (via destination_point)
- Rigorous coordinate validation (-90 <= lat <= 90, -180 <= lon <= 180)
- Temporal timestamp ordering
"""

import math
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Any, Optional

from geospatial.distance import destination_point, haversine_km, bearing_deg


STANDARD_TIME_SLICES = ["T-3H", "T0", "T+6H", "T+12H", "T+24H"]

TIME_SLICE_HOURS_MAP = {
    "T-3H": -3.0,
    "T-3": -3.0,
    "T0": 0.0,
    "T0H": 0.0,
    "NOW": 0.0,
    "T+6H": 6.0,
    "T+6": 6.0,
    "T+12H": 12.0,
    "T+12": 12.0,
    "T+24H": 24.0,
    "T+24": 24.0,
}


def validate_lat_lon(lat: Any, lon: Any) -> bool:
    """Validate latitude and longitude ranges: [-90, 90] and [-180, 180]."""
    try:
        f_lat = float(lat)
        f_lon = float(lon)
        if math.isnan(f_lat) or math.isnan(f_lon) or math.isinf(f_lat) or math.isinf(f_lon):
            return False
        return (-90.0 <= f_lat <= 90.0) and (-180.0 <= f_lon <= 180.0)
    except (TypeError, ValueError):
        return False


def parse_time_slice_hours(time_slice: str) -> float:
    """Convert time-slice identifier ('T-3H', 'T0', 'T+6H', etc.) to elapsed hours."""
    if not isinstance(time_slice, str):
        try:
            return float(time_slice)
        except (TypeError, ValueError):
            return 0.0
    key = time_slice.strip().upper()
    return TIME_SLICE_HOURS_MAP.get(key, 0.0)


def calculate_drift_vector(
    current_speed_ms: float,
    current_bearing_deg: float,
    wind_speed_ms: float = 0.0,
    wind_bearing_deg: float = 0.0,
    wind_factor: float = 0.03,
) -> Dict[str, float]:
    """
    Computes deterministic net surface advection vector combining ocean surface current
    and atmospheric wind leeway (3% rule).
    """
    c_rad = math.radians(current_bearing_deg)
    cx = current_speed_ms * math.sin(c_rad)
    cy = current_speed_ms * math.cos(c_rad)

    w_rad = math.radians(wind_bearing_deg)
    wx = wind_factor * wind_speed_ms * math.sin(w_rad)
    wy = wind_factor * wind_speed_ms * math.cos(w_rad)

    vx = cx + wx
    vy = cy + wy
    net_speed = math.sqrt(vx * vx + vy * vy)
    net_bearing = (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0
    reverse_bearing = (net_bearing + 180.0) % 360.0

    return {
        "net_drift_speed_ms": net_speed,
        "net_drift_bearing_deg": net_bearing,
        "reverse_bearing_deg": reverse_bearing,
        "net_speed_knots": net_speed * 1.94384,
    }


def compute_temporal_position(
    spill_lat: float,
    spill_lon: float,
    net_drift_speed_ms: float,
    net_drift_bearing_deg: float,
    elapsed_hours: float,
    base_uncertainty_km: float = 0.8,
) -> Tuple[float, float, float]:
    """
    Calculate modeled slick centroid position and dispersion uncertainty at a given elapsed time.
    - elapsed_hours < 0: Hydrodynamic backtrack along reverse bearing.
    - elapsed_hours == 0: Observed detection position (T0).
    - elapsed_hours > 0: Forward advection along net drift bearing.
    Returns (lat, lon, uncertainty_km).
    """
    if not validate_lat_lon(spill_lat, spill_lon):
        raise ValueError(f"Invalid spill coordinates: [{spill_lat}, {spill_lon}]")

    h = float(elapsed_hours)
    if abs(h) < 1e-4 or net_drift_speed_ms <= 0.0:
        return float(spill_lat), float(spill_lon), float(base_uncertainty_km)

    bearing = (net_drift_bearing_deg + 180.0) % 360.0 if h < 0.0 else (net_drift_bearing_deg % 360.0)
    distance_km = (net_drift_speed_ms * abs(h) * 3600.0) / 1000.0

    target_lat, target_lon = destination_point(spill_lat, spill_lon, bearing, distance_km)
    # Dispersion uncertainty grows linearly with elapsed advection time/distance
    uncertainty_km = base_uncertainty_km + distance_km * 0.35

    return target_lat, target_lon, uncertainty_km


def compute_drift_trajectory_points(
    spill_lat: float,
    spill_lon: float,
    net_drift_speed_ms: float,
    net_drift_bearing_deg: float,
    start_hours: float,
    end_hours: float,
    steps: int = 20,
) -> List[Tuple[float, float]]:
    """
    Generate ordered, interpolated geographic coordinates [lat, lon] along the drift trajectory.
    """
    if steps < 1:
        steps = 1
    pts: List[Tuple[float, float]] = []
    for s in range(steps + 1):
        frac = s / float(steps)
        h = start_hours + frac * (end_hours - start_hours)
        lat, lon, _ = compute_temporal_position(
            spill_lat, spill_lon, net_drift_speed_ms, net_drift_bearing_deg, h
        )
        pts.append((lat, lon))
    return pts


def compute_all_time_slice_geometries(
    spill_lat: float,
    spill_lon: float,
    net_drift_speed_ms: float,
    net_drift_bearing_deg: float,
    time_slices: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes exact deterministic geometry for all standard milestones:
    T-3H, T0, T+6H, T+12H, T+24H.
    """
    slices = time_slices or STANDARD_TIME_SLICES
    out: Dict[str, Dict[str, Any]] = {}
    for s_name in slices:
        h = parse_time_slice_hours(s_name)
        lat, lon, unc = compute_temporal_position(
            spill_lat, spill_lon, net_drift_speed_ms, net_drift_bearing_deg, h
        )
        out[s_name] = {
            "time_slice": s_name,
            "hours": h,
            "lat": lat,
            "lon": lon,
            "uncertainty_km": unc,
        }
    return out


def sort_and_validate_track(track_points: List[Any]) -> List[Dict[str, Any]]:
    """
    Validates, filters, and temporally orders vessel AIS track points.
    - Rejects coordinates outside valid geographic range
    - Deduplicates adjacent identical coordinates
    - Orders points strictly ascending by timestamp
    """
    if not track_points:
        return []

    valid_pts: List[Dict[str, Any]] = []
    for p in track_points:
        if isinstance(p, dict):
            lat = p.get("lat")
            lon = p.get("lon")
            ts = p.get("timestamp")
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            lat = p[0]
            lon = p[1]
            ts = p[2] if len(p) >= 3 else None
        else:
            continue

        if not validate_lat_lon(lat, lon):
            continue

        pt_dict = {
            "lat": float(lat),
            "lon": float(lon),
            "timestamp": ts,
        }
        if isinstance(p, dict):
            for k in ("speed", "course", "heading", "name", "mmsi"):
                if k in p:
                    pt_dict[k] = p[k]

        valid_pts.append(pt_dict)

    if not valid_pts:
        return []

    def _sort_key(pt):
        t = pt.get("timestamp")
        if t is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        if isinstance(t, (int, float)):
            return datetime.fromtimestamp(t, tz=timezone.utc)
        if isinstance(t, str):
            try:
                # Handle ISO timestamps with Z or offsets
                clean_ts = t.replace("Z", "+00:00")
                return datetime.fromisoformat(clean_ts)
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)
        if isinstance(t, datetime):
            return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        return datetime.min.replace(tzinfo=timezone.utc)

    valid_pts.sort(key=_sort_key)

    # Deduplicate consecutive identical points
    deduped: List[Dict[str, Any]] = []
    for pt in valid_pts:
        if deduped:
            prev = deduped[-1]
            if abs(prev["lat"] - pt["lat"]) < 1e-6 and abs(prev["lon"] - pt["lon"]) < 1e-6:
                continue
        deduped.append(pt)

    return deduped


def find_vessel_position_at_epoch(
    sorted_track: List[Dict[str, Any]],
    target_dt: datetime,
) -> Optional[Dict[str, Any]]:
    """
    Finds the closest vessel track point to the given epoch from a sorted track.
    Returns None if track is empty.
    """
    if not sorted_track:
        return None

    if not target_dt.tzinfo:
        target_dt = target_dt.replace(tzinfo=timezone.utc)

    best_pt = sorted_track[0]
    min_delta = float("inf")

    for pt in sorted_track:
        ts = pt.get("timestamp")
        if not ts:
            continue
        try:
            if isinstance(ts, (int, float)):
                pt_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            elif isinstance(ts, str):
                pt_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, datetime):
                pt_dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            else:
                continue

            dt_sec = abs((pt_dt - target_dt).total_seconds())
            if dt_sec < min_delta:
                min_delta = dt_sec
                best_pt = pt
        except Exception:
            continue

    return best_pt


def calculate_map_bounds(
    geometries: List[Any],
    min_padding_deg: float = 0.035,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculates sensible bounding box [[min_lat, min_lon], [max_lat, max_lon]]
    encompassing all visible scene geometries with appropriate padding.
    Avoids extreme zoom-in and excessive zoom-out.
    """
    lats: List[float] = []
    lons: List[float] = []

    def _extract_coords(item):
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            if len(item) == 2 and isinstance(item[0], (int, float)) and isinstance(item[1], (int, float)):
                if validate_lat_lon(item[0], item[1]):
                    lats.append(float(item[0]))
                    lons.append(float(item[1]))
            else:
                for sub in item:
                    _extract_coords(sub)
        elif isinstance(item, dict):
            if "lat" in item and "lon" in item:
                if validate_lat_lon(item["lat"], item["lon"]):
                    lats.append(float(item["lat"]))
                    lons.append(float(item["lon"]))
            if "coordinates" in item:
                _extract_coords(item["coordinates"])

    for g in geometries:
        _extract_coords(g)

    if not lats or not lons:
        # Fallback to Chennai maritime region
        return ((12.90, 80.15), (13.35, 80.45))

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    # Ensure a minimum visible window
    pad_lat = max(lat_span * 0.18, min_padding_deg)
    pad_lon = max(lon_span * 0.18, min_padding_deg)

    bound_min_lat = max(-89.9, min_lat - pad_lat)
    bound_max_lat = min(89.9, max_lat + pad_lat)
    bound_min_lon = max(-179.9, min_lon - pad_lon)
    bound_max_lon = min(179.9, max_lon + pad_lon)

    return ((bound_min_lat, bound_min_lon), (bound_max_lat, bound_max_lon))
