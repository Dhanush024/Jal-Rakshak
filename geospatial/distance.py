"""
Jal-Rakshak — Geospatial Distance & Bearing Utilities
=======================================================
Single source of truth for all geographic calculations.
Extracted from ais_engine.py to avoid duplication.
"""

import math
from typing import Tuple


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


def normalize_bearing(bearing: float) -> float:
    """Normalize a bearing to [0, 360)."""
    return bearing % 360


def bearing_difference(bearing1: float, bearing2: float) -> float:
    """Absolute angular difference between two bearings (0-180)."""
    diff = abs(bearing1 - bearing2) % 360
    return diff if diff <= 180 else 360 - diff


def polygon_area_km2(coordinates: list) -> float:
    """
    Calculate planar geodesic approximation of polygon area in square kilometers.
    Coordinates can be GeoJSON [[lon, lat], ...] or [(lat, lon), ...].
    """
    if not coordinates or len(coordinates) < 3:
        return 0.0
    pts = []
    for c in coordinates:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            pts.append((float(c[1]), float(c[0])))
    if len(pts) < 3:
        return 0.0
    R = 6371.0
    mean_lat = sum(math.radians(p[0]) for p in pts) / len(pts)
    cos_lat = math.cos(mean_lat)
    x = [R * math.radians(p[1]) * cos_lat for p in pts]
    y = [R * math.radians(p[0]) for p in pts]
    area = 0.0
    n = len(x)
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    return round(abs(area) / 2.0, 3)

