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
