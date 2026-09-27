"""
Jal-Rakshak — AIS Provider Interface
======================================
Abstract interface for AIS data providers.
Supports REAL and DEMO modes with clear data source labeling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum


class DataMode(Enum):
    """Data source mode indicator."""
    REAL = "REAL"
    DEMO = "DEMO"
    UNAVAILABLE = "UNAVAILABLE"


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
    course_over_ground: Optional[float] = None
    vessel_type: Optional[str] = None
    navigation_status: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "mmsi": self.mmsi,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "heading": self.heading,
            "speed_knots": self.speed_knots,
            "timestamp": self.timestamp.isoformat(),
            "course_over_ground": self.course_over_ground,
            "vessel_type": self.vessel_type,
            "navigation_status": self.navigation_status,
        }


@dataclass
class AISQueryResult:
    """Result from an AIS data query."""
    tracks: Dict[str, List[AISRecord]]  # {mmsi: [records]}
    mode: DataMode
    query_bbox: Optional[Tuple[float, float, float, float]] = None  # (min_lat, min_lon, max_lat, max_lon)
    query_time_start: Optional[datetime] = None
    query_time_end: Optional[datetime] = None
    total_vessels: int = 0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        self.total_vessels = len(self.tracks)


class AISProvider(ABC):
    """Abstract interface for AIS data providers."""

    @property
    @abstractmethod
    def mode(self) -> DataMode:
        """Return the data mode (REAL or DEMO)."""
        ...

    @abstractmethod
    def get_live_positions(self, bbox: Tuple[float, float, float, float],
                           timeout_sec: float = 10.0) -> AISQueryResult:
        """
        Get current vessel positions within a bounding box.

        Args:
            bbox: (min_lat, min_lon, max_lat, max_lon)
            timeout_sec: Maximum time to wait for data

        Returns:
            AISQueryResult with current positions
        """
        ...

    @abstractmethod
    def get_historical_tracks(self, bbox: Tuple[float, float, float, float],
                               time_start: datetime, time_end: datetime
                               ) -> AISQueryResult:
        """
        Get historical vessel tracks within a bounding box and time window.

        Args:
            bbox: (min_lat, min_lon, max_lat, max_lon)
            time_start: Start of time window
            time_end: End of time window

        Returns:
            AISQueryResult with historical tracks
        """
        ...
