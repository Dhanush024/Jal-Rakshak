"""
Jal-Rakshak — Ocean & Weather Provider Interface
==================================================
Abstract interfaces for oceanographic and meteorological data.
Supports REAL and DEMO modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum


class DataMode(Enum):
    REAL = "REAL"
    DEMO = "DEMO"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class OceanCurrentPoint:
    """Ocean current measurement at a point."""
    lat: float
    lon: float
    timestamp: datetime
    speed_ms: float          # m/s
    direction_deg: float     # bearing of current flow (0=N, 90=E)
    depth_m: float = 0.0    # surface by default

    def to_dict(self) -> dict:
        return {
            "lat": round(self.lat, 4),
            "lon": round(self.lon, 4),
            "timestamp": self.timestamp.isoformat(),
            "speed_ms": round(self.speed_ms, 3),
            "direction_deg": round(self.direction_deg, 1),
            "depth_m": self.depth_m,
        }


@dataclass
class WindPoint:
    """Wind measurement at a point."""
    lat: float
    lon: float
    timestamp: datetime
    speed_ms: float
    direction_deg: float     # direction wind comes FROM

    def to_dict(self) -> dict:
        return {
            "lat": round(self.lat, 4),
            "lon": round(self.lon, 4),
            "timestamp": self.timestamp.isoformat(),
            "speed_ms": round(self.speed_ms, 2),
            "direction_deg": round(self.direction_deg, 1),
        }


@dataclass
class OceanDataResult:
    """Result from an ocean/weather data query."""
    currents: List[OceanCurrentPoint]
    wind: List[WindPoint]
    mode: DataMode
    source: str               # e.g., "INCOIS", "Copernicus", "Demo"
    query_time: Optional[datetime] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "source": self.source,
            "num_current_points": len(self.currents),
            "num_wind_points": len(self.wind),
            "warnings": self.warnings,
        }


class OceanProvider(ABC):
    """Abstract interface for ocean current data."""

    @property
    @abstractmethod
    def mode(self) -> DataMode:
        ...

    @abstractmethod
    def get_currents(self, lat: float, lon: float,
                     time: datetime, radius_km: float = 50.0
                     ) -> List[OceanCurrentPoint]:
        ...


class WeatherProvider(ABC):
    """Abstract interface for weather/wind data."""

    @property
    @abstractmethod
    def mode(self) -> DataMode:
        ...

    @abstractmethod
    def get_wind(self, lat: float, lon: float,
                 time: datetime) -> List[WindPoint]:
        ...
