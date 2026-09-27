"""
Jal-Rakshak — Ocean & Weather Provider Interface & Implementations
===================================================================
Abstract interfaces and concrete providers for ocean currents and wind:
- DemoOceanProvider: Offline scenario currents and winds
- ConstantOceanProvider: Configurable uniform vector field
- get_ocean_provider: Provider factory for mode switching (DEMO / REAL / AUTO)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from enum import Enum

from config.settings import (
    DEFAULT_CURRENT_SPEED_MS, DEFAULT_CURRENT_BEARING_DEG,
    DEFAULT_WIND_SPEED_MS, DEFAULT_WIND_BEARING_DEG,
)


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
                     time: Optional[datetime] = None, radius_km: float = 50.0
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
                 time: Optional[datetime] = None) -> List[WindPoint]:
        ...


class DemoOceanProvider(OceanProvider, WeatherProvider):
    """Scenario ocean and wind provider."""

    @property
    def mode(self) -> DataMode:
        return DataMode.DEMO

    def get_currents(self, lat: float, lon: float,
                     time: Optional[datetime] = None, radius_km: float = 50.0
                     ) -> List[OceanCurrentPoint]:
        t = time or datetime.now(timezone.utc)
        return [OceanCurrentPoint(
            lat=lat, lon=lon, timestamp=t,
            speed_ms=DEFAULT_CURRENT_SPEED_MS,
            direction_deg=DEFAULT_CURRENT_BEARING_DEG,
        )]

    def get_wind(self, lat: float, lon: float,
                 time: Optional[datetime] = None) -> List[WindPoint]:
        t = time or datetime.now(timezone.utc)
        return [WindPoint(
            lat=lat, lon=lon, timestamp=t,
            speed_ms=DEFAULT_WIND_SPEED_MS,
            direction_deg=DEFAULT_WIND_BEARING_DEG,
        )]


class ConstantOceanProvider(OceanProvider, WeatherProvider):
    """Configurable constant-vector oceanographic & meteorological provider."""

    def __init__(self, current_speed_ms: float = DEFAULT_CURRENT_SPEED_MS,
                 current_bearing_deg: float = DEFAULT_CURRENT_BEARING_DEG,
                 wind_speed_ms: float = DEFAULT_WIND_SPEED_MS,
                 wind_bearing_deg: float = DEFAULT_WIND_BEARING_DEG,
                 is_real_source: bool = False):
        self.current_speed_ms = current_speed_ms
        self.current_bearing_deg = current_bearing_deg
        self.wind_speed_ms = wind_speed_ms
        self.wind_bearing_deg = wind_bearing_deg
        self.is_real_source = is_real_source

    @property
    def mode(self) -> DataMode:
        return DataMode.REAL if self.is_real_source else DataMode.DEMO

    def get_currents(self, lat: float, lon: float,
                     time: Optional[datetime] = None, radius_km: float = 50.0
                     ) -> List[OceanCurrentPoint]:
        t = time or datetime.now(timezone.utc)
        return [OceanCurrentPoint(
            lat=lat, lon=lon, timestamp=t,
            speed_ms=self.current_speed_ms,
            direction_deg=self.current_bearing_deg,
        )]

    def get_wind(self, lat: float, lon: float,
                 time: Optional[datetime] = None) -> List[WindPoint]:
        t = time or datetime.now(timezone.utc)
        return [WindPoint(
            lat=lat, lon=lon, timestamp=t,
            speed_ms=self.wind_speed_ms,
            direction_deg=self.wind_bearing_deg,
        )]


def get_ocean_provider(mode: str = "demo") -> OceanProvider:
    """Return configured OceanProvider instance."""
    mode_str = (mode or "demo").lower()
    if mode_str == "demo":
        return DemoOceanProvider()
    return ConstantOceanProvider(is_real_source=(mode_str == "real"))
