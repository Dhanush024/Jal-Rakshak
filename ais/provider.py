"""
Jal-Rakshak — AIS Provider Interface & Concrete Implementations
=================================================================
Abstract interface and concrete providers for AIS vessel telemetry:
- DemoAISProvider: Deterministic, offline synthetic fleet for demonstration
- FileHistoricalAISProvider: Real historical AIS data ingester (CSV / JSON / GeoJSON)
- get_ais_provider: Factory providing clean mode switching (DEMO / REAL / AUTO)

Adheres strictly to SIH 2026 data integrity:
- Exposes data mode (REAL / DEMO / UNAVAILABLE)
- Filters and validates real inputs (malformed rows, coordinates, duplicates)
- Never silently replaces real requests with synthetic data
"""

import os
import csv
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger("jal_rakshak.ais")


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
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "heading": round(self.heading, 1),
            "speed_knots": round(self.speed_knots, 1),
            "timestamp": self.timestamp.isoformat(),
            "course_over_ground": round(self.course_over_ground, 1) if self.course_over_ground is not None else None,
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
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.total_vessels = len(self.tracks)


class AISProvider(ABC):
    """Abstract interface for AIS data providers."""

    @property
    @abstractmethod
    def mode(self) -> DataMode:
        """Return the data mode (REAL, DEMO, or UNAVAILABLE)."""
        ...

    @abstractmethod
    def get_live_positions(self, bbox: Tuple[float, float, float, float],
                           timeout_sec: float = 10.0) -> AISQueryResult:
        """Get current vessel positions within a bounding box."""
        ...

    @abstractmethod
    def get_historical_tracks(self, bbox: Optional[Tuple[float, float, float, float]] = None,
                               time_start: Optional[datetime] = None,
                               time_end: Optional[datetime] = None) -> AISQueryResult:
        """Get historical vessel tracks within a bounding box and time window."""
        ...


class DemoAISProvider(AISProvider):
    """
    Guaranteed offline synthetic AIS provider.
    Returns deterministic vessel trajectories matching the Chennai scenario.
    """

    @property
    def mode(self) -> DataMode:
        return DataMode.DEMO

    def get_live_positions(self, bbox: Tuple[float, float, float, float],
                           timeout_sec: float = 10.0) -> AISQueryResult:
        historical = self.get_historical_tracks(bbox)
        # Latest ping per vessel
        latest_tracks = {}
        for mmsi, records in historical.tracks.items():
            if records:
                latest_tracks[mmsi] = [records[-1]]
        return AISQueryResult(
            tracks=latest_tracks,
            mode=DataMode.DEMO,
            query_bbox=bbox,
            warnings=["Demonstration provider: synthetic fleet positions."],
        )

    def get_historical_tracks(self, bbox: Optional[Tuple[float, float, float, float]] = None,
                               time_start: Optional[datetime] = None,
                               time_end: Optional[datetime] = None) -> AISQueryResult:
        from demo.scenario import generate_demo_ais_tracks
        tracks = generate_demo_ais_tracks()
        return AISQueryResult(
            tracks=tracks,
            mode=DataMode.DEMO,
            query_bbox=bbox,
            query_time_start=time_start,
            query_time_end=time_end,
            warnings=["Demonstration provider: synthetic historical trajectories."],
        )


class FileHistoricalAISProvider(AISProvider):
    """
    Ingests real historical AIS data from CSV, JSON, or GeoJSON files.
    Standardized to handle US Coast Guard (MarineCadastre), Danish Maritime Authority,
    and Indian coastal AIS archival formats.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    @property
    def mode(self) -> DataMode:
        return DataMode.REAL

    def get_live_positions(self, bbox: Tuple[float, float, float, float],
                           timeout_sec: float = 10.0) -> AISQueryResult:
        historical = self.get_historical_tracks(bbox)
        latest_tracks = {}
        for mmsi, records in historical.tracks.items():
            if records:
                latest_tracks[mmsi] = [records[-1]]
        return AISQueryResult(
            tracks=latest_tracks,
            mode=DataMode.REAL,
            query_bbox=bbox,
            warnings=["Derived latest known positions from historical file archive."],
        )

    def get_historical_tracks(self, bbox: Optional[Tuple[float, float, float, float]] = None,
                               time_start: Optional[datetime] = None,
                               time_end: Optional[datetime] = None) -> AISQueryResult:
        if not os.path.exists(self.file_path):
            return AISQueryResult(
                tracks={},
                mode=DataMode.UNAVAILABLE,
                warnings=[f"AIS file not found: {self.file_path}"],
            )

        warnings = []
        tracks_by_mmsi: Dict[str, List[AISRecord]] = {}

        try:
            if self.file_path.endswith(".json") or self.file_path.endswith(".geojson"):
                self._parse_json(tracks_by_mmsi, warnings)
            else:
                self._parse_csv(tracks_by_mmsi, warnings)

            # Sort records chronologically and filter by bbox/time if requested
            filtered_tracks: Dict[str, List[AISRecord]] = {}
            for mmsi, records in tracks_by_mmsi.items():
                # Deduplicate and sort
                sorted_records = sorted(records, key=lambda r: r.timestamp)
                deduped = []
                last_time = None
                for r in sorted_records:
                    if r.timestamp != last_time:
                        # Spatial filter
                        if bbox:
                            min_lat, min_lon, max_lat, max_lon = bbox
                            if not (min_lat <= r.lat <= max_lat and min_lon <= r.lon <= max_lon):
                                continue
                        # Temporal filter
                        if time_start and r.timestamp < time_start:
                            continue
                        if time_end and r.timestamp > time_end:
                            continue
                        deduped.append(r)
                        last_time = r.timestamp

                if deduped:
                    filtered_tracks[mmsi] = deduped

            return AISQueryResult(
                tracks=filtered_tracks,
                mode=DataMode.REAL,
                query_bbox=bbox,
                query_time_start=time_start,
                query_time_end=time_end,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Failed parsing AIS file {self.file_path}: {e}")
            return AISQueryResult(
                tracks={},
                mode=DataMode.UNAVAILABLE,
                warnings=[f"Parsing error in {self.file_path}: {str(e)}"],
            )

    def _parse_csv(self, tracks: Dict[str, List[AISRecord]], warnings: List[str]):
        """Parse standard AIS CSV file."""
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                warnings.append("Empty CSV file or missing header.")
                return

            # Map field names
            fmap = {k.strip().lower(): k for k in reader.fieldnames}
            mmsi_k = self._find_col(fmap, ["mmsi", "vessel_mmsi", "user_id"])
            lat_k = self._find_col(fmap, ["lat", "latitude", "y"])
            lon_k = self._find_col(fmap, ["lon", "longitude", "long", "x"])
            time_k = self._find_col(fmap, ["basedatetime", "timestamp", "datetime", "time", "date_time_utc"])
            sog_k = self._find_col(fmap, ["sog", "speed", "speed_knots"])
            cog_k = self._find_col(fmap, ["cog", "course", "course_over_ground"])
            hdg_k = self._find_col(fmap, ["heading", "true_heading"])
            name_k = self._find_col(fmap, ["vesselname", "name", "vessel_name"])
            type_k = self._find_col(fmap, ["vesseltype", "type", "ship_type"])
            status_k = self._find_col(fmap, ["status", "navigationalstatus", "navigation_status"])

            if not (mmsi_k and lat_k and lon_k and time_k):
                warnings.append("Missing required AIS columns: MMSI, Latitude, Longitude, or Timestamp.")
                return

            row_num = 0
            invalid_rows = 0
            for row in reader:
                row_num += 1
                try:
                    mmsi = str(row[mmsi_k]).strip()
                    lat = float(row[lat_k])
                    lon = float(row[lon_k])
                    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                        invalid_rows += 1
                        continue

                    # Parse timestamp
                    ts_str = str(row[time_k]).strip()
                    ts = self._parse_iso_time(ts_str)

                    sog = float(row[sog_k]) if sog_k and row.get(sog_k) else 0.0
                    cog = float(row[cog_k]) if cog_k and row.get(cog_k) else None
                    hdg = float(row[hdg_k]) if hdg_k and row.get(hdg_k) else (cog if cog is not None else 0.0)
                    name = str(row[name_k]).strip() if name_k and row.get(name_k) else f"Vessel_{mmsi}"
                    vtype = str(row[type_k]).strip() if type_k and row.get(type_k) else None
                    status = str(row[status_k]).strip() if status_k and row.get(status_k) else None

                    rec = AISRecord(
                        mmsi=mmsi, name=name, lat=lat, lon=lon,
                        heading=hdg, speed_knots=sog, timestamp=ts,
                        course_over_ground=cog, vessel_type=vtype, navigation_status=status,
                    )
                    tracks.setdefault(mmsi, []).append(rec)
                except Exception:
                    invalid_rows += 1

            if invalid_rows > 0:
                warnings.append(f"Skipped {invalid_rows} malformed or invalid coordinate rows.")

    def _parse_json(self, tracks: Dict[str, List[AISRecord]], warnings: List[str]):
        """Parse AIS JSON / GeoJSON array."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records_list = []
        if isinstance(data, list):
            records_list = data
        elif isinstance(data, dict) and "features" in data:  # GeoJSON
            for feat in data["features"]:
                geom = feat.get("geometry", {})
                props = feat.get("properties", {})
                if geom.get("type") == "Point":
                    coords = geom.get("coordinates", [0, 0])
                    p = dict(props)
                    p["lon"] = coords[0]
                    p["lat"] = coords[1]
                    records_list.append(p)
        elif isinstance(data, dict) and "tracks" in data:
            for mmsi, pings in data["tracks"].items():
                for p in pings:
                    p["mmsi"] = mmsi
                    records_list.append(p)

        for item in records_list:
            try:
                mmsi = str(item.get("mmsi", item.get("MMSI", ""))).strip()
                lat = float(item.get("lat", item.get("LAT", item.get("latitude", 0))))
                lon = float(item.get("lon", item.get("LON", item.get("longitude", 0))))
                ts_str = str(item.get("timestamp", item.get("time", item.get("BaseDateTime", ""))))
                ts = self._parse_iso_time(ts_str)
                sog = float(item.get("speed_knots", item.get("sog", item.get("speed", 0))))
                cog = float(item.get("course_over_ground", item.get("cog", 0))) if "cog" in item or "course_over_ground" in item else None
                hdg = float(item.get("heading", item.get("Heading", 0)))
                name = str(item.get("name", item.get("vessel_name", f"Vessel_{mmsi}")))
                vtype = item.get("vessel_type")
                status = item.get("navigation_status")

                rec = AISRecord(
                    mmsi=mmsi, name=name, lat=lat, lon=lon,
                    heading=hdg, speed_knots=sog, timestamp=ts,
                    course_over_ground=cog, vessel_type=vtype, navigation_status=status,
                )
                tracks.setdefault(mmsi, []).append(rec)
            except Exception:
                continue

    @staticmethod
    def _find_col(fmap: Dict[str, str], candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in fmap:
                return fmap[c]
        return None

    @staticmethod
    def _parse_iso_time(ts_str: str) -> datetime:
        # Replace common format variations
        clean = ts_str.replace("Z", "+00:00").replace("/", "-")
        try:
            dt = datetime.fromisoformat(clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y %H:%M:%S"):
                try:
                    return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            raise ValueError(f"Unrecognized timestamp format: {ts_str}")


def get_ais_provider(mode: str = "demo", file_path: Optional[str] = None) -> AISProvider:
    """
    Factory function returning the appropriate AIS provider.
    Supports mode: "demo", "real", "auto".
    """
    mode_str = (mode or "demo").lower()

    if mode_str == "demo":
        return DemoAISProvider()

    if mode_str in ("real", "historical"):
        if file_path and os.path.exists(file_path):
            return FileHistoricalAISProvider(file_path)
        # Check standard default data directory
        default_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical_ais.csv")
        if os.path.exists(default_file):
            return FileHistoricalAISProvider(default_file)
        # Real file not provided
        logger.warning("Real AIS mode requested but no historical file provided. Returning unavailable.")
        # Return a provider that safely yields DataMode.UNAVAILABLE
        class UnavailableAISProvider(AISProvider):
            @property
            def mode(self) -> DataMode:
                return DataMode.UNAVAILABLE
            def get_live_positions(self, bbox, timeout_sec=10.0):
                return AISQueryResult(tracks={}, mode=DataMode.UNAVAILABLE, warnings=["Real AIS provider unconfigured."])
            def get_historical_tracks(self, bbox=None, time_start=None, time_end=None):
                return AISQueryResult(tracks={}, mode=DataMode.UNAVAILABLE, warnings=["No real historical AIS archive supplied."])
        return UnavailableAISProvider()

    # AUTO: prefer real if file exists, else fallback to demo
    if file_path and os.path.exists(file_path):
        return FileHistoricalAISProvider(file_path)
    return DemoAISProvider()
