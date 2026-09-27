"""
Jal-Rakshak — AIS Traffic Filtering Pipeline
==============================================
Progressive filtering of AIS traffic to isolate candidate vessels.
Shows filtering stages: spatial → temporal → trajectory → feasibility.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from ais.provider import AISRecord
from geospatial.distance import haversine_km


@dataclass
class FilterStage:
    """Result of one filtering stage."""
    stage_name: str
    vessels_before: int
    vessels_after: int
    vessels_removed: List[str]  # MMSIs removed
    criteria: str

    def to_dict(self) -> dict:
        return {
            "stage_name": self.stage_name,
            "vessels_before": self.vessels_before,
            "vessels_after": self.vessels_after,
            "vessels_removed_count": len(self.vessels_removed),
            "criteria": self.criteria,
        }


@dataclass
class FilteringResult:
    """Complete result of the filtering pipeline."""
    stages: List[FilterStage]
    candidate_tracks: Dict[str, List[AISRecord]]
    total_initial: int
    total_candidates: int

    def to_dict(self) -> dict:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "total_initial": self.total_initial,
            "total_candidates": self.total_candidates,
            "candidate_mmsis": list(self.candidate_tracks.keys()),
        }


def filter_by_spatial_distance(tracks: Dict[str, List[AISRecord]],
                                center_lat: float, center_lon: float,
                                max_distance_km: float = 50.0
                                ) -> Tuple[Dict[str, List[AISRecord]], FilterStage]:
    """
    Filter vessels by minimum distance to a reference point.
    Keep only vessels that come within max_distance_km at any point.
    """
    kept = {}
    removed = []

    for mmsi, track in tracks.items():
        min_dist = min(
            haversine_km(r.lat, r.lon, center_lat, center_lon)
            for r in track
        )
        if min_dist <= max_distance_km:
            kept[mmsi] = track
        else:
            removed.append(mmsi)

    stage = FilterStage(
        stage_name="Spatial Distance",
        vessels_before=len(tracks),
        vessels_after=len(kept),
        vessels_removed=removed,
        criteria=f"Within {max_distance_km} km of reference point",
    )
    return kept, stage


def filter_by_temporal_window(tracks: Dict[str, List[AISRecord]],
                               time_center: datetime,
                               window_hours: float = 6.0
                               ) -> Tuple[Dict[str, List[AISRecord]], FilterStage]:
    """
    Filter vessels by temporal relevance.
    Keep only vessels with AIS data within the time window.
    """
    kept = {}
    removed = []
    half_window = timedelta(hours=window_hours / 2)
    t_start = time_center - half_window
    t_end = time_center + half_window

    for mmsi, track in tracks.items():
        has_data_in_window = any(
            t_start <= r.timestamp <= t_end for r in track
        )
        if has_data_in_window:
            kept[mmsi] = track
        else:
            removed.append(mmsi)

    stage = FilterStage(
        stage_name="Temporal Window",
        vessels_before=len(tracks),
        vessels_after=len(kept),
        vessels_removed=removed,
        criteria=f"AIS data within ±{window_hours/2:.1f}h of estimated event time",
    )
    return kept, stage


def filter_by_trajectory_intersection(tracks: Dict[str, List[AISRecord]],
                                       region_lat: float, region_lon: float,
                                       region_radius_km: float = 15.0
                                       ) -> Tuple[Dict[str, List[AISRecord]], FilterStage]:
    """
    Filter vessels whose trajectory passes through the region of interest.
    """
    kept = {}
    removed = []

    for mmsi, track in tracks.items():
        passes_through = any(
            haversine_km(r.lat, r.lon, region_lat, region_lon) <= region_radius_km
            for r in track
        )
        if passes_through:
            kept[mmsi] = track
        else:
            removed.append(mmsi)

    stage = FilterStage(
        stage_name="Trajectory Intersection",
        vessels_before=len(tracks),
        vessels_after=len(kept),
        vessels_removed=removed,
        criteria=f"Trajectory passes within {region_radius_km} km of source region",
    )
    return kept, stage


def filter_by_speed_feasibility(tracks: Dict[str, List[AISRecord]],
                                 min_speed_knots: float = 0.5,
                                 max_speed_knots: float = 25.0
                                 ) -> Tuple[Dict[str, List[AISRecord]], FilterStage]:
    """
    Filter out vessels with implausible speeds (anchored or too fast).
    """
    kept = {}
    removed = []

    for mmsi, track in tracks.items():
        avg_speed = sum(r.speed_knots for r in track) / len(track) if track else 0
        if min_speed_knots <= avg_speed <= max_speed_knots:
            kept[mmsi] = track
        else:
            removed.append(mmsi)

    stage = FilterStage(
        stage_name="Speed Feasibility",
        vessels_before=len(tracks),
        vessels_after=len(kept),
        vessels_removed=removed,
        criteria=f"Average speed {min_speed_knots}-{max_speed_knots} knots",
    )
    return kept, stage


def run_filtering_pipeline(tracks: Dict[str, List[AISRecord]],
                            spill_lat: float, spill_lon: float,
                            source_lat: float, source_lon: float,
                            event_time: datetime,
                            spatial_radius_km: float = 50.0,
                            temporal_window_hours: float = 6.0,
                            trajectory_radius_km: float = 15.0
                            ) -> FilteringResult:
    """
    Run the complete AIS traffic filtering pipeline.

    Stages:
    1. Spatial distance from spill
    2. Temporal window around event
    3. Trajectory intersection with source region
    4. Speed feasibility

    Returns detailed filtering result with stage-by-stage breakdown.
    """
    initial_count = len(tracks)
    stages = []
    current_tracks = tracks

    # Stage 1: Spatial
    current_tracks, stage = filter_by_spatial_distance(
        current_tracks, spill_lat, spill_lon, spatial_radius_km
    )
    stages.append(stage)

    # Stage 2: Temporal
    current_tracks, stage = filter_by_temporal_window(
        current_tracks, event_time, temporal_window_hours
    )
    stages.append(stage)

    # Stage 3: Trajectory
    current_tracks, stage = filter_by_trajectory_intersection(
        current_tracks, source_lat, source_lon, trajectory_radius_km
    )
    stages.append(stage)

    # Stage 4: Speed feasibility
    current_tracks, stage = filter_by_speed_feasibility(current_tracks)
    stages.append(stage)

    return FilteringResult(
        stages=stages,
        candidate_tracks=current_tracks,
        total_initial=initial_count,
        total_candidates=len(current_tracks),
    )
