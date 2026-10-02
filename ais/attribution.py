"""
Jal-Rakshak — Vessel Feature Engineering & Attribution
========================================================
Calculate vessel features and produce explainable candidate scoring.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from ais.provider import AISRecord
from geospatial.distance import haversine_km, bearing_deg, bearing_difference
from config.settings import (
    SCORE_WEIGHT_TEMPORAL, SCORE_WEIGHT_SPATIAL,
    SCORE_WEIGHT_TRAJECTORY, SCORE_WEIGHT_HEADING,
    SCORE_WEIGHT_SPEED, SCORE_WEIGHT_DRIFT,
)


@dataclass
class VesselFeatures:
    """Computed features for a candidate vessel."""
    mmsi: str
    name: str
    min_distance_to_spill_km: float
    min_distance_to_source_km: float
    closest_approach_time: datetime
    closest_approach_distance_km: float
    time_diff_to_event_min: float
    avg_speed_knots: float
    heading_at_closest: float
    heading_consistency: float       # 0-1, how consistent heading is toward source
    trajectory_intersects_source: bool
    time_near_source_min: float      # minutes spent within source region
    time_near_spill_min: float
    ais_gap_detected: bool
    ais_gap_duration_min: float
    max_reporting_interval_min: float
    speed_changes: int               # number of significant speed changes
    course_changes: int              # number of significant course changes

    def to_dict(self) -> dict:
        return {
            "mmsi": self.mmsi,
            "name": self.name,
            "min_distance_to_spill_km": round(self.min_distance_to_spill_km, 2),
            "min_distance_to_source_km": round(self.min_distance_to_source_km, 2),
            "closest_approach_time": self.closest_approach_time.isoformat(),
            "closest_approach_distance_km": round(self.closest_approach_distance_km, 2),
            "time_diff_to_event_min": round(self.time_diff_to_event_min, 1),
            "avg_speed_knots": round(self.avg_speed_knots, 1),
            "heading_at_closest": round(self.heading_at_closest, 1),
            "heading_consistency": round(self.heading_consistency, 3),
            "trajectory_intersects_source": self.trajectory_intersects_source,
            "time_near_source_min": round(self.time_near_source_min, 1),
            "time_near_spill_min": round(self.time_near_spill_min, 1),
            "ais_gap_detected": self.ais_gap_detected,
            "ais_gap_duration_min": round(self.ais_gap_duration_min, 1),
            "ais_gap_note": "AIS data gap — does NOT constitute evidence of wrongdoing" if self.ais_gap_detected else "",
            "speed_changes": self.speed_changes,
            "course_changes": self.course_changes,
        }


@dataclass
class CandidateScore:
    """Explainable attribution score for a candidate vessel."""
    mmsi: str
    name: str
    total_score: float          # 0-100
    spatial_score: float        # 0-100
    temporal_score: float       # 0-100
    trajectory_score: float     # 0-100
    heading_score: float        # 0-100
    speed_score: float          # 0-100
    drift_score: float          # 0-100
    behavioral_anomaly_score: float  # 0-100
    evidence_summary: List[str]
    uncertainty_notes: List[str]
    features: VesselFeatures

    # Boolean match indicators
    trajectory_match: bool = False
    time_match: bool = False
    spatial_match: bool = False
    drift_consistency: bool = False

    def to_dict(self) -> dict:
        feats_dict = self.features.to_dict() if hasattr(self, "features") and self.features else {}
        return {
            "mmsi": self.mmsi,
            "name": self.name,
            "score": round(self.total_score, 1),
            "min_distance_km": round(self.features.closest_approach_distance_km, 2) if hasattr(self, "features") and self.features else None,
            "min_distance_to_source_km": round(self.features.min_distance_to_source_km, 2) if hasattr(self, "features") and self.features else None,
            "features": feats_dict,
            "breakdown": {
                "spatial": round(self.spatial_score, 1),
                "temporal": round(self.temporal_score, 1),
                "trajectory": round(self.trajectory_score, 1),
                "heading": round(self.heading_score, 1),
                "speed": round(self.speed_score, 1),
                "drift_consistency": round(self.drift_score, 1),
                "behavioral": round(self.behavioral_anomaly_score, 1),
            },
            "trajectory_match": self.trajectory_match,
            "time_match": self.time_match,
            "spatial_match": self.spatial_match,
            "drift_consistency": self.drift_consistency,
            "evidence_summary": self.evidence_summary,
            "uncertainty_notes": self.uncertainty_notes,
            "disclaimer": (
                "This is an ASSOCIATION SCORE for decision support only. "
                "It does NOT constitute proof of liability or legal attribution."
            ),
        }



def compute_vessel_features(track: List[AISRecord],
                             spill_lat: float, spill_lon: float,
                             source_lat: float, source_lon: float,
                             event_time: datetime,
                             source_radius_km: float = 10.0,
                             spill_radius_km: float = 15.0,
                             gap_threshold_min: float = 15.0
                             ) -> VesselFeatures:
    """Compute all features for a single vessel track."""
    if not track:
        raise ValueError("Empty track")

    name = track[0].name
    mmsi = track[0].mmsi

    # Distances
    spill_dists = [(haversine_km(r.lat, r.lon, spill_lat, spill_lon), r) for r in track]
    source_dists = [(haversine_km(r.lat, r.lon, source_lat, source_lon), r) for r in track]

    min_spill_dist, _ = min(spill_dists, key=lambda x: x[0])
    min_source_dist, closest_rec = min(source_dists, key=lambda x: x[0])

    # Closest approach
    closest_approach_time = closest_rec.timestamp
    closest_approach_distance = min_source_dist

    # Time difference to event
    time_diff_sec = abs((closest_rec.timestamp - event_time).total_seconds())
    time_diff_min = time_diff_sec / 60.0

    # Speed
    avg_speed = sum(r.speed_knots for r in track) / len(track)

    # Heading consistency toward source
    bearing_to_source = bearing_deg(closest_rec.lat, closest_rec.lon, source_lat, source_lon)
    hdg_diff = bearing_difference(closest_rec.heading, bearing_to_source)
    heading_consistency = max(0, 1.0 - hdg_diff / 180.0)

    # Trajectory intersection
    intersects = any(d <= source_radius_km for d, _ in source_dists)

    # Time near source / spill
    time_near_source = sum(1 for d, _ in source_dists if d <= source_radius_km)
    time_near_spill = sum(1 for d, _ in spill_dists if d <= spill_radius_km)

    # AIS gaps
    max_gap = 0
    total_gap = 0
    for i in range(1, len(track)):
        gap = (track[i].timestamp - track[i-1].timestamp).total_seconds() / 60.0
        if gap > max_gap:
            max_gap = gap
        if gap > gap_threshold_min:
            total_gap += gap

    ais_gap = max_gap > gap_threshold_min

    # Speed and course changes
    speed_changes = 0
    course_changes = 0
    for i in range(1, len(track)):
        if abs(track[i].speed_knots - track[i-1].speed_knots) > 2.0:
            speed_changes += 1
        if bearing_difference(track[i].heading, track[i-1].heading) > 15.0:
            course_changes += 1

    return VesselFeatures(
        mmsi=mmsi,
        name=name,
        min_distance_to_spill_km=min_spill_dist,
        min_distance_to_source_km=min_source_dist,
        closest_approach_time=closest_approach_time,
        closest_approach_distance_km=closest_approach_distance,
        time_diff_to_event_min=time_diff_min,
        avg_speed_knots=avg_speed,
        heading_at_closest=closest_rec.heading,
        heading_consistency=heading_consistency,
        trajectory_intersects_source=intersects,
        time_near_source_min=time_near_source,
        time_near_spill_min=time_near_spill,
        ais_gap_detected=ais_gap,
        ais_gap_duration_min=total_gap,
        max_reporting_interval_min=max_gap,
        speed_changes=speed_changes,
        course_changes=course_changes,
    )


def score_candidate(features: VesselFeatures,
                     source_radius_km: float = 10.0,
                     current_bearing: float = 118.0
                     ) -> CandidateScore:
    """
    Produce an explainable candidate score from vessel features.

    Uses configurable weights from settings. All scores 0-100.
    """
    evidence = []
    uncertainty = []

    # Spatial score
    d = features.min_distance_to_source_km
    if d <= 1.0:
        spatial = 100
        evidence.append(f"Passed within {d:.1f} km of probable source region")
    elif d <= 3.0:
        spatial = 100 - (d - 1.0) * 25
    elif d <= 10.0:
        spatial = 50 - (d - 3.0) * 7.14
    else:
        spatial = max(0, 10 - (d - 10.0))

    # Temporal score
    t = features.time_diff_to_event_min
    if t <= 15:
        temporal = 100
        evidence.append(f"Nearest approach within {t:.0f} min of estimated event time")
    elif t <= 30:
        temporal = 100 - (t - 15) * 3.33
    elif t <= 60:
        temporal = 50 - (t - 30) * 1.67
    else:
        temporal = max(0, 20 - (t - 60) * 0.33)
        uncertainty.append(f"Time difference ({t:.0f} min) exceeds typical window")

    # Trajectory score
    trajectory = 100 if features.trajectory_intersects_source else 20
    if features.trajectory_intersects_source:
        evidence.append("Trajectory intersects probable source region")

    # Heading score
    heading = features.heading_consistency * 100

    # Speed score
    s = features.avg_speed_knots
    if 3.0 <= s <= 15.0:
        speed = 80
    elif s > 15.0:
        speed = 60
    elif s > 0.5:
        speed = 40
    else:
        speed = 10

    # Drift consistency
    if d < 5:
        drift = 70
    else:
        drift = 30

    # Behavioral anomaly score
    behavioral = 50  # base
    if features.ais_gap_detected:
        behavioral += 20
        evidence.append(f"AIS data gap detected ({features.ais_gap_duration_min:.0f} min) — anomaly, not proof")
    if features.course_changes > 5:
        behavioral += 15
        evidence.append(f"Unusual course changes ({features.course_changes})")
    if features.speed_changes > 5:
        behavioral += 15
    behavioral = min(100, behavioral)

    # Weighted total
    total = (
        temporal * SCORE_WEIGHT_TEMPORAL +
        spatial * SCORE_WEIGHT_SPATIAL +
        trajectory * SCORE_WEIGHT_TRAJECTORY +
        heading * SCORE_WEIGHT_HEADING +
        speed * SCORE_WEIGHT_SPEED +
        drift * SCORE_WEIGHT_DRIFT
    )

    # Add uncertainty notes
    uncertainty.append("Association score reflects spatial/temporal/trajectory evidence only")
    uncertainty.append("This does NOT constitute proof of liability")

    return CandidateScore(
        mmsi=features.mmsi,
        name=features.name,
        total_score=total,
        spatial_score=spatial,
        temporal_score=temporal,
        trajectory_score=trajectory,
        heading_score=heading,
        speed_score=speed,
        drift_score=drift,
        behavioral_anomaly_score=behavioral,
        evidence_summary=evidence,
        uncertainty_notes=uncertainty,
        features=features,
        trajectory_match=features.trajectory_intersects_source,
        time_match=features.time_diff_to_event_min <= 30,
        spatial_match=features.min_distance_to_source_km <= source_radius_km * 1.5,
        drift_consistency=drift > 50,
    )


def rank_candidates(tracks: Dict[str, List[AISRecord]],
                     spill_lat: float, spill_lon: float,
                     source_lat: float, source_lon: float,
                     event_time: datetime,
                     current_bearing: float = 118.0,
                     source_radius_km: float = 10.0
                     ) -> List[CandidateScore]:
    """Score and rank all candidate vessels."""
    scores = []
    for mmsi, track in tracks.items():
        features = compute_vessel_features(
            track, spill_lat, spill_lon,
            source_lat, source_lon, event_time,
            source_radius_km=source_radius_km,
        )
        score = score_candidate(features, source_radius_km, current_bearing)
        scores.append(score)

    scores.sort(key=lambda s: s.total_score, reverse=True)
    return scores
