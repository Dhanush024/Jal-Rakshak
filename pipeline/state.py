"""
Jal-Rakshak — Pipeline State Definition
==========================================
Shared state object for the LangGraph pipeline.
All node results are accumulated here.
"""

from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime


class PipelineState(TypedDict, total=False):
    """
    Complete state passed through the LangGraph pipeline.

    Every key corresponds to a pipeline stage output.
    Nodes read from and write to this state.
    """

    # ── Input ──
    image_path: str               # Path to SAR image
    spill_lat: float              # Detected spill latitude
    spill_lon: float              # Detected spill longitude
    detection_timestamp: str      # ISO-8601
    app_mode: str                 # "demo" or "live"

    # ── SAR Preprocessing (Node 1) ──
    preprocessed: bool
    preprocessing_steps: List[str]

    # ── Detection (Node 2) ──
    spill_detected: bool
    detection_result: Dict[str, Any]    # DetectionResult.to_dict()
    spill_coords: List[Any]
    detection_confidence: float

    # ── Validation (Node 3) ──
    validated: bool
    validation_result: Dict[str, Any]   # ValidationResult.to_dict()
    validation_confidence: float

    # ── Characterization (Node 4) ──
    characterized: bool
    characterization: Dict[str, Any]    # SpillCharacterization.to_dict()
    spill_area_sq_km: float
    age_estimation: Dict[str, Any]      # AgeEstimationResult.to_dict()
    multi_temporal: Optional[Dict[str, Any]]

    # ── Hindcast (Node 5) ──
    hindcast_done: bool
    hindcast_result: Dict[str, Any]     # DriftResult.to_dict()
    source_lat: float
    source_lon: float
    source_uncertainty_km: float
    source_probability: Dict[str, Any]  # SourceProbabilityResult.to_dict()

    # ── AIS Correlation (Node 6) ──
    ais_done: bool
    ais_data_mode: str                  # "REAL", "DEMO", "UNAVAILABLE"
    ais_tracks: Dict[str, Any]
    filtering_result: Dict[str, Any]    # FilteringResult.to_dict()

    # ── Vessel Attribution (Node 7) ──
    attribution_done: bool
    candidate_scores: List[Dict[str, Any]]  # [CandidateScore.to_dict()]
    top_candidate_mmsi: Optional[str]

    # ── Drift Forecast (Node 8) ──
    forecast_done: bool
    forecast_results: List[Dict[str, Any]]  # [DriftResult.to_dict()]
    coastal_done: bool
    coastal_impact: Dict[str, Any]      # CoastalImpactResult.to_dict()

    # ── Risk Assessment (Node 9) ──
    risk_done: bool
    risk_assessment: Dict[str, Any]     # RiskAssessment.to_dict()
    risk_level: str

    # ── Alerts (Node 10) ──
    alerts_done: bool
    alert_log: Dict[str, Any]

    # ── Report (Node 11) ──
    report_done: bool
    incident_report: Dict[str, Any]     # IncidentReport.to_dict()
    incident_id: str

    # ── Metadata ──
    pipeline_start_time: str
    pipeline_end_time: str
    errors: List[str]
    warnings: List[str]
