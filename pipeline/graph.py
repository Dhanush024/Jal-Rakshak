"""
Jal-Rakshak — LangGraph Pipeline Graph
========================================
11-node pipeline:
  detect → validate → characterize → hindcast → ais_correlate →
  attribute_vessels → forecast_drift → assess_risk → alert →
  generate_report

Conditional branching: if no spill detected, pipeline short-circuits
to report (empty) without running downstream nodes.
"""

import os
import logging
from datetime import datetime, timezone, timedelta

from langgraph.graph import StateGraph, START, END
from pipeline.state import PipelineState
from config.settings import (
    is_demo_mode, MODEL_PATH,
    DEFAULT_CURRENT_SPEED_MS, DEFAULT_CURRENT_BEARING_DEG,
    DEFAULT_WIND_FACTOR, DEFAULT_WIND_SPEED_MS, DEFAULT_WIND_BEARING_DEG,
    DEMO_SPILL_LAT, DEMO_SPILL_LON,
)

logger = logging.getLogger("jal_rakshak.pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# NODE 1 — SAR Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def node_preprocess(state: PipelineState) -> dict:
    """Apply SAR preprocessing to the input image."""
    try:
        import cv2
        from sar.preprocessing import preprocess_sar, PreprocessingLevel

        image_path = state.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            return {
                "preprocessed": False,
                "preprocessing_steps": [],
                "warnings": state.get("warnings", []) + [f"Image not found: {image_path}"],
            }

        # Use SARSceneLoader for robust multi-format ingestion and spatial metadata extraction
        from sar.ingestion import SARSceneLoader
        image, meta = SARSceneLoader.load(
            image_path,
            override_lat=state.get("spill_lat"),
            override_lon=state.get("spill_lon"),
        )

        result = preprocess_sar(image, level=PreprocessingLevel.ENHANCED)

        # Save preprocessed image for downstream nodes
        preprocessed_path = image_path.rsplit(".", 1)[0] + "_preprocessed.png"
        cv2.imwrite(preprocessed_path, result.image)

        return {
            "preprocessed": True,
            "preprocessing_steps": result.steps_applied,
            "sar_metadata": meta.to_dict(),
        }
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return {
            "preprocessed": False,
            "preprocessing_steps": [],
            "errors": state.get("errors", []) + [f"Preprocessing: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 2 — YOLO Detection
# ═══════════════════════════════════════════════════════════════════════════

def node_detect(state: PipelineState) -> dict:
    """Run YOLO segmentation for oil spill detection."""
    try:
        from sar.detection import YOLODetector

        image_path = state.get("image_path", "")
        if not os.path.exists(MODEL_PATH):
            return {
                "spill_detected": False,
                "detection_result": {},
                "spill_coords": [],
                "detection_confidence": 0.0,
                "warnings": state.get("warnings", []) + ["YOLO model not found"],
            }

        detector = YOLODetector()
        result = detector.detect(
            image_path,
            detection_timestamp=state.get("detection_timestamp"),
        )

        primary = result.primary_detection
        coords = primary.polygon if primary else []
        confidence = primary.confidence if primary else 0.0

        valid_dets = result.valid_detections
        all_coords = [d.polygon for d in valid_dets] if valid_dets else ([coords] if coords else [])

        return {
            "spill_detected": result.spill_detected,
            "detection_result": result.to_dict(),
            "spill_coords": coords,
            "all_spill_coords": all_coords,
            "detection_confidence": confidence,
        }
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return {
            "spill_detected": False,
            "detection_result": {},
            "spill_coords": [],
            "all_spill_coords": [],
            "detection_confidence": 0.0,
            "errors": state.get("errors", []) + [f"Detection: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 3 — Classical Validation
# ═══════════════════════════════════════════════════════════════════════════

def node_validate(state: PipelineState) -> dict:
    """Cross-validate YOLO detection with classical SAR consensus layer."""
    try:
        import cv2
        import numpy as np
        from sar.classical import validate_consensus, ValidationStatus

        image_path = state.get("image_path", "")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"validated": False, "validation_result": {}, "validation_confidence": 0.0}

        h, w = image.shape[:2]
        yolo_mask = np.zeros((h, w), dtype=np.uint8)

        all_coords = state.get("all_spill_coords", [])
        if not all_coords and state.get("spill_coords"):
            all_coords = [state["spill_coords"]]

        if all_coords:
            for coords in all_coords:
                if coords and len(coords) >= 3:
                    pts = np.array(coords, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(yolo_mask, [pts], 255)

        confidence = state.get("detection_confidence", 0.5)
        result = validate_consensus(yolo_mask=yolo_mask, image=image, yolo_confidence=confidence)

        is_validated = result.final_validation_status in (
            ValidationStatus.CONFIRMED.value,
            ValidationStatus.PROBABLE.value,
        )

        return {
            "validated": is_validated,
            "validation_result": result.to_dict(),
            "validation_status": result.final_validation_status,
            "validation_confidence": result.classical_agreement,
        }
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "validated": False,
            "validation_result": {},
            "validation_confidence": 0.0,
            "errors": state.get("errors", []) + [f"Validation: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 4 — Spill Characterization
# ═══════════════════════════════════════════════════════════════════════════

def node_characterize(state: PipelineState) -> dict:
    """Compute geometric characterization of the spill."""
    try:
        import cv2
        import numpy as np
        from sar.geometry import characterize_spill

        coords = state.get("spill_coords", [])
        if not coords or len(coords) < 3:
            return {"characterized": False, "characterization": {}, "spill_area_sq_km": 0.0}

        image_path = state.get("image_path", "")
        image = cv2.imread(image_path)
        h, w = image.shape[:2] if image is not None else (512, 512)

        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(coords, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

        centroid_geo = None
        if state.get("spill_lat") and state.get("spill_lon"):
            centroid_geo = (state["spill_lat"], state["spill_lon"])

        result = characterize_spill(mask, centroid_geo=centroid_geo)

        # Estimate spill age from morphology and weathering heuristics
        from sar.weathering import SpillAgeEstimator
        age_res = SpillAgeEstimator.estimate_age(result, wind_speed_ms=DEFAULT_WIND_SPEED_MS)

        return {
            "characterized": True,
            "characterization": result.to_dict(),
            "spill_area_sq_km": result.area_sq_km,
            "age_estimation": age_res.to_dict(),
        }
    except Exception as e:
        logger.error(f"Characterization failed: {e}")
        return {
            "characterized": False,
            "characterization": {},
            "spill_area_sq_km": 0.0,
            "age_estimation": {},
            "errors": state.get("errors", []) + [f"Characterization: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 5 — Ocean Hindcast
# ═══════════════════════════════════════════════════════════════════════════

def node_hindcast(state: PipelineState) -> dict:
    """Run backward drift trajectory to estimate spill origin."""
    try:
        from ocean.hindcast import DriftModel

        spill_lat = state.get("spill_lat", DEMO_SPILL_LAT)
        spill_lon = state.get("spill_lon", DEMO_SPILL_LON)

        ts = state.get("detection_timestamp")
        if ts:
            detection_time = datetime.fromisoformat(ts)
        else:
            detection_time = datetime.now(timezone.utc)

        # Query ocean currents and wind via provider abstraction
        from ocean.provider import get_ocean_provider
        ocean_mode = state.get("app_mode", "demo" if is_demo_mode() else "real")
        ocean_prov = get_ocean_provider(mode=ocean_mode)
        current_pts = ocean_prov.get_currents(spill_lat, spill_lon, detection_time)
        wind_pts = ocean_prov.get_wind(spill_lat, spill_lon, detection_time) if hasattr(ocean_prov, "get_wind") else []

        c_speed = current_pts[0].speed_ms if current_pts else DEFAULT_CURRENT_SPEED_MS
        c_bearing = current_pts[0].direction_deg if current_pts else DEFAULT_CURRENT_BEARING_DEG
        w_speed = wind_pts[0].speed_ms if wind_pts else DEFAULT_WIND_SPEED_MS
        w_bearing = wind_pts[0].direction_deg if wind_pts else DEFAULT_WIND_BEARING_DEG

        drift = DriftModel(
            current_speed_ms=c_speed,
            current_bearing_deg=c_bearing,
            wind_factor=DEFAULT_WIND_FACTOR,
            wind_speed_ms=w_speed,
            wind_bearing_deg=w_bearing,
        )

        result = drift.hindcast(spill_lat, spill_lon, detection_time)

        # Generate 2D source probability surface with Monte Carlo particles
        from ocean.probability import SourceProbabilityModel
        particles = drift.particle_ensemble(spill_lat, spill_lon, detection_time, num_particles=80)
        prob_surface = SourceProbabilityModel.generate_surface(
            origin_lat=result.origin_lat,
            origin_lon=result.origin_lon,
            origin_time=result.origin_time,
            final_uncertainty_km=result.final_uncertainty_km,
            particle_endpoints=particles,
        )

        return {
            "hindcast_done": True,
            "hindcast_result": result.to_dict(),
            "source_lat": result.origin_lat,
            "source_lon": result.origin_lon,
            "source_uncertainty_km": result.final_uncertainty_km,
            "source_probability": prob_surface.to_dict(),
        }
    except Exception as e:
        logger.error(f"Hindcast failed: {e}")
        return {
            "hindcast_done": False,
            "hindcast_result": {},
            "source_lat": state.get("spill_lat", DEMO_SPILL_LAT),
            "source_lon": state.get("spill_lon", DEMO_SPILL_LON),
            "source_uncertainty_km": 0.0,
            "source_probability": {},
            "errors": state.get("errors", []) + [f"Hindcast: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 6 — AIS Correlation
# ═══════════════════════════════════════════════════════════════════════════

def node_ais_correlate(state: PipelineState) -> dict:
    """Fetch AIS data and apply progressive filtering."""
    try:
        from ais.filtering import run_filtering_pipeline
        from ais.provider import AISRecord

        ts = state.get("detection_timestamp")
        if ts:
            event_time = datetime.fromisoformat(ts)
        else:
            event_time = datetime.now(timezone.utc)

        spill_lat = state.get("spill_lat", DEMO_SPILL_LAT)
        spill_lon = state.get("spill_lon", DEMO_SPILL_LON)
        source_lat = state.get("source_lat", spill_lat)
        source_lon = state.get("source_lon", spill_lon)

        # Retrieve historical tracks via configured AISProvider (Demo or File)
        from ais.provider import get_ais_provider
        ais_mode = state.get("app_mode", "demo" if is_demo_mode() else "real")
        ais_file = state.get("ais_file_path")
        ais_prov = get_ais_provider(mode=ais_mode, file_path=ais_file)

        # Define search bounding box around spill and estimated source
        min_lat = min(spill_lat, source_lat) - 0.4
        max_lat = max(spill_lat, source_lat) + 0.4
        min_lon = min(spill_lon, source_lon) - 0.4
        max_lon = max(spill_lon, source_lon) + 0.4
        query_bbox = (min_lat, min_lon, max_lat, max_lon)

        query_res = ais_prov.get_historical_tracks(bbox=query_bbox)
        raw_tracks = query_res.tracks
        data_mode = query_res.mode.value
        warnings = list(query_res.warnings)

        if raw_tracks:
            filtering = run_filtering_pipeline(
                tracks=raw_tracks,
                spill_lat=spill_lat,
                spill_lon=spill_lon,
                source_lat=source_lat,
                source_lon=source_lon,
                event_time=event_time,
            )

            # Serialize tracks for state
            serialized = {}
            for mmsi, track in filtering.candidate_tracks.items():
                serialized[mmsi] = [r.to_dict() for r in track]

            return {
                "ais_done": True,
                "ais_data_mode": data_mode,
                "ais_tracks": serialized,
                "filtering_result": filtering.to_dict(),
            }
        else:
            return {
                "ais_done": True,
                "ais_data_mode": data_mode,
                "ais_tracks": {},
                "filtering_result": {},
                "warnings": state.get("warnings", []) + ["No AIS data available"],
            }
    except Exception as e:
        logger.error(f"AIS correlation failed: {e}")
        return {
            "ais_done": False,
            "ais_data_mode": "UNAVAILABLE",
            "ais_tracks": {},
            "filtering_result": {},
            "errors": state.get("errors", []) + [f"AIS: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 7 — Vessel Attribution
# ═══════════════════════════════════════════════════════════════════════════

def node_attribute(state: PipelineState) -> dict:
    """Score and rank candidate vessels."""
    try:
        from ais.attribution import rank_candidates
        from ais.provider import AISRecord
        from datetime import datetime

        ais_tracks_raw = state.get("ais_tracks", {})
        if not ais_tracks_raw:
            return {"attribution_done": False, "candidate_scores": [], "top_candidate_mmsi": None}

        # Deserialize tracks back to AISRecord objects
        tracks = {}
        for mmsi, records in ais_tracks_raw.items():
            tracks[mmsi] = [
                AISRecord(
                    mmsi=r["mmsi"], name=r["name"],
                    lat=r["lat"], lon=r["lon"],
                    heading=r["heading"], speed_knots=r["speed_knots"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    vessel_type=r.get("vessel_type"),
                ) for r in records
            ]

        ts = state.get("detection_timestamp")
        event_time = datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)

        scores = rank_candidates(
            tracks=tracks,
            spill_lat=state.get("spill_lat", DEMO_SPILL_LAT),
            spill_lon=state.get("spill_lon", DEMO_SPILL_LON),
            source_lat=state.get("source_lat", state.get("spill_lat", DEMO_SPILL_LAT)),
            source_lon=state.get("source_lon", state.get("spill_lon", DEMO_SPILL_LON)),
            event_time=event_time,
            current_bearing=DEFAULT_CURRENT_BEARING_DEG,
        )

        return {
            "attribution_done": True,
            "candidate_scores": [s.to_dict() for s in scores],
            "top_candidate_mmsi": scores[0].mmsi if scores else None,
        }
    except Exception as e:
        logger.error(f"Attribution failed: {e}")
        return {
            "attribution_done": False,
            "candidate_scores": [],
            "top_candidate_mmsi": None,
            "errors": state.get("errors", []) + [f"Attribution: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 8 — Drift Forecast
# ═══════════════════════════════════════════════════════════════════════════

def node_forecast(state: PipelineState) -> dict:
    """Predict future oil slick movement."""
    try:
        from ocean.hindcast import DriftModel

        spill_lat = state.get("spill_lat", DEMO_SPILL_LAT)
        spill_lon = state.get("spill_lon", DEMO_SPILL_LON)

        ts = state.get("detection_timestamp")
        current_time = datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)

        drift = DriftModel(
            current_speed_ms=DEFAULT_CURRENT_SPEED_MS,
            current_bearing_deg=DEFAULT_CURRENT_BEARING_DEG,
            wind_factor=DEFAULT_WIND_FACTOR,
            wind_speed_ms=DEFAULT_WIND_SPEED_MS,
            wind_bearing_deg=DEFAULT_WIND_BEARING_DEG,
        )

        results = drift.forecast(spill_lat, spill_lon, current_time,
                                  forecast_hours=[6, 12, 24])

        # Assess coastal proximity, landfall ETA, and sensitive area threats
        from coastal.impact import assess_coastal_impact
        coastal_res = assess_coastal_impact(
            spill_lat=spill_lat,
            spill_lon=spill_lon,
            current_speed_ms=DEFAULT_CURRENT_SPEED_MS,
            current_bearing_deg=DEFAULT_CURRENT_BEARING_DEG,
            wind_speed_ms=DEFAULT_WIND_SPEED_MS,
            wind_bearing_deg=DEFAULT_WIND_BEARING_DEG,
            wind_leeway_factor=DEFAULT_WIND_FACTOR,
            forecast_results=[r.to_dict() for r in results],
        )

        return {
            "forecast_done": True,
            "forecast_results": [r.to_dict() for r in results],
            "coastal_done": True,
            "coastal_impact": coastal_res.to_dict(),
        }
    except Exception as e:
        logger.error(f"Forecast/Coastal impact failed: {e}")
        return {
            "forecast_done": False,
            "forecast_results": [],
            "coastal_done": False,
            "coastal_impact": {},
            "errors": state.get("errors", []) + [f"Forecast: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 9 — Risk Assessment
# ═══════════════════════════════════════════════════════════════════════════

def node_risk(state: PipelineState) -> dict:
    """Calculate multi-factor risk assessment incorporating coastal impact."""
    try:
        from risk.engine import assess_risk

        coastal = state.get("coastal_impact", {})
        eta_to_coast = coastal.get("eta_to_coast_hours")
        sensitive_count = len(coastal.get("threatened_assets", []))

        result = assess_risk(
            spill_area_sq_km=state.get("spill_area_sq_km", 0.0),
            detection_confidence=state.get("detection_confidence", 0.0),
            validation_confidence=state.get("validation_confidence", 0.0),
            wind_speed_ms=DEFAULT_WIND_SPEED_MS,
            current_speed_ms=DEFAULT_CURRENT_SPEED_MS,
            eta_to_coast_hours=eta_to_coast,
            sensitive_areas_nearby=sensitive_count,
        )

        return {
            "risk_done": True,
            "risk_assessment": result.to_dict(),
            "risk_level": result.level.value,
        }
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        return {
            "risk_done": False,
            "risk_assessment": {},
            "risk_level": "UNKNOWN",
            "errors": state.get("errors", []) + [f"Risk: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 10 — Alerts
# ═══════════════════════════════════════════════════════════════════════════

def node_alert(state: PipelineState) -> dict:
    """Generate and dispatch alerts based on risk level."""
    try:
        from alerts.manager import AlertManager

        manager = AlertManager(simulation_mode=True)
        incident_id = f"JR-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        manager.set_incident(incident_id)

        risk_level = state.get("risk_level", "LOW")
        spill_lat = state.get("spill_lat", DEMO_SPILL_LAT)
        spill_lon = state.get("spill_lon", DEMO_SPILL_LON)
        location = f"{spill_lat:.4f}°N, {spill_lon:.4f}°E"

        # Community alert
        manager.generate_community_alert(
            location=location,
            risk_level=risk_level,
            lat=spill_lat, lon=spill_lon,
            recommended_action="Monitor official updates. Avoid affected waters.",
        )

        # Authority alert
        candidate_vessels = [
            s.get("name", s.get("mmsi", "unknown"))
            for s in state.get("candidate_scores", [])[:3]
        ]
        manager.generate_authority_alert(
            location=location,
            risk_level=risk_level,
            spill_area_sq_km=state.get("spill_area_sq_km", 0.0),
            candidate_vessels=candidate_vessels,
        )

        return {
            "alerts_done": True,
            "alert_log": manager.alert_log.to_dict(),
            "incident_id": incident_id,
        }
    except Exception as e:
        logger.error(f"Alert generation failed: {e}")
        return {
            "alerts_done": False,
            "alert_log": {},
            "errors": state.get("errors", []) + [f"Alerts: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 11 — Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def node_report(state: PipelineState) -> dict:
    """Generate comprehensive incident report."""
    try:
        from reporting.incident import generate_report

        incident_id = state.get("incident_id", f"JR-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
        is_demo = is_demo_mode() or state.get("app_mode") == "demo"

        report = generate_report(
            incident_id=incident_id,
            detection_result=state.get("detection_result"),
            characterization=state.get("characterization"),
            validation_result=state.get("validation_result"),
            hindcast_result=state.get("hindcast_result"),
            vessel_scores=state.get("candidate_scores"),
            forecast_results=state.get("forecast_results"),
            coastal_impact=state.get("coastal_impact"),
            risk_assessment=state.get("risk_assessment"),
            alert_log=state.get("alert_log"),
            age_estimation=state.get("age_estimation"),
            source_probability=state.get("source_probability"),
            multi_temporal=state.get("multi_temporal"),
            is_demo=is_demo,
        )

        return {
            "report_done": True,
            "incident_report": report.to_dict(),
            "pipeline_end_time": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            "report_done": False,
            "incident_report": {},
            "pipeline_end_time": datetime.now(timezone.utc).isoformat(),
            "errors": state.get("errors", []) + [f"Report: {str(e)}"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE — skip downstream if no spill detected
# ═══════════════════════════════════════════════════════════════════════════

def route_after_detection(state: PipelineState) -> str:
    """If no spill detected, skip to report. Otherwise continue."""
    if state.get("spill_detected"):
        return "validate"
    return "report"


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_pipeline() -> StateGraph:
    """
    Build the 11-node LangGraph pipeline.

    Flow:
      preprocess → detect →(if spill)→ validate → characterize → hindcast →
      ais_correlate → attribute → forecast → risk → alert → report → END

      detect →(no spill)→ report → END
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("preprocess", node_preprocess)
    graph.add_node("detect", node_detect)
    graph.add_node("validate", node_validate)
    graph.add_node("characterize", node_characterize)
    graph.add_node("hindcast", node_hindcast)
    graph.add_node("ais_correlate", node_ais_correlate)
    graph.add_node("attribute", node_attribute)
    graph.add_node("forecast", node_forecast)
    graph.add_node("risk", node_risk)
    graph.add_node("alert", node_alert)
    graph.add_node("report", node_report)

    # Edges
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "detect")

    # Conditional branch after detection
    graph.add_conditional_edges("detect", route_after_detection, {
        "validate": "validate",
        "report": "report",
    })

    # Linear chain when spill detected
    graph.add_edge("validate", "characterize")
    graph.add_edge("characterize", "hindcast")
    graph.add_edge("hindcast", "ais_correlate")
    graph.add_edge("ais_correlate", "attribute")
    graph.add_edge("attribute", "forecast")
    graph.add_edge("forecast", "risk")
    graph.add_edge("risk", "alert")
    graph.add_edge("alert", "report")

    graph.add_edge("report", END)

    return graph


def compile_pipeline():
    """Build and compile the pipeline. Returns a runnable."""
    graph = build_pipeline()
    return graph.compile()


def run_pipeline(image_path: str,
                 spill_lat: float = None,
                 spill_lon: float = None,
                 detection_timestamp: str = None,
                 app_mode: str = None,
                 ais_file_path: str = None) -> PipelineState:
    """
    Convenience function to run the full pipeline.

    Args:
        image_path: Path to SAR image
        spill_lat: Spill latitude (defaults to demo value)
        spill_lon: Spill longitude (defaults to demo value)
        detection_timestamp: ISO timestamp of detection (defaults to scenario time in demo)
        app_mode: "demo", "real", or "auto" (defaults to config)
        ais_file_path: Optional path to real historical AIS CSV/JSON file

    Returns:
        Final PipelineState with all results
    """
    from config.settings import is_demo_mode

    if spill_lat is None:
        spill_lat = DEMO_SPILL_LAT
    if spill_lon is None:
        spill_lon = DEMO_SPILL_LON
    if app_mode is None:
        app_mode = "demo" if is_demo_mode() else "real"

    if detection_timestamp is None:
        if app_mode == "demo":
            from demo.scenario import CHENNAI_SCENARIO
            detection_timestamp = CHENNAI_SCENARIO.detection_time.isoformat()
        else:
            detection_timestamp = datetime.now(timezone.utc).isoformat()

    initial_state: PipelineState = {
        "image_path": image_path,
        "spill_lat": spill_lat,
        "spill_lon": spill_lon,
        "detection_timestamp": detection_timestamp,
        "app_mode": app_mode,
        "ais_file_path": ais_file_path,
        "pipeline_start_time": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "warnings": [],
    }

    compiled = compile_pipeline()
    return compiled.invoke(initial_state)
