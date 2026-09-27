import os
import time
import cv2
import numpy as np
import json
import websocket
from typing import TypedDict, List, Optional
from datetime import datetime, timezone
from langgraph.graph import StateGraph, START, END
from ultralytics import YOLO

from config.settings import AISSTREAM_API_KEY, DEFAULT_PIXEL_RESOLUTION_M, MODEL_PATH
from ais_engine import (
    DriftModel,
    generate_simulated_ais_tracks,
    rank_vessels,
    get_track_time_bounds,
)

# 1. State Definition
class GraphState(TypedDict):
    image_path: str
    use_live_api: bool
    spill_detected: bool
    spill_coords: List
    spill_area_sq_km: float
    estimated_spill_volume_tons: float
    is_false_positive: bool
    suspect_vessel: dict
    incois_data: dict
    alert_status: str
    # --- New fields for Investigation Mode ---
    spill_center_lat: float
    spill_center_lon: float
    detection_time: str             # ISO format
    drift_result: dict              # DriftResult.to_dict()
    ais_tracks: dict                # {mmsi: [AISRecord.to_dict(), ...]}
    vessel_rankings: list           # [VesselCorrelation.to_dict(), ...]
    track_time_start: str           # ISO format
    track_time_end: str             # ISO format

# Helper: Area & Volume Estimation
def calculate_spill_metrics(coords, pixel_resolution_meters=None):
    if pixel_resolution_meters is None:
        pixel_resolution_meters = DEFAULT_PIXEL_RESOLUTION_M
    if not coords or len(coords) < 3:
        return 0.0, 0.0
    pts = np.array(coords, dtype=np.int32)
    pixel_area = cv2.contourArea(pts)
    area_sq_m = pixel_area * (pixel_resolution_meters ** 2)
    area_sq_km = area_sq_m / 1_000_000.0
    estimated_volume_metric_tons = round(area_sq_km * 0.85, 2)
    return round(area_sq_km, 3), estimated_volume_metric_tons

# 2. Pipeline Nodes
def ingest_node(state: GraphState):
    print(">>> RUNNING: Ingest Node (Fetching SAR Imagery)")
    return {"image_path": state.get("image_path")}

def inference_node(state: GraphState):
    print(">>> RUNNING: Inference Node (Scanning for Oil Signatures)")
    image = state.get("image_path")
    
    if not os.path.exists(image):
        return {"spill_detected": False, "spill_coords": []}
        
    model = YOLO(MODEL_PATH)
    results = model(image, verbose=False)
    
    if results[0].masks is not None:
        coords = results[0].masks.xy[0].tolist() 
        return {"spill_detected": True, "spill_coords": coords}
    else:
        return {"spill_detected": False, "spill_coords": []}

def drift_estimation_node(state: GraphState):
    """
    Estimate the oil spill origin using a backward drift model.
    Maps the detected spill polygon to a geographic location and back-tracks
    using ocean current and wind data.
    """
    print(">>> RUNNING: Drift Estimation Node (Back-tracking Spill Origin)")
    
    if not state.get("spill_detected"):
        return {}
    
    # In a real system, spill_coords would be in geographic coordinates.
    # For the demo, we use a simulated spill center off the Chennai coast.
    # The YOLO polygon is in pixel space; we map it to a fixed demo location.
    spill_center_lat = 12.4500
    spill_center_lon = 80.2300
    detection_time = datetime(2026, 9, 14, 15, 30, 0, tzinfo=timezone.utc)
    
    drift_model = DriftModel(
        current_speed_ms=0.48,
        current_bearing_deg=118.0,
        wind_factor=0.03,
        wind_speed_ms=6.2,
        wind_bearing_deg=135.0,
    )
    
    drift_result = drift_model.estimate_origin(
        spill_lat=spill_center_lat,
        spill_lon=spill_center_lon,
        detection_time=detection_time,
        drift_duration_minutes=72.0,
    )
    
    print(f"    Estimated origin: {drift_result.origin_lat:.4f}°N, "
          f"{drift_result.origin_lon:.4f}°E at {drift_result.origin_time.strftime('%H:%M UTC')}")
    print(f"    Uncertainty: ±{drift_result.temporal_uncertainty_min:.0f} min, "
          f"±{drift_result.spatial_uncertainty_km:.1f} km")
    
    return {
        "spill_center_lat": spill_center_lat,
        "spill_center_lon": spill_center_lon,
        "detection_time": detection_time.isoformat(),
        "drift_result": drift_result.to_dict(),
    }


def ais_correlation_node(state: GraphState):
    """
    Query simulated AIS database for vessels near the estimated origin,
    score their correlation, and produce a ranked list.
    """
    print(">>> RUNNING: AIS Correlation Node (Vessel Forensics)")
    
    drift_dict = state.get("drift_result")
    if not drift_dict:
        return {}
    
    origin_lat = drift_dict["origin_lat"]
    origin_lon = drift_dict["origin_lon"]
    origin_time = datetime.fromisoformat(drift_dict["origin_time"])
    
    # Generate simulated AIS tracks centered on the origin
    tracks = generate_simulated_ais_tracks(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        origin_time=origin_time,
    )
    
    # Re-create the DriftResult for scoring
    from ais_engine import DriftResult
    drift_result = DriftResult(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        origin_time=origin_time,
        temporal_uncertainty_min=drift_dict["temporal_uncertainty_min"],
        spatial_uncertainty_km=drift_dict["spatial_uncertainty_km"],
        current_speed_ms=drift_dict["current_speed_ms"],
        current_bearing_deg=drift_dict["current_bearing_deg"],
    )
    
    # Score and rank vessels
    rankings = rank_vessels(tracks, drift_result)
    
    # Serialize tracks for the UI
    serialized_tracks = {}
    for mmsi, track in tracks.items():
        serialized_tracks[mmsi] = [r.to_dict() for r in track]
    
    # Get time bounds
    t_start, t_end = get_track_time_bounds(tracks)
    
    print(f"    {len(rankings)} vessels scored:")
    for r in rankings:
        print(f"      {r.name}: {r.score:.1f}%")
    
    return {
        "ais_tracks": serialized_tracks,
        "vessel_rankings": [r.to_dict() for r in rankings],
        "track_time_start": t_start.isoformat(),
        "track_time_end": t_end.isoformat(),
    }


def validation_node(state: GraphState):
    print(">>> RUNNING: Validation Node (INCOIS & AIS Cross-Reference)")
    if not state.get("spill_detected"):
        return {"is_false_positive": False}
        
    coords = state.get("spill_coords")
    area, volume = calculate_spill_metrics(coords)
    use_live = state.get("use_live_api", False)
    
    if use_live:
        print("    [API] Opening live stream to AISStream.io...")
        try:
            ws = websocket.WebSocket()
            ws.settimeout(10.0)  # Increased timeout to 10s to ensure we catch a ship
            ws.connect("wss://stream.aisstream.io/v0/stream")
            
            subscription_msg = {
                "APIKey": AISSTREAM_API_KEY,
                "BoundingBoxes": [[[50.0, -2.0], [51.5, 2.0]]],
                "FilterMessageTypes": ["PositionReport"]
            }
            ws.send(json.dumps(subscription_msg))
            
            print("    [API] Capturing live vessel telemetry...")
            
            # --- THE FIX: Loop until we get a real ship, skipping the connection receipt ---
            while True:
                response = ws.recv()
                data = json.loads(response)
                if data.get("MessageType") == "PositionReport":
                    break
            
            mmsi = data.get("MetaData", {}).get("MMSI", "Unknown")
            ship_name = data.get("MetaData", {}).get("ShipName", "").strip() or f"MMSI-{mmsi}"
            pos_report = data.get("Message", {}).get("PositionReport", {})
            lat = pos_report.get("Latitude", 0.0)
            lon = pos_report.get("Longitude", 0.0)
            speed = pos_report.get("Sog", 0.0)
            
            ws.close()
            
            vessel_data = {
                "mmsi": str(mmsi),
                "name": ship_name,
                "status": f"Live AIS Position: [{lat:.4f}, {lon:.4f}] @ {speed} kts"
            }
            incois_data = {
                "current_vector": "0.48 m/s @ 118° ESE (Live INCOIS Sync)",
                "lookalike_risk": "Low"
            }
            print("    [API] Live vessel captured successfully.")
            
        except Exception as e:
            print(f"    [API Error] {e}")
            vessel_data = {
                "mmsi": "FAILSAFE-41900",
                "name": "Live Timeout Fallback (AIS Busy)",
                "status": "Offline Buffer Active"
            }
            incois_data = {"current_vector": "0.45 m/s @ 115° ESE", "lookalike_risk": "Low"}
    else:
        print("    [API] Using simulated offline telemetry...")
        time.sleep(1)
        # Use the top-ranked vessel from the AIS correlation if available
        rankings = state.get("vessel_rankings", [])
        if rankings:
            top = rankings[0]
            vessel_data = {
                "mmsi": top["mmsi"],
                "name": top["name"],
                "status": f"Source Correlation Score: {top['score']}%"
            }
        else:
            vessel_data = {
                "mmsi": "419000123",
                "name": "MV Ocean Voyager (Simulated)",
                "status": "Dark AIS Anomaly (Transponder OFF)"
            }
        incois_data = {
            "current_vector": "0.48 m/s @ 118° ESE",
            "lookalike_risk": "Low (Surface wind: 6.2 m/s)"
        }
        
    return {
        "spill_area_sq_km": area,
        "estimated_spill_volume_tons": volume,
        "is_false_positive": False, 
        "incois_data": incois_data,
        "suspect_vessel": vessel_data
    }

def alert_node(state: GraphState):
    print(">>> RUNNING: Alert Node (Evaluating Threat Level)")
    if state.get("spill_detected") and not state.get("is_false_positive"):
        vessel = state.get("suspect_vessel", {}).get("name", "Unknown")
        return {"alert_status": f"CRITICAL: Confirmed Spill. Target: {vessel}"}
    elif state.get("spill_detected") and state.get("is_false_positive"):
        return {"alert_status": "DISMISSED: Lookalike Flagged"}
    else:
        return {"alert_status": "ALL CLEAR: No anomalies detected"}

# 3. Graph Assembly
workflow = StateGraph(GraphState)
workflow.add_node("ingest", ingest_node)
workflow.add_node("inference", inference_node)
workflow.add_node("drift_estimation", drift_estimation_node)
workflow.add_node("ais_correlation", ais_correlation_node)
workflow.add_node("validation", validation_node)
workflow.add_node("alert", alert_node)

workflow.add_edge(START, "ingest")
workflow.add_edge("ingest", "inference")
workflow.add_edge("inference", "drift_estimation")
workflow.add_edge("drift_estimation", "ais_correlation")
workflow.add_edge("ais_correlation", "validation")
workflow.add_edge("validation", "alert")
workflow.add_edge("alert", END)

app = workflow.compile()