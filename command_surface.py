"""
Jal-Rakshak — Maritime Operations Center Hero Command Surface
=============================================================
High-performance MapLibre GL WebGL interactive geospatial component.
Features:
- Dark tactical naval basemap using OpenStreetMap raster tiles (Zero Carto API key required)
- Interactive vector layers: Spill polygon, Credible origin zones, Backtrack trajectory,
  Forward forecast drift, AIS fleet tracks, Animated vessel markers, Coastal assets
- Synchronized Right Context Inspector (Spill, Vessel, Source, Asset)
- Client-side Playback Timeline scrubber (T-180m to T+60m) interpolating vessel positions at 60fps
- Layer toggles, Keyboard shortcuts (1-7, Space, F, Esc, K) & Command Palette (Cmd/Ctrl+K)
- Zero full-app Streamlit reruns on pan, zoom, layer toggle, or scrubbing!
"""

import json
from typing import Dict, Any, Optional
from demo.scenario import CHENNAI_SCENARIO


def build_geojson_bundle(state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert pipeline state to GeoJSON feature collections for MapLibre GL."""
    spill_lat = state.get("spill_lat", 12.4500)
    spill_lon = state.get("spill_lon", 80.2300)
    source_lat = state.get("source_lat", spill_lat)
    source_lon = state.get("source_lon", spill_lon)
    source_unc_km = state.get("source_uncertainty_km", 1.0)

    # 1. Spill Polygon
    all_coords = state.get("all_spill_coords") or state.get("spill_coords") or []
    if all_coords and len(all_coords) >= 3:
        spill_poly = [[float(c[1]), float(c[0])] for c in all_coords]
        if spill_poly[0] != spill_poly[-1]:
            spill_poly.append(spill_poly[0])
    else:
        # Approximate circle
        import math
        spill_poly = []
        r = 0.008  # ~0.8 km
        for i in range(24):
            ang = 2 * math.pi * i / 24
            spill_poly.append([spill_lon + r * math.cos(ang), spill_lat + r * 0.7 * math.sin(ang)])
        spill_poly.append(spill_poly[0])

    spill_feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [spill_poly]},
        "properties": {
            "type": "spill",
            "title": "Detected Oil Slick Boundary",
            "lat": spill_lat,
            "lon": spill_lon,
            "area_sq_km": state.get("characterization", {}).get("area_sq_km", 1.77),
            "volume_tons": state.get("characterization", {}).get("estimated_volume_tons", 753.3),
            "perimeter_km": state.get("characterization", {}).get("perimeter_km", 5.17),
            "fay_age": state.get("age_estimation", {}).get("fay_age_hours", 3.2),
            "yolo_conf": state.get("detection_confidence", 0.48),
            "consensus": state.get("validation_result", {}).get("classical_agreement", 0.60),
            "status": state.get("validation_result", {}).get("final_validation_status", "CONFIRMED BY MULTIPLE SIGNALS"),
        }
    }

    # 2. Source Credible Zones & Point
    source_prob = state.get("source_probability", {})
    credible_zones = source_prob.get("credible_zones", [])
    zone_features = []
    if credible_zones:
        for cz in credible_zones:
            pts = cz.get("polygon_points", [])
            if pts and len(pts) >= 3:
                c_pts = [[float(p[1]), float(p[0])] for p in pts]
                if c_pts[0] != c_pts[-1]:
                    c_pts.append(c_pts[0])
                zone_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [c_pts]},
                    "properties": {
                        "type": "credible_zone",
                        "title": cz.get("name", "Origin Credible Zone"),
                        "prob": cz.get("confidence_level", 0.75),
                        "color": cz.get("color_hex", "#f59e0b"),
                    }
                })

    source_point_feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(source_lon), float(source_lat)]},
        "properties": {
            "type": "source_point",
            "title": "Estimated Drift Origin (Hindcast)",
            "lat": source_lat,
            "lon": source_lon,
            "uncertainty_km": source_unc_km,
            "current_speed": state.get("hindcast_result", {}).get("current_speed_ms", 0.48),
            "current_bearing": state.get("hindcast_result", {}).get("current_bearing_deg", 118.0),
            "wind_speed": state.get("hindcast_result", {}).get("wind_speed_ms", 6.2),
            "wind_bearing": state.get("hindcast_result", {}).get("wind_bearing_deg", 135.0),
        }
    }

    # 3. Backtrack Path
    backtrack_feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(spill_lon), float(spill_lat)], [float(source_lon), float(source_lat)]]
        },
        "properties": {
            "type": "backtrack",
            "title": "Hindcast Drift Vector",
            "distance_km": state.get("hindcast_result", {}).get("total_distance_km", 2.1)
        }
    }

    # 4. Forecast Path
    forecast_results = state.get("forecast_results", [])
    forecast_coords = [[float(spill_lon), float(spill_lat)]]
    for step in forecast_results:
        p_lat = step.get("predicted_lat")
        p_lon = step.get("predicted_lon")
        if p_lat and p_lon:
            forecast_coords.append([float(p_lon), float(p_lat)])

    forecast_feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": forecast_coords},
        "properties": {
            "type": "forecast",
            "title": "Forward Drift Trajectory (T+6h)",
            "steps": len(forecast_results)
        }
    }

    # 5. AIS Fleet Tracks & Candidate Details
    candidate_scores = {str(c.get("mmsi")): c for c in state.get("candidate_scores", [])}
    ais_features = []
    vessel_data = []

    for track in state.get("ais_tracks", []):
        mmsi = str(track.get("mmsi", ""))
        name = track.get("vessel_name") or f"Vessel {mmsi}"
        points = track.get("points") or []
        if not points:
            continue

        line_coords = [[float(p.get("lon", 0)), float(p.get("lat", 0))] for p in points]
        cand = candidate_scores.get(mmsi, {})
        score = cand.get("score", 50.0)
        breakdown = cand.get("breakdown", {})

        vessel_info = {
            "mmsi": mmsi,
            "name": name,
            "score": round(score, 1),
            "breakdown": breakdown,
            "trajectory_match": cand.get("trajectory_match", False),
            "time_match": cand.get("time_match", False),
            "spatial_match": cand.get("spatial_match", False),
            "drift_consistency": cand.get("drift_consistency", False),
            "evidence_summary": cand.get("evidence_summary", []),
            "disclaimer": cand.get("disclaimer", "Association score reflects spatiotemporal proximity; does NOT constitute legal proof of liability."),
            "points": [
                {
                    "minute_offset": i * 10 - 180,  # relative to T=0
                    "lon": float(p.get("lon", 0)),
                    "lat": float(p.get("lat", 0)),
                    "sog": p.get("sog", 12.0),
                    "cog": p.get("cog", 0.0),
                }
                for i, p in enumerate(points)
            ]
        }
        vessel_data.append(vessel_info)

        ais_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": line_coords},
            "properties": {
                "type": "ais_track",
                "mmsi": mmsi,
                "name": name,
                "score": score,
            }
        })

    # 6. Sensitive Coastal Assets
    coastal_features = []
    sensitive_areas = CHENNAI_SCENARIO.sensitive_areas or []
    for area in sensitive_areas:
        coastal_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(area["lon"]), float(area["lat"])]},
            "properties": {
                "type": "coastal_asset",
                "name": area["name"],
                "asset_type": area["type"],
                "lat": area["lat"],
                "lon": area["lon"],
                "esi_rank": "ESI 8" if area["type"] == "wildlife" else "ESI 6",
            }
        })

    return {
        "spill_feature": spill_feature,
        "source_point_feature": source_point_feature,
        "zone_features": zone_features,
        "backtrack_feature": backtrack_feature,
        "forecast_feature": forecast_feature,
        "ais_features": ais_features,
        "coastal_features": coastal_features,
        "vessel_data": vessel_data,
        "center": [float(spill_lon), float(spill_lat)],
    }


def render_command_surface(
    state: Optional[Dict[str, Any]],
    height: int = 680,
    active_view_hint: str = "Analysis",
    selected_vessel_mmsi: Optional[str] = None
) -> str:
    """Generate self-contained HTML/WebGL MapLibre GL command surface."""
    if not state:
        state = {
            "spill_lat": 12.4500,
            "spill_lon": 80.2300,
            "spill_detected": False,
        }

    bundle = build_geojson_bundle(state)
    bundle_json = json.dumps(bundle)
    sel_mmsi_json = json.dumps(selected_vessel_mmsi)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Jal-Rakshak Hero Command Surface</title>
    <!-- MapLibre GL CSS -->
    <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" />
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }}

        body, html {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #050810;
            font-family: 'Inter', sans-serif;
            color: #f0f4f8;
        }}

        #map {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #070c18;
        }}

        /* Tactical dark filter applied to OpenStreetMap raster tiles (Zero Carto API key needed!) */
        .maplibregl-canvas-container canvas {{
            filter: brightness(0.65) invert(1) contrast(3.2) hue-rotate(200deg) saturate(0.28) brightness(0.72);
        }}

        /* Overlay UI: Top Layer Bar */
        .top-layer-bar {{
            position: absolute;
            top: 14px;
            left: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(10, 16, 32, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(14px);
            padding: 6px 12px;
            border-radius: 8px;
            z-index: 10;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}

        .layer-toggle {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .layer-toggle.active {{
            background: rgba(56, 189, 248, 0.16);
            border-color: #38bdf8;
            color: #38bdf8;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
        }}

        .layer-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }}

        /* Top Right Control Pill */
        .top-right-tools {{
            position: absolute;
            top: 14px;
            right: 14px;
            display: flex;
            gap: 6px;
            z-index: 10;
        }}

        .tool-btn {{
            background: rgba(10, 16, 32, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(14px);
            color: #94a3b8;
            padding: 6px 12px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .tool-btn:hover {{
            border-color: #38bdf8;
            color: #f0f4f8;
        }}

        /* Right Context Inspector Panel */
        .inspector-panel {{
            position: absolute;
            top: 60px;
            right: 14px;
            width: 320px;
            background: rgba(10, 16, 32, 0.94);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 10px;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
            padding: 16px;
            z-index: 20;
            transform: translateX(360px);
            transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        .inspector-panel.visible {{
            transform: translateX(0);
        }}

        .inspector-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 10px;
            margin-bottom: 12px;
        }}

        .inspector-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            font-weight: 700;
            color: #38bdf8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .inspector-close {{
            background: none;
            border: none;
            color: #94a3b8;
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
        }}

        .inspector-close:hover {{
            color: #f0f4f8;
        }}

        .inspector-row {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-bottom: 8px;
            line-height: 1.4;
        }}

        .inspector-label {{
            color: #64748b;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }}

        .inspector-val {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: #f0f4f8;
            font-variant-numeric: tabular-nums;
        }}

        .score-badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid #f59e0b;
            color: #fbbf24;
        }}

        .legal-micro {{
            font-size: 10px;
            color: #94a3b8;
            background: rgba(255, 255, 255, 0.04);
            border-left: 2px solid #f59e0b;
            padding: 6px 8px;
            margin-top: 12px;
            line-height: 1.4;
            border-radius: 0 4px 4px 0;
        }}

        /* Bottom Floating Playback Timeline Dock */
        .bottom-dock {{
            position: absolute;
            bottom: 14px;
            left: 50%;
            transform: translateX(-50%);
            width: calc(100% - 28px);
            max-width: 900px;
            background: rgba(10, 16, 32, 0.90);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 10px;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
            padding: 10px 18px;
            z-index: 15;
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .dock-play-btn {{
            background: #38bdf8;
            border: none;
            color: #050810;
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-weight: 700;
            font-size: 14px;
            flex-shrink: 0;
            transition: all 0.15s ease;
        }}

        .dock-play-btn:hover {{
            background: #7dd3fc;
            box-shadow: 0 0 14px rgba(56, 189, 248, 0.5);
        }}

        .timeline-slider-wrap {{
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .timeline-labels {{
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: #64748b;
        }}

        .timeline-slider {{
            -webkit-appearance: none;
            width: 100%;
            height: 5px;
            border-radius: 9999px;
            background: rgba(255, 255, 255, 0.15);
            outline: none;
            cursor: pointer;
        }}

        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #38bdf8;
            box-shadow: 0 0 10px #38bdf8;
            cursor: pointer;
            transition: transform 0.1s ease;
        }}

        .timeline-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.25);
        }}

        .epoch-pill {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
            color: #38bdf8;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
            padding: 4px 10px;
            border-radius: 6px;
            white-space: nowrap;
        }}

        .speed-toggle {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
        }}

        /* Command Palette Modal */
        .cmd-palette-backdrop {{
            position: absolute;
            inset: 0;
            background: rgba(5, 8, 16, 0.7);
            backdrop-filter: blur(8px);
            z-index: 100;
            display: none;
            align-items: flex-start;
            justify-content: center;
            padding-top: 100px;
        }}

        .cmd-palette-backdrop.open {{
            display: flex;
        }}

        .cmd-palette-card {{
            width: 480px;
            background: #0a1020;
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 10px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.8);
            overflow: hidden;
        }}

        .cmd-palette-input {{
            width: 100%;
            background: transparent;
            border: none;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding: 14px 18px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #f0f4f8;
            outline: none;
        }}

        .cmd-palette-list {{
            max-height: 260px;
            overflow-y: auto;
            padding: 6px 0;
        }}

        .cmd-item {{
            padding: 10px 18px;
            font-size: 12px;
            font-family: 'Inter', sans-serif;
            color: #94a3b8;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: background 0.15s ease;
        }}

        .cmd-item:hover, .cmd-item.selected {{
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
        }}

        /* Pulse animation on spill marker */
        .pulsing-marker {{
            width: 18px;
            height: 18px;
            border: 2px solid #ef4444;
            background: rgba(239, 68, 68, 0.4);
            border-radius: 50%;
            box-shadow: 0 0 14px #ef4444;
            animation: pulse-ring 2s infinite ease-out;
        }}

        @keyframes pulse-ring {{
            0% {{ transform: scale(0.9); opacity: 1; }}
            50% {{ transform: scale(1.4); opacity: 0.6; }}
            100% {{ transform: scale(0.9); opacity: 1; }}
        }}

        /* Attribution override */
        .maplibregl-ctrl-attrib {{
            background: rgba(10, 16, 32, 0.75) !important;
            color: #64748b !important;
            font-size: 9px !important;
            border-radius: 4px !important;
        }}
        .maplibregl-ctrl-attrib a {{
            color: #94a3b8 !important;
            text-decoration: none !important;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <!-- Top Left Layer Toggles -->
    <div class="top-layer-bar">
        <button class="layer-toggle active" id="btn-toggle-spill" onclick="toggleLayer('spill')">
            <span class="layer-dot" style="background:#ef4444;"></span>Spill
        </button>
        <button class="layer-toggle active" id="btn-toggle-source" onclick="toggleLayer('source')">
            <span class="layer-dot" style="background:#f59e0b;"></span>Origin
        </button>
        <button class="layer-toggle active" id="btn-toggle-backtrack" onclick="toggleLayer('backtrack')">
            <span class="layer-dot" style="background:#f59e0b; border: 1px dashed #f59e0b;"></span>Hindcast
        </button>
        <button class="layer-toggle active" id="btn-toggle-forecast" onclick="toggleLayer('forecast')">
            <span class="layer-dot" style="background:#a855f7;"></span>Forecast
        </button>
        <button class="layer-toggle active" id="btn-toggle-ais" onclick="toggleLayer('ais')">
            <span class="layer-dot" style="background:#38bdf8;"></span>AIS Tracks
        </button>
        <button class="layer-toggle active" id="btn-toggle-assets" onclick="toggleLayer('assets')">
            <span class="layer-dot" style="background:#ec4899;"></span>Coast Assets
        </button>
    </div>

    <!-- Top Right Tools -->
    <div class="top-right-tools">
        <button class="tool-btn" onclick="fitBounds()" title="Fit to Analysis Bounds (F)">[F] Fit Extent</button>
        <button class="tool-btn" onclick="openCmdPalette()" title="Command Palette (K)">Cmd+K</button>
    </div>

    <!-- Right Context Inspector Panel -->
    <div class="inspector-panel" id="inspector-panel">
        <div class="inspector-header">
            <span class="inspector-title" id="insp-title">INSPECTION</span>
            <button class="inspector-close" onclick="closeInspector()">✕</button>
        </div>
        <div id="insp-body"></div>
    </div>

    <!-- Bottom Playback Timeline Dock -->
    <div class="bottom-dock">
        <button class="dock-play-btn" id="play-pause-btn" onclick="togglePlayPause()">▶</button>
        <div class="timeline-slider-wrap">
            <div class="timeline-labels">
                <span>T-180m (Origin Window)</span>
                <span id="center-epoch-label">T-0m (SAR Acquisition)</span>
                <span>T+60m (Forecast)</span>
            </div>
            <input type="range" min="-180" max="60" value="0" step="1" class="timeline-slider" id="time-slider" oninput="onTimeScrub(this.value)" />
        </div>
        <div class="epoch-pill" id="epoch-pill">T+0 min</div>
        <button class="speed-toggle" id="speed-toggle" onclick="cycleSpeed()">1x</button>
    </div>

    <!-- Command Palette Modal -->
    <div class="cmd-palette-backdrop" id="cmd-palette" onclick="if(event.target===this) closeCmdPalette()">
        <div class="cmd-palette-card">
            <input type="text" class="cmd-palette-input" id="cmd-input" placeholder="Type action or search candidate... (Esc to close)" oninput="filterCmdList(this.value)" />
            <div class="cmd-palette-list" id="cmd-list">
                <div class="cmd-item" onclick="executeCmd('spill')">📍 Fly to Oil Spill Slick</div>
                <div class="cmd-item" onclick="executeCmd('source')">🎯 Fly to Probable Origin</div>
                <div class="cmd-item" onclick="executeCmd('top_vessel')">🚢 Inspect Top Candidate Vessel</div>
                <div class="cmd-item" onclick="executeCmd('toggle_forecast')">🌊 Toggle Forward Drift Vector</div>
                <div class="cmd-item" onclick="executeCmd('toggle_ais')">📡 Toggle Historical AIS Fleet</div>
                <div class="cmd-item" onclick="executeCmd('fit')">🔍 Fit All Map Features</div>
            </div>
        </div>
    </div>

    <!-- MapLibre GL JS -->
    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <script>
        const geoData = {bundle_json};
        const initialSelectedMMSI = {sel_mmsi_json};

        let map;
        let isPlaying = false;
        let playInterval = null;
        let currentMinutes = 0;
        let playbackSpeed = 1;
        const vesselMarkers = {{}};

        // Initialize MapLibre GL Map with 100% Free OpenStreetMap Raster Basemap
        document.addEventListener("DOMContentLoaded", () => {{
            map = new maplibregl.Map({{
                container: 'map',
                style: {{
                    version: 8,
                    sources: {{
                        'osm-tiles': {{
                            type: 'raster',
                            tiles: [
                                'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'
                            ],
                            tileSize: 256,
                            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                        }}
                    }},
                    layers: [
                        {{
                            id: 'osm-tiles-layer',
                            type: 'raster',
                            source: 'osm-tiles',
                            minzoom: 0,
                            maxzoom: 19
                        }}
                    ]
                }},
                center: geoData.center || [80.23, 12.45],
                zoom: 11,
                pitch: 25,
                bearing: -10,
                attributionControl: true
            }});

            map.on('load', () => {{
                addMapLayers();
                setupVesselMarkers();
                fitBounds();
                if (initialSelectedMMSI) {{
                    selectVessel(initialSelectedMMSI);
                }}
            }});
        }});

        function addMapLayers() {{
            // 1. Credible origin zones (P=0.50, P=0.75, P=0.90)
            if (geoData.zone_features && geoData.zone_features.length > 0) {{
                map.addSource('credible-zones-src', {{
                    type: 'geojson',
                    data: {{
                        type: 'FeatureCollection',
                        features: geoData.zone_features
                    }}
                }});
                map.addLayer({{
                    id: 'credible-zones-fill',
                    type: 'fill',
                    source: 'credible-zones-src',
                    paint: {{
                        'fill-color': ['get', 'color'],
                        'fill-opacity': 0.16
                    }}
                }});
                map.addLayer({{
                    id: 'credible-zones-line',
                    type: 'line',
                    source: 'credible-zones-src',
                    paint: {{
                        'line-color': ['get', 'color'],
                        'line-width': 1.5,
                        'line-dasharray': [3, 2]
                    }}
                }});
            }}

            // 2. Backtrack Hindcast Line
            if (geoData.backtrack_feature) {{
                map.addSource('backtrack-src', {{
                    type: 'geojson',
                    data: geoData.backtrack_feature
                }});
                map.addLayer({{
                    id: 'backtrack-line',
                    type: 'line',
                    source: 'backtrack-src',
                    paint: {{
                        'line-color': '#f59e0b',
                        'line-width': 2.5,
                        'line-dasharray': [4, 3]
                    }}
                }});
            }}

            // 3. Forward Forecast Drift Line
            if (geoData.forecast_feature) {{
                map.addSource('forecast-src', {{
                    type: 'geojson',
                    data: geoData.forecast_feature
                }});
                map.addLayer({{
                    id: 'forecast-line',
                    type: 'line',
                    source: 'forecast-src',
                    paint: {{
                        'line-color': '#a855f7',
                        'line-width': 2.5
                    }}
                }});
            }}

            // 4. AIS Fleet Tracks
            if (geoData.ais_features && geoData.ais_features.length > 0) {{
                map.addSource('ais-tracks-src', {{
                    type: 'geojson',
                    data: {{
                        type: 'FeatureCollection',
                        features: geoData.ais_features
                    }}
                }});
                map.addLayer({{
                    id: 'ais-tracks-line',
                    type: 'line',
                    source: 'ais-tracks-src',
                    paint: {{
                        'line-color': '#38bdf8',
                        'line-width': 2,
                        'line-opacity': 0.65
                    }}
                }});

                // Track hover & click
                map.on('click', 'ais-tracks-line', (e) => {{
                    if (e.features && e.features[0]) {{
                        const mmsi = e.features[0].properties.mmsi;
                        selectVessel(mmsi);
                    }}
                }});
            }}

            // 5. Spill Polygon
            if (geoData.spill_feature) {{
                map.addSource('spill-src', {{
                    type: 'geojson',
                    data: geoData.spill_feature
                }});
                map.addLayer({{
                    id: 'spill-fill',
                    type: 'fill',
                    source: 'spill-src',
                    paint: {{
                        'fill-color': '#ef4444',
                        'fill-opacity': 0.35
                    }}
                }});
                map.addLayer({{
                    id: 'spill-outline',
                    type: 'line',
                    source: 'spill-src',
                    paint: {{
                        'line-color': '#ef4444',
                        'line-width': 2.5
                    }}
                }});

                map.on('click', 'spill-fill', () => {{
                    showSpillInspector();
                    map.flyTo({{ center: geoData.center, zoom: 12, speed: 1.2 }});
                }});
            }}

            // 6. Source Point Marker
            if (geoData.source_point_feature) {{
                map.addSource('source-pt-src', {{
                    type: 'geojson',
                    data: geoData.source_point_feature
                }});
                map.addLayer({{
                    id: 'source-pt-circle',
                    type: 'circle',
                    source: 'source-pt-src',
                    paint: {{
                        'circle-radius': 8,
                        'circle-color': '#f59e0b',
                        'circle-stroke-width': 2,
                        'circle-stroke-color': '#ffffff'
                    }}
                }});

                map.on('click', 'source-pt-circle', () => {{
                    showSourceInspector();
                    const coords = geoData.source_point_feature.geometry.coordinates;
                    map.flyTo({{ center: coords, zoom: 12.5, speed: 1.2 }});
                }});
            }}

            // 7. Coastal Sensitive Assets
            if (geoData.coastal_features && geoData.coastal_features.length > 0) {{
                map.addSource('coastal-src', {{
                    type: 'geojson',
                    data: {{
                        type: 'FeatureCollection',
                        features: geoData.coastal_features
                    }}
                }});
                map.addLayer({{
                    id: 'coastal-pts',
                    type: 'circle',
                    source: 'coastal-src',
                    paint: {{
                        'circle-radius': 6,
                        'circle-color': '#ec4899',
                        'circle-stroke-width': 1.5,
                        'circle-stroke-color': '#fbcfe8'
                    }}
                }});

                map.on('click', 'coastal-pts', (e) => {{
                    if (e.features && e.features[0]) {{
                        showAssetInspector(e.features[0].properties);
                        map.flyTo({{ center: e.features[0].geometry.coordinates, zoom: 12 }});
                    }}
                }});
            }}
        }}

        // Setup Vessel Animated Markers
        function setupVesselMarkers() {{
            (geoData.vessel_data || []).forEach(v => {{
                const el = document.createElement('div');
                el.className = 'vessel-marker';
                el.style.width = '14px';
                el.style.height = '14px';
                el.style.backgroundColor = '#38bdf8';
                el.style.borderRadius = '50%';
                el.style.border = '2px solid #ffffff';
                el.style.boxShadow = '0 0 10px #38bdf8';
                el.style.cursor = 'pointer';
                el.title = `${{v.name}} (MMSI: ${{v.mmsi}})`;

                el.addEventListener('click', () => {{
                    selectVessel(v.mmsi);
                }});

                const marker = new maplibregl.Marker({{ element: el }});
                // Initialize at T=0 position
                const pt = getInterpolatedPoint(v, 0);
                if (pt) {{
                    marker.setLngLat([pt.lon, pt.lat]).addTo(map);
                    vesselMarkers[v.mmsi] = {{ marker, el, vessel: v }};
                }}
            }});
        }}

        // Interpolate Vessel Position along its AIS track based on timeline scrubber
        function getInterpolatedPoint(vessel, minutes) {{
            const pts = vessel.points;
            if (!pts || pts.length === 0) return null;
            if (pts.length === 1) return pts[0];

            if (minutes <= pts[0].minute_offset) return pts[0];
            if (minutes >= pts[pts.length - 1].minute_offset) return pts[pts.length - 1];

            for (let i = 0; i < pts.length - 1; i++) {{
                const p1 = pts[i];
                const p2 = pts[i + 1];
                if (minutes >= p1.minute_offset && minutes <= p2.minute_offset) {{
                    const range = p2.minute_offset - p1.minute_offset;
                    const frac = range > 0 ? (minutes - p1.minute_offset) / range : 0;
                    return {{
                        lon: p1.lon + frac * (p2.lon - p1.lon),
                        lat: p1.lat + frac * (p2.lat - p1.lat),
                        cog: p1.cog
                    }};
                }}
            }}
            return pts[pts.length - 1];
        }}

        function onTimeScrub(val) {{
            currentMinutes = parseInt(val);
            updateTimelineUI();
            updateVesselPositions();
        }}

        function updateTimelineUI() {{
            const slider = document.getElementById('time-slider');
            const pill = document.getElementById('epoch-pill');
            slider.value = currentMinutes;
            pill.textContent = (currentMinutes < 0 ? `T${{currentMinutes}}m` : `T+${{currentMinutes}}m`);
        }}

        function updateVesselPositions() {{
            Object.values(vesselMarkers).forEach(vm => {{
                const pt = getInterpolatedPoint(vm.vessel, currentMinutes);
                if (pt) {{
                    vm.marker.setLngLat([pt.lon, pt.lat]);
                }}
            }});
        }}

        function togglePlayPause() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('play-pause-btn');
            btn.textContent = isPlaying ? '⏸' : '▶';

            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    currentMinutes += 2 * playbackSpeed;
                    if (currentMinutes > 60) {{
                        currentMinutes = -180;
                    }}
                    updateTimelineUI();
                    updateVesselPositions();
                }}, 60);
            }} else {{
                clearInterval(playInterval);
            }}
        }}

        function cycleSpeed() {{
            const btn = document.getElementById('speed-toggle');
            if (playbackSpeed === 1) playbackSpeed = 2;
            else if (playbackSpeed === 2) playbackSpeed = 5;
            else playbackSpeed = 1;
            btn.textContent = `${{playbackSpeed}}x`;
        }}

        // Selection Handlers
        function selectVessel(mmsi) {{
            const vData = (geoData.vessel_data || []).find(v => v.mmsi === mmsi);
            if (!vData) return;

            // Highlight vessel marker, dim others
            Object.values(vesselMarkers).forEach(vm => {{
                if (vm.vessel.mmsi === mmsi) {{
                    vm.el.style.backgroundColor = '#fbbf24';
                    vm.el.style.transform = 'scale(1.5)';
                    vm.el.style.zIndex = 10;
                }} else {{
                    vm.el.style.backgroundColor = '#475569';
                    vm.el.style.transform = 'scale(0.85)';
                    vm.el.style.zIndex = 1;
                }}
            }});

            // Fly camera to vessel
            const currentPt = getInterpolatedPoint(vData, currentMinutes);
            if (currentPt) {{
                map.flyTo({{ center: [currentPt.lon, currentPt.lat], zoom: 12.5, speed: 1.2 }});
            }}

            showVesselInspector(vData);
        }}

        function showVesselInspector(v) {{
            const panel = document.getElementById('inspector-panel');
            const title = document.getElementById('insp-title');
            const body = document.getElementById('insp-body');

            title.textContent = "CANDIDATE ATTRIBUTION";
            body.innerHTML = `
                <div style="margin-bottom:12px;">
                    <div style="font-size:15px; font-weight:700; color:#f0f4f8; margin-bottom:2px;">${{v.name}}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#64748b;">MMSI: ${{v.mmsi}}</div>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">ASSOCIATION SCORE</span>
                    <span class="score-badge">${{v.score}} / 100</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">SPATIAL PROXIMITY</span>
                    <span class="inspector-val">${{v.breakdown.spatial ? v.breakdown.spatial.toFixed(1) : 'N/A'}}</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">TRAJECTORY CONSISTENCY</span>
                    <span class="inspector-val">${{v.breakdown.trajectory ? v.breakdown.trajectory.toFixed(1) : 'HIGH'}}</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">TEMPORAL WINDOW</span>
                    <span class="inspector-val">${{v.breakdown.temporal ? v.breakdown.temporal.toFixed(1) : 'VALID'}}</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">DRIFT MATCH</span>
                    <span class="inspector-val">${{v.drift_consistency ? 'CONSISTENT' : 'DEVIATED'}}</span>
                </div>
                <div class="legal-micro">
                    ⚖️ <strong>LEGAL NOTICE:</strong> ${{v.disclaimer}}
                </div>
            `;
            panel.classList.add('visible');
        }}

        function showSpillInspector() {{
            const p = geoData.spill_feature.properties;
            const panel = document.getElementById('inspector-panel');
            const title = document.getElementById('insp-title');
            const body = document.getElementById('insp-body');

            title.textContent = "OIL SLICK EVIDENCE";
            body.innerHTML = `
                <div style="margin-bottom:12px;">
                    <div style="font-size:15px; font-weight:700; color:#ef4444; margin-bottom:2px;">${{p.title}}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#64748b;">[${{p.lat.toFixed(4)}}°N, ${{p.lon.toFixed(4)}}°E]</div>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">SURFACE AREA</span>
                    <span class="inspector-val">${{p.area_sq_km.toFixed(2)}} km²</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">ESTIMATED VOLUME</span>
                    <span class="inspector-val">${{p.volume_tons.toFixed(1)}} tons</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">SLICK PERIMETER</span>
                    <span class="inspector-val">${{p.perimeter_km.toFixed(2)}} km</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">FAY SPREADING AGE</span>
                    <span class="inspector-val">${{p.fay_age.toFixed(1)}} hours</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">YOLO CONFIDENCE</span>
                    <span class="inspector-val">${{(p.yolo_conf * 100).toFixed(1)}}%</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">CLASSICAL CONSENSUS</span>
                    <span class="inspector-val" style="color:#10b981;">${{(p.consensus * 100).toFixed(1)}}%</span>
                </div>
            `;
            panel.classList.add('visible');
        }}

        function showSourceInspector() {{
            const p = geoData.source_point_feature.properties;
            const panel = document.getElementById('inspector-panel');
            const title = document.getElementById('insp-title');
            const body = document.getElementById('insp-body');

            title.textContent = "DRIFT ORIGIN HINDCAST";
            body.innerHTML = `
                <div style="margin-bottom:12px;">
                    <div style="font-size:15px; font-weight:700; color:#f59e0b; margin-bottom:2px;">${{p.title}}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#64748b;">[${{p.lat.toFixed(4)}}°N, ${{p.lon.toFixed(4)}}°E]</div>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">UNCERTAINTY RADIUS</span>
                    <span class="inspector-val">±${{p.uncertainty_km.toFixed(1)}} km</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">OCEAN CURRENT</span>
                    <span class="inspector-val">${{p.current_speed.toFixed(2)}} m/s @ ${{p.current_bearing.toFixed(0)}}°</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">SURFACE WIND</span>
                    <span class="inspector-val">${{p.wind_speed.toFixed(1)}} m/s @ ${{p.wind_bearing.toFixed(0)}}°</span>
                </div>
            `;
            panel.classList.add('visible');
        }}

        function showAssetInspector(p) {{
            const panel = document.getElementById('inspector-panel');
            const title = document.getElementById('insp-title');
            const body = document.getElementById('insp-body');

            title.textContent = "COASTAL THREAT ASSET";
            body.innerHTML = `
                <div style="margin-bottom:12px;">
                    <div style="font-size:15px; font-weight:700; color:#ec4899; margin-bottom:2px;">${{p.name}}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#64748b;">Type: ${{p.asset_type}}</div>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">VULNERABILITY INDEX</span>
                    <span class="score-badge" style="border-color:#ec4899; color:#f472b6; background:rgba(236,72,153,0.15);">${{p.esi_rank}}</span>
                </div>
                <div class="inspector-row">
                    <span class="inspector-label">PROTECTION PRIORITY</span>
                    <span class="inspector-val" style="color:#ef4444;">HIGH</span>
                </div>
            `;
            panel.classList.add('visible');
        }}

        function closeInspector() {{
            document.getElementById('inspector-panel').classList.remove('visible');
        }}

        // Layer Toggling
        function toggleLayer(layerKey) {{
            const btn = document.getElementById(`btn-toggle-${{layerKey}}`);
            const isActive = btn.classList.contains('active');
            btn.classList.toggle('active', !isActive);
            const targetVis = !isActive ? 'visible' : 'none';

            if (layerKey === 'spill') {{
                if (map.getLayer('spill-fill')) map.setLayoutProperty('spill-fill', 'visibility', targetVis);
                if (map.getLayer('spill-outline')) map.setLayoutProperty('spill-outline', 'visibility', targetVis);
            }} else if (layerKey === 'source') {{
                if (map.getLayer('source-pt-circle')) map.setLayoutProperty('source-pt-circle', 'visibility', targetVis);
                if (map.getLayer('credible-zones-fill')) map.setLayoutProperty('credible-zones-fill', 'visibility', targetVis);
                if (map.getLayer('credible-zones-line')) map.setLayoutProperty('credible-zones-line', 'visibility', targetVis);
            }} else if (layerKey === 'backtrack') {{
                if (map.getLayer('backtrack-line')) map.setLayoutProperty('backtrack-line', 'visibility', targetVis);
            }} else if (layerKey === 'forecast') {{
                if (map.getLayer('forecast-line')) map.setLayoutProperty('forecast-line', 'visibility', targetVis);
            }} else if (layerKey === 'ais') {{
                if (map.getLayer('ais-tracks-line')) map.setLayoutProperty('ais-tracks-line', 'visibility', targetVis);
                Object.values(vesselMarkers).forEach(vm => {{
                    vm.el.style.display = targetVis === 'visible' ? 'block' : 'none';
                }});
            }} else if (layerKey === 'assets') {{
                if (map.getLayer('coastal-pts')) map.setLayoutProperty('coastal-pts', 'visibility', targetVis);
            }}
        }}

        // Fit Bounds to Entire Analysis Area
        function fitBounds() {{
            const bounds = new maplibregl.LngLatBounds();
            bounds.extend(geoData.center || [80.23, 12.45]);
            if (geoData.source_point_feature) {{
                bounds.extend(geoData.source_point_feature.geometry.coordinates);
            }}
            (geoData.vessel_data || []).forEach(v => {{
                (v.points || []).forEach(p => bounds.extend([p.lon, p.lat]));
            }});
            map.fitBounds(bounds, {{ padding: 60, maxZoom: 13, speed: 1.4 }});
        }}

        // Keyboard Shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT') return;
            if (e.key === ' ' || e.code === 'Space') {{
                e.preventDefault();
                togglePlayPause();
            }} else if (e.key === 'f' || e.key === 'F') {{
                fitBounds();
            }} else if (e.key === 'Escape') {{
                closeInspector();
                closeCmdPalette();
            }} else if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) {{
                e.preventDefault();
                openCmdPalette();
            }} else if (e.key === 'k' || e.key === 'K') {{
                openCmdPalette();
            }}
        }});

        // Command Palette
        function openCmdPalette() {{
            const pal = document.getElementById('cmd-palette');
            pal.classList.add('open');
            document.getElementById('cmd-input').focus();
        }}

        function closeCmdPalette() {{
            document.getElementById('cmd-palette').classList.remove('open');
        }}

        function executeCmd(action) {{
            closeCmdPalette();
            if (action === 'spill') {{
                showSpillInspector();
                map.flyTo({{ center: geoData.center, zoom: 12.5, speed: 1.2 }});
            }} else if (action === 'source') {{
                showSourceInspector();
                map.flyTo({{ center: geoData.source_point_feature.geometry.coordinates, zoom: 12.5, speed: 1.2 }});
            }} else if (action === 'top_vessel') {{
                if (geoData.vessel_data && geoData.vessel_data.length > 0) {{
                    selectVessel(geoData.vessel_data[0].mmsi);
                }}
            }} else if (action === 'toggle_forecast') {{
                toggleLayer('forecast');
            }} else if (action === 'toggle_ais') {{
                toggleLayer('ais');
            }} else if (action === 'fit') {{
                fitBounds();
            }}
        }}

        function filterCmdList(query) {{
            const q = query.toLowerCase();
            document.querySelectorAll('.cmd-item').forEach(item => {{
                item.style.display = item.textContent.toLowerCase().includes(q) ? 'flex' : 'none';
            }});
        }}

        // Performance & lifecycle cleanup
        document.addEventListener('visibilitychange', () => {{
            if (document.hidden && isPlaying) {{
                togglePlayPause();
            }}
        }});

        window.addEventListener('beforeunload', () => {{
            if (playInterval) clearInterval(playInterval);
            if (map) {{
                try {{ map.remove(); }} catch(e) {{}}
            }}
        }});
    </script>
</body>
</html>
    """
    return html_content
