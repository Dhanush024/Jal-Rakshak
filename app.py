# pyrefly: ignore [missing-import]
"""
Jal-Rakshak — Maritime Oil Spill Intelligence Dashboard
=========================================================
Streamlit-based UI integrating the 11-node LangGraph pipeline.
"""
import streamlit as st
import os
import sys
import subprocess
import time
import json
import numpy as np
from datetime import datetime, timezone, timedelta

# ──────────────────────────────────────────────────────────────
# OpenCV headless setup (production Codespaces hack)
# ──────────────────────────────────────────────────────────────
cv2_target = "/tmp/opencv_headless"

if sys.platform.startswith("linux") and not os.path.exists(os.path.join(cv2_target, "cv2")):
    os.makedirs(cv2_target, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", cv2_target,
         "--no-deps", "opencv-python-headless"],
        check=True, capture_output=True,
    )

if os.path.exists(cv2_target) and cv2_target not in sys.path:
    sys.path.insert(0, cv2_target)

if hasattr(sys, "OpenCV_LOADER"):
    delattr(sys, "OpenCV_LOADER")
for mod in list(sys.modules.keys()):
    if mod == "cv2" or mod.startswith("cv2."):
        del sys.modules[mod]

import cv2
from PIL import Image
import folium
from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium

# ──────────────────────────────────────────────────────────────
# Project imports
# ──────────────────────────────────────────────────────────────
from config.settings import (
    is_demo_mode, DEMO_SPILL_LAT, DEMO_SPILL_LON,
    MAP_BASEMAP, CARTO_API_KEY
)
from pipeline.graph import run_pipeline, compile_pipeline
from geospatial.distance import haversine_km
from demo.scenario import CHENNAI_SCENARIO, get_or_create_demo_sar_patch

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Jal-Rakshak Dashboard", layout="wide", page_icon="🌊")

# ========================= CSS =========================
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 16px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .alert-critical {
        background: linear-gradient(90deg, #491217 0%, #1c0a0c 100%);
        border: 1px solid #f85149;
        border-left: 5px solid #f85149;
        border-radius: 8px;
        padding: 14px 18px;
        color: #ff7b72;
        font-weight: bold;
    }
    .alert-high {
        background: linear-gradient(90deg, #3d2300 0%, #1c1000 100%);
        border: 1px solid #d29922;
        border-left: 5px solid #d29922;
        border-radius: 8px;
        padding: 14px 18px;
        color: #e3b341;
        font-weight: bold;
    }
    .alert-medium {
        background: linear-gradient(90deg, #2d2605 0%, #1a1502 100%);
        border: 1px solid #e3b341;
        border-left: 5px solid #e3b341;
        border-radius: 8px;
        padding: 14px 18px;
        color: #f2cc60;
        font-weight: bold;
    }
    .alert-low {
        background: linear-gradient(90deg, #0d2818 0%, #06150c 100%);
        border: 1px solid #3fb950;
        border-left: 5px solid #3fb950;
        border-radius: 8px;
        padding: 14px 18px;
        color: #56d364;
        font-weight: bold;
    }
    .investigation-header {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .investigation-header h2 {
        color: #58a6ff;
        margin: 0 0 8px 0;
    }
    .investigation-header p {
        color: #8b949e;
        margin: 0;
        font-size: 14px;
    }
    .evidence-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .evidence-card h3 {
        margin: 0 0 12px 0;
        color: #f0f6fc;
        font-size: 18px;
    }
    .evidence-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #21262d;
        font-size: 13px;
    }
    .evidence-row:last-child {
        border-bottom: none;
    }
    .evidence-label {
        color: #8b949e;
    }
    .evidence-value {
        color: #f0f6fc;
        font-weight: 600;
    }
    .match-yes {
        color: #3fb950;
        font-weight: bold;
    }
    .match-no {
        color: #f85149;
        font-weight: bold;
    }
    .correlation-score-big {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        padding: 10px 0 4px 0;
    }
    .correlation-bar-container {
        margin: 8px 0;
    }
    .correlation-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: #c9d1d9;
        margin-bottom: 4px;
    }
    .correlation-bar-track {
        background: #21262d;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .correlation-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }
    .uncertainty-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .data-badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-observed { background:#1f3d2b; color:#3fb950; }
    .badge-inferred { background:#3d2f1f; color:#d29922; }
    .badge-predicted { background:#1f1a3d; color:#a78bfa; }
    .badge-simulated { background:#2d1521; color:#f97583; border: 1px dashed #f97583; }
    .risk-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        border-left: 4px solid;
    }
    .risk-CRITICAL { border-left-color: #FF4B4B; }
    .risk-HIGH     { border-left-color: #ff8c00; }
    .risk-MEDIUM   { border-left-color: #ffc107; }
    .risk-LOW      { border-left-color: #3fb950; }
    .hero-container {
        background: radial-gradient(circle at 20% 30%, #1a2332 0%, #0d1117 70%);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 36px 30px;
        margin-bottom: 24px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Jal-Rakshak: AI Maritime Oil Spill Intelligence")
st.markdown("Automated SAR Analysis | YOLOv8 Segmentation | 11-Node LangGraph Forensic Pipeline")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("🎛️ Control Panel")

    # 1. Mode selector
    mode_selection = st.radio(
        "Operating Mode",
        ["🔶 DEMO MODE (Simulation)", "🟢 LIVE SENSORS (AIS & Ocean)"],
        index=0,
    )
    is_demo = "DEMO" in mode_selection

    # 2. Quick Demo Launch
    st.markdown("### ⚡ Quick-Start Scenario")
    if st.button("🚀 Load Chennai Spill Scenario", type="primary", use_container_width=True):
        demo_patch = get_or_create_demo_sar_patch()
        st.session_state["active_image_path"] = demo_patch
        st.session_state["spill_lat"] = CHENNAI_SCENARIO.spill_lat
        st.session_state["spill_lon"] = CHENNAI_SCENARIO.spill_lon
        st.session_state["auto_run"] = True
        st.rerun()

    st.divider()

    # 3. Geofencing
    st.markdown("### 🛰️ Sector Geofence")
    st.caption("Draw surveillance polygon on Sentinel-1 swath.")

    m = folium.Map(location=[13.0, 80.3], zoom_start=7)
    Fullscreen(position="topright", title="Fullscreen",
               title_cancel="Exit", force_separate_button=True).add_to(m)
    Draw(export=False, position="topleft", draw_options={
        "polyline": False, "poly": True, "circle": False,
        "marker": False, "circlemarker": False, "rectangle": True,
    }).add_to(m)
    map_data = st_folium(m, height=220, width=280)

    if map_data and map_data.get("all_drawings"):
        if len(map_data["all_drawings"]) > 0:
            st.success("✅ Custom Sector Active")

    st.divider()

    # 4. Manual upload
    st.markdown("### 📁 Satellite Imagery")
    uploaded_file = st.file_uploader(
        "Upload SAR imagery (JPG/PNG/GeoTIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )
    if uploaded_file is not None:
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["active_image_path"] = temp_path

    # 4b. Optional Historical AIS Upload
    st.markdown("### 🚢 Historical AIS Archive (Optional)")
    st.caption("Upload real historical AIS telemetry (CSV/JSON) or leave empty for demo.")
    uploaded_ais = st.file_uploader(
        "Upload AIS Telemetry",
        type=["csv", "json", "geojson"],
        key="ais_uploader",
    )
    if uploaded_ais is not None:
        temp_ais_path = "temp_historical_ais.csv"
        with open(temp_ais_path, "wb") as f:
            f.write(uploaded_ais.getbuffer())
        st.session_state["active_ais_path"] = temp_ais_path
        st.success("✅ Real AIS Telemetry Loaded")
    else:
        st.session_state.pop("active_ais_path", None)

    # 5. Spill coordinates
    st.markdown("### 📍 Observation Coordinates")
    default_lat = st.session_state.get("spill_lat", DEMO_SPILL_LAT)
    default_lon = st.session_state.get("spill_lon", DEMO_SPILL_LON)
    spill_lat = st.number_input("Latitude", value=default_lat, format="%.4f")
    spill_lon = st.number_input("Longitude", value=default_lon, format="%.4f")

# Data mode indicator banner
if is_demo:
    st.markdown(
        '<span class="data-badge badge-simulated">DEMO MODE</span> '
        '<span style="color:#a0a0c0; font-size:13px;">Using calibrated simulation data (Bay of Bengal / Chennai scenario)</span>',
        unsafe_allow_html=True,
    )
st.divider()


# ========================= HELPER: Build Investigation Map =========================

def build_investigation_map(state, slider_minutes=0, selected_vessel_mmsi=None):
    """Build the forensic Folium map with pipeline results, candidate vessel tracks, and timeline position."""
    spill_lat = state.get("spill_lat", DEMO_SPILL_LAT)
    spill_lon = state.get("spill_lon", DEMO_SPILL_LON)
    source_lat = state.get("source_lat", spill_lat)
    source_lon = state.get("source_lon", spill_lon)
    uncertainty_km = state.get("source_uncertainty_km", 5.0)

    carto_key = CARTO_API_KEY or os.getenv("CARTO_API_KEY", "")
    use_carto = bool(carto_key) and (MAP_BASEMAP.lower() in ("cartodb_dark", "cartodb dark_matter", "cartodb"))

    if use_carto:
        tiles_url = f"https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png?api_key={carto_key}"
        fmap = folium.Map(
            location=[source_lat, source_lon],
            zoom_start=11,
            tiles=None,
        )
        folium.TileLayer(
            tiles=tiles_url,
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name="CartoDB Dark Matter",
        ).add_to(fmap)
    else:
        # Default: Standard OpenStreetMap — zero API key required, zero broken watermarks
        fmap = folium.Map(
            location=[source_lat, source_lon],
            zoom_start=11,
            tiles="OpenStreetMap",
        )

    # 1. Detected Spill Marker
    folium.CircleMarker(
        location=[spill_lat, spill_lon], radius=14,
        color="#ff4b4b", fill=True, fill_color="#ff4b4b", fill_opacity=0.35,
        tooltip="🛢️ Detected Oil Spill Region",
    ).add_to(fmap)
    folium.Marker(
        location=[spill_lat, spill_lon],
        icon=folium.DivIcon(html='<div style="font-size:22px;">🛢️</div>'),
        tooltip="Oil Spill Detection Point",
    ).add_to(fmap)

    # 2. Source Probability Likelihood Surface & Credible Zones
    source_prob = state.get("source_probability", {})
    credible_zones = source_prob.get("credible_zones", [])
    if credible_zones:
        for cz in reversed(credible_zones):  # Outer to inner for proper SVG layering
            pts = cz.get("polygon_points", [])
            if pts:
                folium.Polygon(
                    locations=pts,
                    color=cz.get("color_hex", "#ff8c00"),
                    weight=1.5,
                    fill=True,
                    fill_color=cz.get("color_hex", "#ff8c00"),
                    fill_opacity=cz.get("fill_opacity", 0.15),
                    tooltip=f"{cz.get('name')}: {cz.get('description')}",
                ).add_to(fmap)
    else:
        folium.Circle(
            location=[source_lat, source_lon],
            radius=uncertainty_km * 1000,
            color="#ff8c00", fill=True, fill_color="#ff8c00", fill_opacity=0.15,
            dash_array="10 6",
            tooltip=f"Estimated Origin Zone (±{uncertainty_km:.1f} km)",
        ).add_to(fmap)

    folium.Marker(
        location=[source_lat, source_lon],
        icon=folium.DivIcon(html='<div style="font-size:12px; color:#ff8c00; font-weight:bold; white-space:nowrap; background:rgba(0,0,0,0.6); padding:2px 4px; border-radius:3px;">▲ ESTIMATED ORIGIN</div>'),
    ).add_to(fmap)

    # 3. Drift path (origin -> spill)
    folium.PolyLine(
        locations=[[source_lat, source_lon], [spill_lat, spill_lon]],
        color="#ff8c00", weight=2, dash_array="8 4", opacity=0.7,
        tooltip="Hindcast drift path",
    ).add_to(fmap)

    # 4. Forecast trajectories
    forecasts = state.get("forecast_results", [])
    forecast_colors = ["#00d4ff", "#a78bfa", "#f97583"]
    for i, fc in enumerate(forecasts):
        dest_lat = fc.get("destination_lat")
        dest_lon = fc.get("destination_lon")
        if dest_lat and dest_lon:
            color = forecast_colors[i % len(forecast_colors)]
            folium.PolyLine(
                locations=[[spill_lat, spill_lon], [dest_lat, dest_lon]],
                color=color, weight=2, dash_array="4 8", opacity=0.6,
                tooltip=f"Forward Forecast: {fc.get('direction', '')}",
            ).add_to(fmap)
            folium.CircleMarker(
                location=[dest_lat, dest_lon], radius=6,
                color=color, fill=True, fill_opacity=0.5,
                tooltip=f"Predicted position ({fc.get('direction', '')})",
            ).add_to(fmap)

    # 5. Sensitive Coastal Areas
    for sa in CHENNAI_SCENARIO.sensitive_areas:
        icon_symbol = "🐦" if sa.get("type") == "wildlife" else ("⚓" if sa.get("type") == "port" else "🎣")
        folium.Marker(
            location=[sa["lat"], sa["lon"]],
            icon=folium.DivIcon(html=f'<div style="font-size:16px;">{icon_symbol}</div>'),
            tooltip=f"{sa['name']} ({sa.get('type', 'sensitive')})",
        ).add_to(fmap)

    # 6. Candidate Vessels & AIS Tracks
    tracks_data = state.get("ais_tracks", {})
    candidates = state.get("candidate_scores", [])
    ranking_map = {c["mmsi"]: c for c in candidates}

    # Reference time
    ts = state.get("detection_timestamp")
    base_time = datetime.fromisoformat(ts) if ts else datetime(2026, 9, 14, 15, 30, tzinfo=timezone.utc)
    slider_time = base_time + timedelta(minutes=slider_minutes)

    vessel_color_palette = ["#ff4b4b", "#00d4ff", "#ffc107", "#a78bfa", "#3fb950"]

    for idx, (mmsi, recs) in enumerate(tracks_data.items()):
        if not recs:
            continue
        v_color = vessel_color_palette[idx % len(vessel_color_palette)]
        is_selected = (mmsi == selected_vessel_mmsi)
        rank_info = ranking_map.get(mmsi, {})
        score = rank_info.get("score", 0)
        vessel_name = recs[0].get("name", mmsi)

        # Plot vessel track polyline
        track_pts = [[r["lat"], r["lon"]] for r in recs]
        folium.PolyLine(
            locations=track_pts,
            color=v_color,
            weight=4 if is_selected else 2,
            opacity=0.85 if is_selected else 0.45,
            dash_array="6 3" if not is_selected else None,
            tooltip=f"{vessel_name} (Association: {score:.0f}/100)",
        ).add_to(fmap)

        # Interpolate vessel position at slider_time
        v_pos = recs[0]
        min_dt = float("inf")
        for r in recs:
            r_time = datetime.fromisoformat(r["timestamp"])
            dt = abs((r_time - slider_time).total_seconds())
            if dt < min_dt:
                min_dt = dt
                v_pos = r

        # Vessel marker
        marker_size = 11 if is_selected else 7
        folium.CircleMarker(
            location=[v_pos["lat"], v_pos["lon"]],
            radius=marker_size,
            color=v_color,
            fill=True,
            fill_color=v_color,
            fill_opacity=1.0 if is_selected else 0.7,
            tooltip=f"🚢 {vessel_name} | {v_pos.get('speed_knots', 0)} kn | Score: {score:.0f}",
            popup=f"<b>{vessel_name}</b><br>MMSI: {mmsi}<br>Speed: {v_pos.get('speed_knots', 0)} kn<br>Heading: {v_pos.get('heading', 0)}°<br>Score: {score:.0f}/100",
        ).add_to(fmap)

        # If selected, draw connecting line to estimated origin zone
        if is_selected:
            folium.PolyLine(
                locations=[[v_pos["lat"], v_pos["lon"]], [source_lat, source_lon]],
                color="#3fb950",
                weight=2,
                dash_array="5 5",
                opacity=0.8,
                tooltip=f"Distance to origin: {haversine_km(v_pos['lat'], v_pos['lon'], source_lat, source_lon):.1f} km",
            ).add_to(fmap)

    return fmap


def render_data_badge(classification):
    """Render a colored badge for data classification."""
    badge_class = f"badge-{classification.lower()}"
    return f'<span class="data-badge {badge_class}">{classification}</span>'


# ========================= MAIN FLOW =========================

active_image = st.session_state.get("active_image_path")

if active_image and os.path.exists(active_image):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Raw Satellite Imagery")
        image = Image.open(active_image)
        st.image(image, use_container_width=True)

        run_btn = st.button("🚀 Run 11-Node Pipeline", type="primary", use_container_width=True)
        auto_run = st.session_state.pop("auto_run", False)

        if run_btn or auto_run:
            with st.spinner("Executing 11-node LangGraph intelligence pipeline..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                stages = [
                    "SAR Preprocessing", "YOLO Detection", "Classical Validation",
                    "Spill Characterization", "Ocean Hindcast", "AIS Correlation",
                    "Vessel Attribution", "Drift Forecast", "Risk Assessment",
                    "Alert Generation", "Report Generation",
                ]

                for i, stage in enumerate(stages):
                    status_text.text(f"⚙️ Stage {i+1}/11: {stage}...")
                    progress_bar.progress((i + 1) / len(stages))
                    time.sleep(0.15)

                try:
                    ais_file_to_pass = st.session_state.get("active_ais_path")
                    final_state = run_pipeline(
                        image_path=active_image,
                        spill_lat=spill_lat,
                        spill_lon=spill_lon,
                        app_mode="demo" if is_demo else "live",
                        ais_file_path=ais_file_to_pass,
                    )
                    st.session_state["pipeline_result"] = final_state
                    progress_bar.progress(1.0)
                    status_text.text("✅ Pipeline execution successful!")
                    st.success("All 11 Nodes Completed Successfully!")
                except Exception as e:
                    st.error(f"Pipeline Execution Failed: {e}")
                    st.info("Check model weights in best.pt and python dependencies")

    # ---- Render results if pipeline has run ----
    final_state = st.session_state.get("pipeline_result")
    if final_state:
        # ──── Detection Results ────
        with col1:
            st.subheader("Processed Analysis (Instance Segmentation)")

            if final_state.get("spill_detected"):
                img_cv = cv2.imread(active_image)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                overlay = np.zeros_like(img_cv)

                all_coords = final_state.get("all_spill_coords")
                if not all_coords and final_state.get("spill_coords"):
                    all_coords = [final_state["spill_coords"]]

                total_pts = 0
                if all_coords:
                    for poly in all_coords:
                        if poly and len(poly) >= 3:
                            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                            # Mask overlay in crimson
                            cv2.fillPoly(overlay, [pts], (255, 30, 30))
                            # Clean boundary contour in gold
                            cv2.polylines(img_cv, [pts], isClosed=True, color=(255, 230, 50), thickness=2)
                            total_pts += len(poly)

                    cv2.addWeighted(overlay, 0.40, img_cv, 0.60, 0, img_cv)
                    st.image(img_cv, use_container_width=True,
                             caption=f"YOLOv8 Segmentation — {len(all_coords)} valid marine slick(s) ({total_pts} boundary points)")
                else:
                    st.info("No anomalies detected in this sector.")
            else:
                img_cv = cv2.imread(active_image)
                if img_cv is not None:
                    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True, caption="SAR Analysis — No Valid Marine Slicks Detected")
                st.info("No valid marine oil spill anomalies detected in this sector.")
                # Show rejection rationale if any detection was filtered
                det_res = final_state.get("detection_result", {})
                all_dets = det_res.get("all_detections", [])
                rejections = [d for d in all_dets if not d.get("is_valid_marine", True)]
                if rejections:
                    for rej in rejections:
                        st.caption(f"ℹ️ Artifact #{rej.get('detection_id')}: {rej.get('rejection_reason', 'Terrestrial false positive rejected.')}")

        # ──── Intelligence Panel ────
        with col2:
            st.subheader("Actionable Intelligence")

            if final_state.get("spill_detected"):
                # Risk level banner
                risk_level = final_state.get("risk_level", "UNKNOWN")
                alert_css = f"alert-{risk_level.lower()}" if risk_level != "UNKNOWN" else "alert-medium"
                st.markdown(f'<div class="{alert_css}">⚠️ RISK LEVEL: {risk_level}</div>',
                           unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # Characterization metrics
                char = final_state.get("characterization", {})
                area = char.get("area_sq_km", final_state.get("spill_area_sq_km", 0))
                vol = char.get("estimated_volume_tons")
                perimeter = char.get("perimeter_km", 0)

                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Estimated Surface Area</div>
                    <div class="metric-value">{area:.4f} km²</div>
                </div>
                ''', unsafe_allow_html=True)

                if vol is not None:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Estimated Volume (empirical)</div>
                        <div class="metric-value">{vol:.2f} Metric Tons</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.caption("⚠️ Volume is an empirical estimate (0.5mm thickness assumption)")

                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Perimeter</div>
                    <div class="metric-value">{perimeter:.3f} km</div>
                </div>
                ''', unsafe_allow_html=True)

                # Spill Age Estimation
                age_data = final_state.get("age_estimation", {})
                if age_data and age_data.get("status") == "ESTIMATED":
                    age_rng = age_data.get("estimated_age_range_hours")
                    age_str = f"{age_rng[0]:.1f} – {age_rng[1]:.1f} hrs" if age_rng else f"{age_data.get('best_estimate_hours', 'N/A')} hrs"
                    fay_reg = age_data.get("fay_regime", "N/A").replace("_", " ").title()
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Estimated Spill Age (Weathering)</div>
                        <div class="metric-value" style="color: #38bdf8;">{age_str}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:4px;">Regime: {fay_reg} (Conf: {age_data.get('confidence', 0):.0%})</div>
                    </div>
                    ''', unsafe_allow_html=True)
                elif age_data and age_data.get("status") == "UNKNOWN":
                    st.caption("ℹ️ Spill Age: UNKNOWN (Insufficient morphological / temporal resolution)")

                # Validation
                val = final_state.get("validation_result", {})
                if val:
                    val_conf = val.get("validation_confidence", 0)
                    st.markdown(f"**Validation Confidence:** {val_conf:.1%}")
                    indicators = val.get("look_alike_indicators", [])
                    for ind in indicators:
                        st.warning(ind)

                # AIS data mode
                ais_mode = final_state.get("ais_data_mode", "UNKNOWN")
                st.markdown(f"**AIS Data Mode:** {render_data_badge(ais_mode)}", unsafe_allow_html=True)

            else:
                st.success("✅ No spill detected — sector clear")

        # ──────────────────────────────────────────────────────
        # INVESTIGATION MODE
        # ──────────────────────────────────────────────────────
        if final_state.get("spill_detected") and final_state.get("hindcast_done"):
            st.divider()
            st.markdown("""
            <div class="investigation-header">
                <h2>🔍 INVESTIGATION MODE — Forensic Reconstruction</h2>
                <p>Backward trajectory integration (Euler hindcast) fused with AIS vessel tracking to identify candidate vessels.</p>
            </div>
            """, unsafe_allow_html=True)

            hindcast = final_state.get("hindcast_result", {})
            candidates = final_state.get("candidate_scores", [])

            # ── Trace source trigger ──
            if "trace_started" not in st.session_state:
                st.session_state.trace_started = False

            if st.button("← TRACE SOURCE (AUTOMATED WORKFLOW)", type="primary", use_container_width=True, key="trace_btn"):
                st.session_state.trace_started = True

            if st.session_state.trace_started:
                top_vessel = candidates[0] if candidates else None
                trace_steps = [
                    ("🛢️", "Spill detected in SAR swath", f"Location: {final_state.get('spill_lat', 0):.4f}°N, {final_state.get('spill_lon', 0):.4f}°E"),
                    ("🌊", "Ocean current backward drift (Euler hindcast)",
                     f"Current: {hindcast.get('current_speed_ms', 0)} m/s @ {hindcast.get('current_bearing_deg', 0)}°"),
                    ("📍", f"Estimated origin zone at {hindcast.get('origin_time', 'N/A')[:16]}",
                     f"Position: {hindcast.get('origin_lat', 0):.4f}°N, {hindcast.get('origin_lon', 0):.4f}°E (±{final_state.get('source_uncertainty_km', 0):.1f} km)"),
                    ("📡", "AIS spatial & temporal filtering",
                     f"Candidate vessels within window: {len(candidates)}"),
                    ("🎯", "Trajectory intersection & drift alignment scoring", "Computing multi-factor association scores..."),
                    ("⚓", f"Primary Candidate: {top_vessel['name']}" if top_vessel else "No candidate vessels",
                     f"Association score: {top_vessel['score']:.0f}/100" if top_vessel else ""),
                ]

                with st.status("🔎 Executing forensic reconstruction...", expanded=False) as status:
                    for icon, title, detail in trace_steps:
                        st.markdown(f"**{icon} {title}**")
                        if detail:
                            st.caption(detail)
                    status.update(label="✅ Source trace complete", state="complete", expanded=False)

                # Uncertainty badges
                st.markdown(
                    f'<span class="uncertainty-badge">📐 Spatial uncertainty: '
                    f'±{final_state.get("source_uncertainty_km", 0):.1f} km</span>'
                    f'<span class="uncertainty-badge">🔧 Method: Euler integration (constant field)</span>'
                    f'<span class="uncertainty-badge">📊 Data: {render_data_badge("INFERRED")}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                # ── Timeline Scrubber ──
                st.markdown("#### ⏱️ Forensic Incident Timeline Scrubbing")
                slider_minutes = st.slider(
                    "Scrub Timeline (minutes relative to detection)",
                    min_value=-180,
                    max_value=60,
                    value=0,
                    step=5,
                    format="%d min",
                    help="Scrub through the incident window to observe vessel movements relative to the spill origin zone.",
                )
                ts_str = final_state.get("detection_timestamp", "2026-09-14T15:30:00+00:00")
                try:
                    cur_time = datetime.fromisoformat(ts_str) + timedelta(minutes=slider_minutes)
                    st.caption(f"🕒 Forensic Simulation Time: **{cur_time.strftime('%Y-%m-%d %H:%M UTC')}** ({slider_minutes:+d} min relative to detection)")
                except Exception:
                    pass

                # ── Map + Evidence ──
                map_col, evidence_col = st.columns([3, 2])

                # Vessel selector
                selected_mmsi = None
                if candidates:
                    candidate_options = [c["name"] for c in candidates]
                    selected_name = evidence_col.selectbox("🎯 Focus on Candidate Vessel:", candidate_options, index=0)
                    sel = next((c for c in candidates if c["name"] == selected_name), candidates[0])
                    selected_mmsi = sel["mmsi"]
                else:
                    sel = None

                with map_col:
                    fmap = build_investigation_map(
                        final_state,
                        slider_minutes=slider_minutes,
                        selected_vessel_mmsi=selected_mmsi,
                    )
                    st_folium(fmap, height=540, use_container_width=True, key="investigation_map")

                with evidence_col:
                    # Candidate vessel scores
                    st.markdown("#### Candidate Vessel Scores")
                    for r in candidates[:5]:
                        score = r.get("score", 0)
                        if score >= 75:
                            bar_color = "#ff4b4b"
                        elif score >= 50:
                            bar_color = "#ff8c00"
                        elif score >= 25:
                            bar_color = "#ffc107"
                        else:
                            bar_color = "#484f58"

                        st.markdown(f"""
                        <div class="correlation-bar-container">
                            <div class="correlation-bar-label">
                                <span>{r.get('name', r.get('mmsi', '?'))}</span>
                                <span>{score:.0f}/100</span>
                            </div>
                            <div class="correlation-bar-track">
                                <div class="correlation-bar-fill" style="width: {score}%; background-color: {bar_color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    # Evidence card for selected candidate
                    if sel:
                        breakdown = sel.get("breakdown", {})

                        traj_icon = '<span class="match-yes">✓</span>' if sel.get("trajectory_match") else '<span class="match-no">✗</span>'
                        time_icon = '<span class="match-yes">✓</span>' if sel.get("time_match") else '<span class="match-no">✗</span>'
                        spat_icon = '<span class="match-yes">✓</span>' if sel.get("spatial_match") else '<span class="match-no">✗</span>'
                        drift_icon = '<span class="match-yes">✓</span>' if sel.get("drift_consistency") else '<span class="match-no">✗</span>'

                        score_val = sel.get("score", 0)
                        score_color = "#ff4b4b" if score_val >= 70 else ("#ff8c00" if score_val >= 40 else "#ffc107")

                        st.markdown(f"""
                        <div class="evidence-card">
                            <h3>{sel.get('name', sel.get('mmsi'))}</h3>
                            <div class="evidence-row"><span class="evidence-label">MMSI</span><span class="evidence-value">{sel.get('mmsi')}</span></div>
                            <div class="evidence-row"><span class="evidence-label">Spatial Score</span><span class="evidence-value">{breakdown.get('spatial', 0):.0f}/100</span></div>
                            <div class="evidence-row"><span class="evidence-label">Temporal Score</span><span class="evidence-value">{breakdown.get('temporal', 0):.0f}/100</span></div>
                            <div class="evidence-row"><span class="evidence-label">Trajectory Score</span><span class="evidence-value">{breakdown.get('trajectory', 0):.0f}/100</span></div>
                            <div class="evidence-row"><span class="evidence-label">Drift Consistency</span><span class="evidence-value">{breakdown.get('drift_consistency', 0):.0f}/100</span></div>
                            <hr style="border-color: #21262d; margin: 12px 0;">
                            <div class="evidence-row"><span class="evidence-label">TRAJECTORY MATCH</span>{traj_icon}</div>
                            <div class="evidence-row"><span class="evidence-label">TIME MATCH</span>{time_icon}</div>
                            <div class="evidence-row"><span class="evidence-label">SPATIAL MATCH</span>{spat_icon}</div>
                            <div class="evidence-row"><span class="evidence-label">DRIFT CONSISTENCY</span>{drift_icon}</div>
                            <hr style="border-color: #21262d; margin: 12px 0;">
                            <div class="correlation-score-big" style="color: {score_color};">
                                {score_val:.0f}
                            </div>
                            <div style="text-align:center; color:#8b949e; font-size:12px; letter-spacing:1px;">ASSOCIATION SCORE</div>
                        </div>
                        """, unsafe_allow_html=True)

                        evidence = sel.get("evidence_summary", [])
                        if evidence:
                            st.markdown("**Evidence:**")
                            for e in evidence:
                                st.markdown(f"  • {e}")

                        notes = sel.get("uncertainty_notes", [])
                        if notes:
                            st.markdown("**Uncertainty & Caveats:**")
                            for n in notes:
                                st.caption(f"⚠️ {n}")

                # ──────────────────────────────────────────────────
                # Risk Assessment Panel
                # ──────────────────────────────────────────────────
                if final_state.get("risk_done"):
                    st.divider()
                    risk = final_state.get("risk_assessment", {})
                    risk_level = risk.get("level", "UNKNOWN")

                    st.markdown(f"""
                    <div class="risk-section risk-{risk_level}">
                        <h3 style="color: white; margin-top: 0;">⚡ Risk Assessment — {risk_level}</h3>
                        <p style="color: #a0a0c0;">Overall Score: {risk.get('overall_score', 0):.0f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)

                    factors = risk.get("factors", [])
                    if factors:
                        fcols = st.columns(min(len(factors), 4))
                        for i, f in enumerate(factors[:4]):
                            with fcols[i]:
                                st.metric(f["name"], f"{f['score']:.0f}", delta=None)
                                st.caption(f["description"][:60])

                    recs = risk.get("recommendations", [])
                    if recs:
                        st.markdown("**📋 Tactical Recommendations:**")
                        for r in recs:
                            st.markdown(f"  ✅ {r}")

                # ──────────────────────────────────────────────────
                # Forecast Panel
                # ──────────────────────────────────────────────────
                if final_state.get("forecast_done"):
                    st.divider()
                    st.markdown(f"### 🌊 Forward Drift Forecast {render_data_badge('PREDICTED')}", unsafe_allow_html=True)

                    forecasts = final_state.get("forecast_results", [])
                    if forecasts:
                        fcols = st.columns(len(forecasts))
                        for i, fc in enumerate(forecasts):
                            with fcols[i]:
                                st.markdown(f"**{fc.get('direction', '').replace('_', ' ').title()}**")
                                st.metric("Distance", f"{fc.get('total_distance_km', 0):.1f} km")
                                st.metric("Uncertainty", f"±{fc.get('spatial_uncertainty_km', 0):.1f} km")
                                dest = fc.get("destination_lat", 0)
                                st.caption(f"→ {dest:.4f}°N, {fc.get('destination_lon', 0):.4f}°E")

                # ──────────────────────────────────────────────────
                # Coastal Impact & Shoreline Vulnerability Panel
                # ──────────────────────────────────────────────────
                if final_state.get("coastal_done"):
                    st.divider()
                    st.markdown(f"### 🏖️ Coastal Impact & Shoreline Threat Assessment {render_data_badge('PREDICTED')}", unsafe_allow_html=True)
                    coastal = final_state.get("coastal_impact", {})

                    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
                    with c_col1:
                        st.metric("Shoreline Distance", f"{coastal.get('shortest_distance_to_coast_km', 0):.1f} km")
                        st.caption(f"Nearest: {coastal.get('nearest_shoreline_point', {}).get('name', 'N/A')}")
                    with c_col2:
                        eta_val = coastal.get("eta_to_coast_hours")
                        st.metric("Estimated Landfall ETA", f"{eta_val:.1f} hrs" if eta_val else "No Landfall")
                        unc = coastal.get("eta_uncertainty_range_hours")
                        if unc:
                            st.caption(f"Window: {unc[0]:.1f} - {unc[1]:.1f} hrs")
                    with c_col3:
                        st.metric("Vulnerability Score", f"{coastal.get('coastal_vulnerability_score', 0):.0f}/100")
                        st.caption(f"Tier: {coastal.get('risk_tier', 'LOW')}")
                    with c_col4:
                        st.metric("Threatened Assets", f"{coastal.get('threatened_assets_count', 0)}")
                        st.caption("Ecological & Infrastructure")

                    # Threatened assets table
                    threatened = coastal.get("threatened_assets", [])
                    if threatened:
                        st.markdown("**🛡️ High-Priority Protected Assets in Impact Zone:**")
                        for t in threatened:
                            t_level = t.get("threat_level", "MONITOR")
                            t_color = "#ff4b4b" if t_level == "IMMINENT" else ("#ff8c00" if t_level == "HIGH_RISK" else "#ffc107")
                            st.markdown(f"""
                            <div style="background:#161b22; border-left:4px solid {t_color}; padding:10px 14px; border-radius:6px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:white;">{t['name']} ({t['category'].upper()})</strong>
                                    <span style="color:{t_color}; font-weight:700; font-size:12px;">{t_level} • ESI {t['esi']}/10</span>
                                </div>
                                <div style="font-size:12px; color:#8b949e; margin-top:3px;">
                                    Distance: {t['distance_from_spill_km']:.1f} km | Corridor: {t['distance_from_corridor_km']:.1f} km | Authority: {t['contact_authority']}
                                </div>
                                <div style="font-size:12px; color:#3fb950; margin-top:4px;">
                                    Tactical Strategy: {t['recommended_strategy']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Booming & Dispersant guidelines
                    crecs = coastal.get("containment_recommendations", [])
                    dres = coastal.get("dispersant_restrictions", [])
                    if crecs or dres:
                        st.markdown("**🎯 Environmental Countermeasure Rules:**")
                        for r in crecs:
                            st.markdown(f"  • 🛡️ {r}")
                        for d in dres:
                            st.markdown(f"  • 🚫 **{d}**")

                # ──────────────────────────────────────────────────
                # Alerts Panel
                # ──────────────────────────────────────────────────
                if final_state.get("alerts_done"):
                    st.divider()
                    alert_log = final_state.get("alert_log", {})
                    alerts = alert_log.get("alerts", [])
                    st.markdown(f"### 🔔 Early Warning Alerts Generated ({len(alerts)})")

                    for a in alerts:
                        sim_tag = " [SIMULATED]" if a.get("is_simulated") else ""
                        st.info(f"**{a.get('title', 'Alert')}{sim_tag}** — {a.get('message', '')}")

                # ──────────────────────────────────────────────────
                # Incident Report
                # ──────────────────────────────────────────────────
                if final_state.get("report_done"):
                    st.divider()
                    st.markdown("### 📄 Investigation Incident Report")
                    report = final_state.get("incident_report", {})

                    col_rep1, col_rep2, col_rep3 = st.columns([2, 1, 1])
                    with col_rep1:
                        st.markdown(f"**Incident ID:** `{report.get('incident_id', 'N/A')}`")
                        st.markdown(f"**Classification:** {render_data_badge(report.get('classification', 'CONFIDENTIAL'))}", unsafe_allow_html=True)
                    with col_rep2:
                        report_json_str = json.dumps(report, indent=2, default=str)
                        st.download_button(
                            "📥 Report JSON",
                            data=report_json_str,
                            file_name=f"incident_report_{report.get('incident_id', 'jal_rakshak')}.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                    with col_rep3:
                        import tempfile
                        from reporting.pdf import generate_pdf_report
                        pdf_tmp = os.path.join(tempfile.gettempdir(), f"{report.get('incident_id', 'JR-REPORT')}.pdf")
                        try:
                            generate_pdf_report(report, pdf_tmp)
                            with open(pdf_tmp, "rb") as f_pdf:
                                pdf_bytes = f_pdf.read()
                            st.download_button(
                                "📄 Official PDF",
                                data=pdf_bytes,
                                file_name=f"{report.get('incident_id', 'JR-REPORT')}_Dossier.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.caption(f"PDF export notice: {e}")

                    with st.expander(f"📑 View Incident Report Sections", expanded=False):
                        for section in report.get("sections", []):
                            classification = section.get("data_classification", "UNKNOWN")
                            st.markdown(
                                f"#### {section['title']} {render_data_badge(classification)}",
                                unsafe_allow_html=True,
                            )
                            st.json(section["content"])

                        limitations = report.get("limitations", [])
                        if limitations:
                            st.markdown("**⚠️ Known Limitations:**")
                            for lim in limitations:
                                st.caption(f"• {lim}")

                # Forensic disclaimer
                st.markdown("---")
                st.caption(
                    "⚖️ **Disclaimer:** This platform identifies vessels whose AIS trajectories are statistically "
                    "correlated with the estimated spill origin. Association scores reflect spatial, temporal, and "
                    "trajectory evidence only. They do **NOT** constitute proof of liability or legal attribution. "
                    "This is an operational decision-support tool, not a legal adjudication system."
                )

else:
    # ── Welcome / Hero Section ──
    st.markdown("""
    <div class="hero-container">
        <h1 style="color: #58a6ff; font-size: 32px; margin-bottom: 8px;">Jal-Rakshak: AI Maritime Intelligence Platform</h1>
        <p style="color: #c9d1d9; font-size: 16px; max-width: 750px; margin: 0 auto 24px auto;">
            End-to-end maritime oil spill detection, Euler ocean-current hindcast, progressive AIS vessel filtering,
            explainable candidate attribution, and forward drift early warning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    hcol1, hcol2, hcol3 = st.columns([1, 2, 1])
    with hcol2:
        if st.button("🚀 Launch Chennai Coast Demonstration Scenario", type="primary", use_container_width=True):
            demo_patch = get_or_create_demo_sar_patch()
            st.session_state["active_image_path"] = demo_patch
            st.session_state["spill_lat"] = CHENNAI_SCENARIO.spill_lat
            st.session_state["spill_lon"] = CHENNAI_SCENARIO.spill_lon
            st.session_state["auto_run"] = True
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">1. Detection & Validation</div>
            <div style="color:#f0f6fc; font-size:14px; margin-top:8px;">
                Sentinel-1 SAR ingestion, YOLOv8 segmentation, and Lee speckle filtering cross-validation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with pcol2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">2. Forensic Hindcast</div>
            <div style="color:#f0f6fc; font-size:14px; margin-top:8px;">
                Backward Euler integration with ocean current & wind drift vectors with growing spatial uncertainty bounds.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with pcol3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">3. AIS Vessel Attribution</div>
            <div style="color:#f0f6fc; font-size:14px; margin-top:8px;">
                Multi-stage spatial/temporal filtering and explainable candidate vessel association scores.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with pcol4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">4. Early Warning & Reports</div>
            <div style="color:#f0f6fc; font-size:14px; margin-top:8px;">
                Forward drift trajectory forecasting, multi-factor risk engine, simulated alerts, and incident reports.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.info("💡 **Getting Started:** Click **Launch Chennai Coast Demonstration Scenario** above, or upload a Sentinel-1 SAR image in the sidebar.")