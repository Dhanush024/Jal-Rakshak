# pyrefly: ignore [missing-import]
import streamlit as st
import os, sys, subprocess, time
from datetime import datetime, timezone, timedelta

# 1. Target directory in /tmp for clean headless OpenCV
cv2_target = "/tmp/opencv_headless"

# 2. Unpack headless OpenCV to /tmp BEFORE any cv2 import is ever attempted
if not os.path.exists(os.path.join(cv2_target, "cv2")):
    os.makedirs(cv2_target, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--target", cv2_target,
            "--no-deps",
            "opencv-python-headless"
        ],
        check=True
    )

# 3. Give /tmp top priority in the Python search path
if cv2_target not in sys.path:
    sys.path.insert(0, cv2_target)

# 4. Clear the internal OpenCV recursion guard and purge cached modules
if hasattr(sys, "OpenCV_LOADER"):
    delattr(sys, "OpenCV_LOADER")
for mod in list(sys.modules.keys()):
    if mod == "cv2" or mod.startswith("cv2."):
        del sys.modules[mod]

# 5. Clean import from /tmp (no GUI/libGL dependencies needed)
import cv2

import numpy as np
from PIL import Image
import folium
from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium
from pipeline import app, GraphState
from ais_engine import (
    get_vessel_at_time,
    get_breadcrumb_trail,
    generate_simulated_ais_tracks,
    AISRecord,
)

st.set_page_config(page_title="Jal-Rakshak Dashboard", layout="wide")

# ========================= CSS =========================
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    .metric-label {
        font-size: 14px;
        color: #A0A0A0;
    }
    .alert-critical {
        background-color: rgba(255, 75, 75, 0.2);
        border: 1px solid #FF4B4B;
        padding: 15px;
        border-radius: 5px;
        color: #FF4B4B;
        font-weight: bold;
    }
    /* Investigation Mode styles */
    .investigation-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 20px 28px;
        border-radius: 12px;
        border-left: 5px solid #00d4ff;
        margin-bottom: 20px;
    }
    .investigation-header h2 {
        color: #00d4ff;
        margin: 0 0 6px 0;
        font-size: 22px;
    }
    .investigation-header p {
        color: #a0a0c0;
        margin: 0;
        font-size: 14px;
    }
    .evidence-card {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .evidence-card h3 {
        color: #58a6ff;
        margin: 0 0 14px 0;
        font-size: 18px;
        letter-spacing: 1.5px;
    }
    .evidence-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 1px solid #21262d;
        font-size: 14px;
    }
    .evidence-label { color: #8b949e; }
    .evidence-value { color: #e6edf3; font-weight: 600; }
    .match-yes { color: #3fb950; font-weight: bold; font-size: 16px; }
    .match-no  { color: #f85149; font-weight: bold; font-size: 16px; }
    .correlation-score-big {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        padding: 10px 0;
    }
    .correlation-bar-container {
        background-color: #161b22;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .correlation-bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .correlation-bar-label span:first-child { color: #c9d1d9; font-weight: 600; font-size: 14px; }
    .correlation-bar-label span:last-child  { color: #8b949e; font-size: 13px; }
    .correlation-bar-track {
        background-color: #21262d;
        border-radius: 4px;
        height: 10px;
        overflow: hidden;
    }
    .correlation-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }
    .uncertainty-badge {
        background-color: #1c1c2e;
        border: 1px solid #30365d;
        color: #a0a0c0;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 13px;
        display: inline-block;
        margin-right: 10px;
        margin-bottom: 8px;
    }
    .timeline-label {
        text-align: center;
        font-size: 12px;
        color: #8b949e;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    .trace-step {
        padding: 8px 16px;
        margin: 4px 0;
        border-radius: 6px;
        font-size: 14px;
    }
    .trace-active {
        background-color: rgba(0, 212, 255, 0.1);
        border-left: 3px solid #00d4ff;
        color: #e6edf3;
    }
    .trace-pending {
        color: #484f58;
    }
</style>
""", unsafe_allow_html=True)

st.title("Jal-Rakshak: AI Maritime Surveillance")
st.markdown("Automated SAR Image Processing | YOLOv8 Segmentation | LangGraph Orchestration")
st.divider()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Control Panel")
    
    # 1. THE GEOFENCING FEATURE
    st.markdown("### 🛰️ Sentinel-1 Sector Monitoring")
    st.info("Click the top-right icon to expand fullscreen for precision drawing.")
    
    # Create the map
    m = folium.Map(location=[15.0, 70.0], zoom_start=4)
    
    # Add Fullscreen Button
    Fullscreen(
        position="topright",
        title="Expand to Fullscreen",
        title_cancel="Exit Fullscreen",
        force_separate_button=True
    ).add_to(m)
    
    # Add Drawing Toolbar
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "poly": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        }
    ).add_to(m)
    
    # Render the interactive map
    map_data = st_folium(m, height=260, width=300)
    
    # Capture drawn boundaries
    if map_data and map_data.get("all_drawings"):
        if len(map_data["all_drawings"]) > 0:
            geom = map_data["all_drawings"][-1]["geometry"]
            st.success("✅ Custom Geofence Activated!")
            st.markdown(f"*Monitoring {len(geom['coordinates'][0])} boundary points for new SAR passes.*")
    st.divider()
    
    # 2. Existing Manual Upload (Fallback for testing)
    st.markdown("### Manual Override (Testing)")
    uploaded_file = st.file_uploader("Upload SAR Satellite Imagery (JPG/PNG)", type=["jpg", "jpeg", "png"])
    live_api_toggle = st.toggle("🌐 Enable Live External Network Requests", value=False)


# ========================= HELPER: Build Investigation Map =========================

def build_investigation_map(final_state, slider_minutes, selected_vessel_mmsi=None):
    """Build the forensic Folium map with all investigation layers."""
    drift = final_state["drift_result"]
    origin_lat = drift["origin_lat"]
    origin_lon = drift["origin_lon"]
    spill_lat = final_state["spill_center_lat"]
    spill_lon = final_state["spill_center_lon"]
    uncertainty_km = drift["spatial_uncertainty_km"]

    # Center map on origin
    fmap = folium.Map(
        location=[origin_lat, origin_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )

    # --- Layer 1: Oil spill polygon (current detection position) ---
    folium.CircleMarker(
        location=[spill_lat, spill_lon],
        radius=14,
        color="#ff4b4b",
        fill=True,
        fill_color="#ff4b4b",
        fill_opacity=0.35,
        popup="Oil Spill Detection",
        tooltip="🛢️ Detected Spill",
    ).add_to(fmap)
    folium.Marker(
        location=[spill_lat, spill_lon],
        icon=folium.DivIcon(html='<div style="font-size:20px;">🛢️</div>'),
        tooltip="Oil Spill Detection Point",
    ).add_to(fmap)

    # --- Layer 2: Origin uncertainty ellipse ---
    folium.Circle(
        location=[origin_lat, origin_lon],
        radius=uncertainty_km * 1000,  # meters
        color="#ff8c00",
        fill=True,
        fill_color="#ff8c00",
        fill_opacity=0.12,
        dash_array="10 6",
        tooltip=f"Estimated Origin Zone (±{uncertainty_km} km)",
        popup=f"Origin: {drift['origin_time_str']}<br>Uncertainty: ±{drift['temporal_uncertainty_min']:.0f} min, ±{uncertainty_km} km",
    ).add_to(fmap)
    folium.Marker(
        location=[origin_lat, origin_lon],
        icon=folium.DivIcon(html='<div style="font-size:14px; color:#ff8c00; font-weight:bold; white-space:nowrap;">▲ ESTIMATED ORIGIN</div>'),
    ).add_to(fmap)

    # --- Layer 3: Drift arrow (origin → spill) ---
    folium.PolyLine(
        locations=[[origin_lat, origin_lon], [spill_lat, spill_lon]],
        color="#ff8c00",
        weight=2,
        dash_array="8 4",
        opacity=0.6,
        tooltip="Estimated drift path",
    ).add_to(fmap)

    # --- Layer 4: Vessel positions at slider time ---
    tracks_data = final_state.get("ais_tracks", {})
    rankings = final_state.get("vessel_rankings", [])
    ranking_map = {r["mmsi"]: r for r in rankings}
    t_start = datetime.fromisoformat(final_state["track_time_start"])
    slider_time = t_start + timedelta(minutes=slider_minutes)

    # Reconstruct tracks from serialized data for interpolation
    vessel_colors = {
        rankings[0]["mmsi"]: "#ff4b4b" if rankings else "#888",
    }
    color_cycle = ["#ff4b4b", "#00d4ff", "#ffc107", "#8b949e"]

    for idx, (mmsi, track_dicts) in enumerate(tracks_data.items()):
        # Reconstruct AISRecord objects for interpolation
        track_records = [
            AISRecord(
                mmsi=d["mmsi"], name=d["name"],
                lat=d["lat"], lon=d["lon"],
                heading=d["heading"], speed_knots=d["speed_knots"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
            )
            for d in track_dicts
        ]

        vessel_info = get_vessel_at_time(track_records, slider_time)
        v_color = color_cycle[idx % len(color_cycle)]
        is_selected = (mmsi == selected_vessel_mmsi)
        rank_info = ranking_map.get(mmsi, {})
        score = rank_info.get("score", 0)

        # Vessel marker
        marker_size = 10 if is_selected else 7
        folium.CircleMarker(
            location=[vessel_info["lat"], vessel_info["lon"]],
            radius=marker_size,
            color=v_color,
            fill=True,
            fill_color=v_color,
            fill_opacity=0.9 if is_selected else 0.7,
            tooltip=f"{vessel_info['name']} | {vessel_info['speed_knots']} kn | {score:.0f}%",
            popup=(
                f"<b>{vessel_info['name']}</b><br>"
                f"MMSI: {vessel_info['mmsi']}<br>"
                f"Position: {vessel_info['lat']}°N, {vessel_info['lon']}°E<br>"
                f"Heading: {vessel_info['heading']}°<br>"
                f"Speed: {vessel_info['speed_knots']} kn<br>"
                f"Time: {vessel_info['timestamp_str']}<br>"
                f"Correlation: {score:.1f}%"
            ),
        ).add_to(fmap)

        # Breadcrumb trail for selected vessel
        if is_selected:
            trail = get_breadcrumb_trail(track_records, slider_time, window_minutes=60)
            if len(trail) >= 2:
                trail_coords = [[p["lat"], p["lon"]] for p in trail]
                folium.PolyLine(
                    locations=trail_coords,
                    color=v_color,
                    weight=3,
                    opacity=0.7,
                    tooltip=f"{vessel_info['name']} — last 60 min trail",
                ).add_to(fmap)
                # Fading dots
                for p in trail[::5]:  # every 5th point to avoid clutter
                    folium.CircleMarker(
                        location=[p["lat"], p["lon"]],
                        radius=3,
                        color=v_color,
                        fill=True,
                        fill_opacity=p["opacity"],
                        stroke=False,
                    ).add_to(fmap)

            # Intersection line from vessel to origin
            folium.PolyLine(
                locations=[
                    [vessel_info["lat"], vessel_info["lon"]],
                    [origin_lat, origin_lon],
                ],
                color="#3fb950",
                weight=2,
                dash_array="4 4",
                opacity=0.5,
                tooltip="Distance to origin",
            ).add_to(fmap)

    # Label for vessel name on marker
    folium.Marker(
        location=[vessel_info["lat"], vessel_info["lon"]],
        icon=folium.DivIcon(
            html=f'<div style="font-size:11px; color:{v_color}; font-weight:bold; white-space:nowrap; margin-top:-20px;">{vessel_info["name"]}</div>'
        ),
    ).add_to(fmap)

    return fmap


# ========================= MAIN FLOW =========================

col1, col2 = st.columns([2, 1])

if uploaded_file is not None:
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with col1:
        st.subheader("Raw Satellite Imagery")
        image = Image.open(temp_path)
        st.image(image, use_container_width=True)
        
        if st.button("Run Pipeline", type="primary", use_container_width=True):
            with st.spinner("Initializing LangGraph Multi-Agent Workflow..."):
                
                initial_state = {
                    "image_path": temp_path,
                    "use_live_api": live_api_toggle,
                    "spill_detected": False,
                    "spill_coords": [],
                    "spill_area_sq_km": 0.0,
                    "estimated_spill_volume_tons": 0.0,
                    "is_false_positive": False,
                    "suspect_vessel": {},
                    "incois_data": {},
                    "alert_status": "Pending",
                    # New fields
                    "spill_center_lat": 0.0,
                    "spill_center_lon": 0.0,
                    "detection_time": "",
                    "drift_result": {},
                    "ais_tracks": {},
                    "vessel_rankings": [],
                    "track_time_start": "",
                    "track_time_end": "",
                }
                
                try:
                    final_state = app.invoke(initial_state)
                    st.session_state["pipeline_result"] = final_state
                    st.success("Pipeline Execution Complete!")
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")
                    st.info("Make sure best.pt and pipeline.py are in the same folder as app.py")

    # ---- Render results if pipeline has run ----
    final_state = st.session_state.get("pipeline_result")
    if final_state:
        with col1:
            st.subheader("Processed Analysis (Instance Segmentation)")
            
            if final_state["spill_detected"] and len(final_state["spill_coords"]) > 0:
                img_cv = cv2.imread(temp_path)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                pts = np.array(final_state["spill_coords"], np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                overlay = img_cv.copy()
                cv2.fillPoly(overlay, [pts], (255, 0, 0))
                cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0, img_cv)
                cv2.polylines(img_cv, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                
                st.image(img_cv, use_container_width=True, caption=f"YOLOv8 Polygon Extracted: {len(final_state['spill_coords'])} Boundary Points")
            else:
                st.info("No anomalies detected in this sector.")
                
        with col2:
            st.subheader("Actionable Intelligence")
            
            if final_state["spill_detected"]:
                st.markdown(f'<div class="alert-critical">{final_state["alert_status"]}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Estimated Surface Area</div>
                    <div class="metric-value">{final_state["spill_area_sq_km"]} km²</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Estimated Volume</div>
                    <div class="metric-value">{final_state["estimated_spill_volume_tons"]} Metric Tons</div>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown("### Cross-Reference Logs")
                st.json(final_state["incois_data"])
                st.markdown("### Suspect Vessel ID")
                st.json(final_state["suspect_vessel"])
                
            else:
                st.success(final_state["alert_status"])

        # ========================= INVESTIGATION MODE =========================
        if final_state.get("spill_detected") and final_state.get("drift_result"):
            st.divider()
            st.markdown("""
            <div class="investigation-header">
                <h2>🔍 INVESTIGATION MODE — Forensic Reconstruction</h2>
                <p>AI-driven backward analysis: trace the spill to its source using AIS vessel correlation.</p>
            </div>
            """, unsafe_allow_html=True)

            # --- TRACE SOURCE button ---
            if "trace_started" not in st.session_state:
                st.session_state.trace_started = False

            if st.button("← TRACE SOURCE", type="primary", use_container_width=True, key="trace_btn"):
                st.session_state.trace_started = True

            if st.session_state.trace_started:
                drift = final_state["drift_result"]
                rankings = final_state.get("vessel_rankings", [])
                top_vessel = rankings[0] if rankings else None

                # --- Animated trace sequence ---
                trace_steps = [
                    ("🛢️", "Current spill detected", "Analyzing SAR imagery..."),
                    ("🌊", "AI estimates drift", f"Current: {drift['current_speed_ms']} m/s @ {drift['current_bearing_deg']}° ESE"),
                    ("📍", f"Estimated origin: {drift['origin_time_str']}", f"Position: {drift['origin_lat']}°N, {drift['origin_lon']}°E"),
                    ("📡", "AIS database queried", f"Scanning {len(final_state.get('ais_tracks', {}))} vessel tracks..."),
                    ("🚢", f"{len(rankings)} vessels found in zone", "Running correlation analysis..."),
                    ("🎯", "Trajectory intersection analysis", "Matching vessel paths to origin zone..."),
                    ("⚓", f"Primary match: {top_vessel['name']}" if top_vessel else "No match", ""),
                    ("📊", f"Source Correlation Score: {top_vessel['score']:.0f}%" if top_vessel else "N/A", ""),
                ]

                with st.status("🔎 Tracing spill source...", expanded=True) as status:
                    for icon, title, detail in trace_steps:
                        st.markdown(f"**{icon} {title}**")
                        if detail:
                            st.caption(detail)
                        time.sleep(0.6)
                    status.update(label="✅ Source trace complete", state="complete", expanded=False)

                # --- Uncertainty badges ---
                st.markdown(
                    f'<span class="uncertainty-badge">⏱️ Estimated origin: {drift["origin_time_str"]} ± {drift["temporal_uncertainty_min"]:.0f} min</span>'
                    f'<span class="uncertainty-badge">📐 Spatial uncertainty: ±{drift["spatial_uncertainty_km"]} km</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

                # --- Timeline slider ---
                st.markdown("### ⏳ AIS Timeline Reconstruction")

                t_start = datetime.fromisoformat(final_state["track_time_start"])
                t_end = datetime.fromisoformat(final_state["track_time_end"])
                origin_time = datetime.fromisoformat(drift["origin_time"])
                total_minutes = int((t_end - t_start).total_seconds() / 60)

                # Calculate default slider position = origin time
                origin_offset = int((origin_time - t_start).total_seconds() / 60)
                origin_offset = max(0, min(origin_offset, total_minutes))

                slider_val = st.slider(
                    "Drag to reconstruct vessel positions at any point in time",
                    min_value=0,
                    max_value=total_minutes,
                    value=origin_offset,
                    step=1,
                    format="%d min",
                    key="timeline_slider",
                )

                current_time = t_start + timedelta(minutes=slider_val)
                
                # Timeline labels
                tcol1, tcol2, tcol3 = st.columns(3)
                with tcol1:
                    st.caption(f"🟢 Start: {t_start.strftime('%H:%M UTC')}")
                with tcol2:
                    st.caption(f"▲ Origin: {origin_time.strftime('%H:%M UTC')} | 🕐 Current: **{current_time.strftime('%H:%M UTC')}**")
                with tcol3:
                    st.caption(f"🔴 End: {t_end.strftime('%H:%M UTC')}")

                # --- Vessel selector ---
                vessel_names = {r["mmsi"]: f"{r['name']} ({r['score']:.0f}%)" for r in rankings}
                selected_mmsi = st.selectbox(
                    "Select vessel to inspect",
                    options=list(vessel_names.keys()),
                    format_func=lambda x: vessel_names[x],
                    key="vessel_select",
                )

                # --- Map + Evidence panel side by side ---
                map_col, evidence_col = st.columns([3, 2])

                with map_col:
                    fmap = build_investigation_map(final_state, slider_val, selected_mmsi)
                    st_folium(fmap, height=520, use_container_width=True, key="investigation_map")

                with evidence_col:
                    # --- Source Correlation Score bars ---
                    st.markdown("#### Source Correlation Score")

                    for r in rankings:
                        score = r["score"]
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
                                <span>{r['name']}</span>
                                <span>{score:.0f}%</span>
                            </div>
                            <div class="correlation-bar-track">
                                <div class="correlation-bar-fill" style="width: {score}%; background-color: {bar_color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    # --- Vessel Evidence Card ---
                    sel_rank = ranking_map = {r["mmsi"]: r for r in rankings}.get(selected_mmsi)
                    if sel_rank:
                        # Get vessel state at current slider time
                        tracks_data = final_state.get("ais_tracks", {})
                        track_dicts = tracks_data.get(selected_mmsi, [])
                        if track_dicts:
                            track_records = [
                                AISRecord(
                                    mmsi=d["mmsi"], name=d["name"],
                                    lat=d["lat"], lon=d["lon"],
                                    heading=d["heading"], speed_knots=d["speed_knots"],
                                    timestamp=datetime.fromisoformat(d["timestamp"]),
                                ) for d in track_dicts
                            ]
                            v_now = get_vessel_at_time(track_records, current_time)
                        else:
                            v_now = {"lat": 0, "lon": 0, "heading": 0, "speed_knots": 0}

                        traj_icon = '<span class="match-yes">✓</span>' if sel_rank["trajectory_match"] else '<span class="match-no">✗</span>'
                        time_icon = '<span class="match-yes">✓</span>' if sel_rank["time_match"] else '<span class="match-no">✗</span>'
                        spat_icon = '<span class="match-yes">✓</span>' if sel_rank["spatial_match"] else '<span class="match-no">✗</span>'
                        drift_icon = '<span class="match-yes">✓</span>' if sel_rank["drift_consistency"] else '<span class="match-no">✗</span>'

                        score_color = "#ff4b4b" if sel_rank["score"] >= 70 else ("#ff8c00" if sel_rank["score"] >= 40 else "#ffc107")

                        st.markdown(f"""
                        <div class="evidence-card">
                            <h3>{sel_rank['name']}</h3>
                            <div class="evidence-row"><span class="evidence-label">AIS ID (MMSI)</span><span class="evidence-value">{sel_rank['mmsi']}</span></div>
                            <div class="evidence-row"><span class="evidence-label">Speed</span><span class="evidence-value">{v_now.get('speed_knots', sel_rank['speed_at_closest'])} kn</span></div>
                            <div class="evidence-row"><span class="evidence-label">Heading</span><span class="evidence-value">{v_now.get('heading', sel_rank['heading_at_closest'])}°</span></div>
                            <div class="evidence-row"><span class="evidence-label">Closest Distance</span><span class="evidence-value">{sel_rank['closest_distance_km']} km</span></div>
                            <div class="evidence-row"><span class="evidence-label">Nearest Approach</span><span class="evidence-value">{sel_rank['closest_time_str']}</span></div>
                            <div class="evidence-row"><span class="evidence-label">Time from Origin</span><span class="evidence-value">{sel_rank['time_diff_min']:.0f} min</span></div>
                            <hr style="border-color: #21262d; margin: 12px 0;">
                            <div class="evidence-row"><span class="evidence-label">TRAJECTORY MATCH</span>{traj_icon}</div>
                            <div class="evidence-row"><span class="evidence-label">TIME MATCH</span>{time_icon}</div>
                            <div class="evidence-row"><span class="evidence-label">SPATIAL MATCH</span>{spat_icon}</div>
                            <div class="evidence-row"><span class="evidence-label">DRIFT CONSISTENCY</span>{drift_icon}</div>
                            <hr style="border-color: #21262d; margin: 12px 0;">
                            <div class="correlation-score-big" style="color: {score_color};">
                                {sel_rank['score']:.0f}%
                            </div>
                            <div style="text-align:center; color:#8b949e; font-size:12px; letter-spacing:1px;">SOURCE CORRELATION SCORE</div>
                        </div>
                        """, unsafe_allow_html=True)

                # --- Forensic disclaimer ---
                st.markdown("---")
                st.caption(
                    "⚖️ **Disclaimer:** This system identifies vessels whose AIS trajectories are statistically "
                    "correlated with the estimated spill origin. It does not constitute proof of liability. "
                    "Source Correlation Scores reflect the strength of spatial, temporal, and trajectory evidence only."
                )

else:
    with col1:
        st.info("Awaiting satellite telemetry. Please upload an image from the sidebar to begin.")