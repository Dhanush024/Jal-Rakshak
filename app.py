# pyrefly: ignore [missing-import]
"""
Jal-Rakshak — Maritime Intelligence & Satellite Operations Platform
====================================================================
Production-grade operational dashboard for Sentinel-1 SAR oil spill detection,
classical multi-signal consensus validation, ocean hindcast/forecast,
AIS candidate correlation, coastal threat assessment, and automated incident dossiers.
"""
import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import subprocess
import time
import json
import math
import numpy as np
from datetime import datetime, timezone, timedelta

# ──────────────────────────────────────────────────────────────
# OpenCV headless setup (production Codespaces safety)
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
from folium.plugins import Draw, Fullscreen, AntPath, MarkerCluster
from streamlit_folium import st_folium

# ──────────────────────────────────────────────────────────────
# Project imports
# ──────────────────────────────────────────────────────────────
from config.settings import (
    is_demo_mode, DEMO_SPILL_LAT, DEMO_SPILL_LON,
    MAP_BASEMAP, CARTO_API_KEY
)
from pipeline.graph import run_pipeline, compile_pipeline
from geospatial.distance import haversine_km, destination_point, bearing_deg, polygon_area_km2
from demo.scenario import CHENNAI_SCENARIO, get_or_create_demo_sar_patch
from reporting.pdf import generate_pdf_report
from sar.detection import YOLODetector, render_detection_overlay
from sar.classical import (
    generate_diagnostic_panels, ValidationStatus, validate_consensus,
    normalize_validation_result,
)
from sar.landmask import extract_land_mask, intersect_with_ocean

# ──────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jal-Rakshak | Maritime Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# DESIGN SYSTEM CSS: Maritime Operations Center
# Inspired by Motion, Bklit, Watermelon UI, Manus & Haikei
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        /* Surface Foundation */
        --bg-void: #070a12;
        --bg-panel: rgba(10, 15, 28, 0.55);
        --bg-panel-hover: rgba(14, 22, 42, 0.70);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glass-bright: rgba(255, 255, 255, 0.14);
        --border-cyan-glow: rgba(0, 229, 255, 0.32);
        
        /* Tactical Accents */
        --accent-cyan: #00e5ff;
        --accent-sky: #38bdf8;
        --accent-blue: #0284c7;
        --accent-violet: #8b5cf6;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        --accent-green: #10b981;

        /* Typography */
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;

        /* Text Hierarchy */
        --text-pure: #ffffff;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* Global Foundation & Atmosphere */
    html, body, [class*="css"] {
        font-family: var(--font-sans);
        color: var(--text-primary);
        letter-spacing: -0.01em;
        -webkit-font-smoothing: antialiased;
    }
    
    .stApp {
        background-color: var(--bg-void);
        background-image: 
            /* Subtle tactical scanline texture (non-intrusive) */
            linear-gradient(rgba(18, 24, 38, 0) 50%, rgba(0, 0, 0, 0.22) 50%),
            /* Fine orbital coordinate grid (32px) */
            linear-gradient(90deg, rgba(255, 255, 255, 0.016) 1px, transparent 1px),
            linear-gradient(rgba(255, 255, 255, 0.016) 1px, transparent 1px),
            /* Atmospheric orbital vignettes */
            radial-gradient(ellipse 90% 55% at 50% -12%, rgba(0, 229, 255, 0.065) 0%, transparent 72%),
            radial-gradient(ellipse 70% 45% at 92% 100%, rgba(139, 92, 246, 0.04) 0%, transparent 65%);
        background-size: 100% 4px, 32px 32px, 32px 32px, 100% 100%, 100% 100%;
        background-attachment: fixed;
    }

    /* Streamlit Header Bar */
    header[data-testid="stHeader"] {
        background: rgba(7, 10, 18, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid var(--border-glass);
    }
    
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3.5rem;
        max-width: 1460px;
    }

    /* ─── AMBIENT ATMOSPHERIC BACKGROUND EFFECTS ─── */
    @keyframes scanline-sweep {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }
    .ambient-scanline {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 14px;
        background: linear-gradient(180deg, transparent 0%, rgba(0, 229, 255, 0.04) 50%, rgba(0, 229, 255, 0.12) 51%, transparent 100%);
        pointer-events: none;
        z-index: 1;
        animation: scanline-sweep 14s linear infinite;
        opacity: 0.65;
    }

    @keyframes orbital-spin {
        0% { transform: translate(-50%, -50%) rotate(0deg); }
        100% { transform: translate(-50%, -50%) rotate(360deg); }
    }
    .ambient-orbital-ring {
        position: fixed;
        top: 38%;
        left: 78%;
        width: 650px;
        height: 650px;
        border: 1px dashed rgba(0, 229, 255, 0.032);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: orbital-spin 120s linear infinite;
    }
    .ambient-orbital-ring-inner {
        position: fixed;
        top: 38%;
        left: 78%;
        width: 420px;
        height: 420px;
        border: 1px dotted rgba(139, 92, 246, 0.028);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: orbital-spin 80s linear infinite reverse;
    }

    @media (prefers-reduced-motion: reduce) {
        .ambient-scanline, .ambient-orbital-ring, .ambient-orbital-ring-inner {
            display: none !important;
            animation: none !important;
        }
    }

    /* ─── COMMAND CENTER HUD STRIP ─── */
    .command-hud-strip {
        display: flex;
        justify-content: space-between;
        align-items: stretch;
        gap: 12px;
        margin-bottom: 10px;
    }
    .hud-pill {
        padding: 10px 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .command-ops-ribbon {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(10, 15, 28, 0.65);
        border: 1px solid var(--border-glass);
        backdrop-filter: blur(20px);
        padding: 8px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }

    /* ─── SAR INTELLIGENCE WORKSPACE & EVIDENCE TIMELINE (PHASE 5) ─── */
    .sar-viewer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(10, 15, 28, 0.65);
        border: 1px solid var(--border-glass);
        backdrop-filter: blur(20px);
        padding: 8px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .evidence-timeline {
        padding: 14px;
        margin-top: 12px;
        border-radius: 8px;
    }
    .timeline-step {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(14, 22, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 6px;
        padding: 10px 12px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .timeline-step:hover {
        border-color: rgba(0, 229, 255, 0.30);
        background: rgba(18, 28, 54, 0.70);
        transform: translateX(2px);
    }
    .timeline-step-badge {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 700;
        color: var(--accent-cyan);
        background: rgba(0, 229, 255, 0.12);
        border: 1px solid rgba(0, 229, 255, 0.28);
        border-radius: 4px;
        padding: 3px 6px;
        white-space: nowrap;
    }
    .timeline-step-content {
        flex: 1;
        min-width: 0;
    }
    .timeline-step-title {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
        color: var(--text-pure);
        text-transform: uppercase;
    }
    .timeline-step-desc {
        font-size: 10.5px;
        font-family: var(--font-mono);
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-top: 2px;
    }
    .timeline-step-status {
        font-size: 9.5px;
        font-family: var(--font-mono);
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .status-pass {
        color: #34d399;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .status-warn {
        color: #fbbf24;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .status-fail {
        color: #f87171;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .timeline-arrow {
        text-align: center;
        color: var(--accent-cyan);
        font-size: 13px;
        line-height: 1;
        margin: 4px 0;
        opacity: 0.70;
    }
    .polygon-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 229, 255, 0.08);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        font-family: var(--font-mono);
        color: var(--text-primary);
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* ─── PURPOSEFUL MICRO-INTERACTIONS & MOTION (PHASE 4) ─── */
    @keyframes panel-entrance {
        0% {
            opacity: 0;
            transform: translateY(8px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slide-in-evidence {
        0% {
            opacity: 0;
            transform: translateX(14px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes subtle-shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    @keyframes vessel-radar-ping {
        0% { transform: scale(0.9); opacity: 0.8; }
        70% { transform: scale(1.4); opacity: 0.1; }
        100% { transform: scale(1.6); opacity: 0; }
    }

    /* Tab panels smooth entrance */
    div[data-testid="stTabs"] div[role="tabpanel"] {
        animation: panel-entrance 0.26s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    /* Cards Micro-Interactions:
       Hover: slight translateY(-2px), subtle border illumination, very small background shift.
       Click: brief compression and content transition */
    .glass-panel,
    .metric-card,
    .vessel-card,
    .recent-card,
    .evidence-box {
        background: var(--bg-panel) !important;
        border: 1px solid var(--border-glass) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.42), inset 0 1px 0 0 rgba(255, 255, 255, 0.08) !important;
        transition: transform 0.22s cubic-bezier(0.16, 1, 0.3, 1),
                    border-color 0.22s ease,
                    box-shadow 0.22s ease,
                    background 0.22s ease !important;
        animation: panel-entrance 0.32s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    }

    .glass-panel:hover,
    .metric-card:hover,
    .vessel-card:hover,
    .recent-card:hover,
    .evidence-box:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(0, 229, 255, 0.28) !important;
        background: rgba(14, 22, 42, 0.70) !important;
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.55),
                    0 0 16px rgba(0, 229, 255, 0.08),
                    inset 0 1px 0 0 rgba(255, 255, 255, 0.14) !important;
    }

    .glass-panel:active,
    .metric-card:active,
    .vessel-card:active,
    .recent-card:active,
    .evidence-box:active {
        transform: translateY(0) scale(0.985) !important;
        transition: transform 0.08s ease !important;
    }

    /* Buttons Micro-Interactions */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 8px 16px;
        transition: transform 0.18s cubic-bezier(0.16, 1, 0.3, 1),
                    background 0.18s ease,
                    border-color 0.18s ease,
                    box-shadow 0.18s ease !important;
        font-family: var(--font-sans);
    }
    .stButton > button:hover {
        transform: translateY(-1.5px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.975) !important;
        transition: transform 0.08s ease !important;
    }



    /* ─── COMMAND CENTER BRAND BAR ─── */
    .brand-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 22px;
        margin-bottom: 18px;
        position: relative;
    }
    .brand-title-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: var(--text-pure);
        margin: 0;
        text-shadow: 0 0 16px rgba(0, 229, 255, 0.25);
    }
    .brand-subtitle {
        font-size: 11px;
        font-weight: 500;
        color: var(--text-secondary);
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .brand-meta-group {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .telemetry-readout {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    .readout-label {
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: var(--text-muted);
        font-family: var(--font-mono);
    }
    .readout-val {
        font-size: 12px;
        font-weight: 600;
        color: var(--accent-sky);
        font-family: var(--font-mono);
        letter-spacing: 0.3px;
    }

    /* Pulsing Status Dot */
    @keyframes live-pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .status-pulse {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: var(--accent-green);
        box-shadow: 0 0 10px var(--accent-green);
        animation: live-pulse 2s infinite ease-in-out;
    }
    .status-pulse-sm {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--accent-cyan);
        box-shadow: 0 0 8px var(--accent-cyan);
        margin-right: 6px;
    }

    /* ─── TYPOGRAPHY & TELEMETRY LABELS ─── */
    .telemetry-label, .metric-label {
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
        margin-bottom: 5px !important;
        font-family: var(--font-sans) !important;
    }
    .telemetry-value, .metric-value {
        font-size: 24px !important;
        font-weight: 700 !important;
        font-family: var(--font-mono) !important;
        color: var(--text-pure) !important;
        line-height: 1.15 !important;
        letter-spacing: -0.5px !important;
    }
    .telemetry-value-sm {
        font-size: 14px !important;
        font-weight: 600 !important;
        font-family: var(--font-mono) !important;
        color: var(--text-primary) !important;
    }
    .telemetry-sub, .metric-sub {
        font-size: 11px !important;
        font-family: var(--font-mono) !important;
        color: var(--text-secondary) !important;
        margin-top: 5px !important;
        letter-spacing: 0.2px !important;
    }
    .telemetry-micro-label {
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: var(--accent-cyan);
        margin-bottom: 2px;
        font-family: var(--font-mono);
        display: block;
    }
    .telemetry-coords {
        font-size: 11px;
        font-family: var(--font-mono);
        color: var(--text-muted);
        margin-top: 3px;
        letter-spacing: 0.3px;
    }
    .telemetry-chip {
        display: inline-block;
        font-size: 10px;
        font-weight: 700;
        font-family: var(--font-mono);
        color: var(--accent-cyan);
        background: rgba(0, 229, 255, 0.12);
        border: 1px solid rgba(0, 229, 255, 0.28);
        border-radius: 4px;
        padding: 1px 6px;
        letter-spacing: 0.5px;
    }

    /* ─── NAVIGATION TABS (MISSION WORKSPACE SELECTOR) ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(10, 15, 28, 0.72) !important;
        border: 1px solid var(--border-glass) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        padding: 5px;
        border-radius: 8px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 18px;
        border-radius: 6px;
        color: var(--text-secondary);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        transition: all 0.16s ease;
        border: 1px solid transparent;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(255, 255, 255, 0.06);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(14, 165, 233, 0.15) !important;
        color: var(--accent-cyan) !important;
        font-weight: 700 !important;
        border: 1px solid rgba(0, 229, 255, 0.32) !important;
        box-shadow: 0 2px 12px rgba(0, 229, 255, 0.18), inset 0 1px 0 0 rgba(255, 255, 255, 0.15) !important;
    }

    /* ─── STATUS BADGES & DATA CLASSIFICATION TAGS ─── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-family: var(--font-mono);
    }
    .badge-confirmed {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.14);
    }
    .badge-probable {
        background: rgba(0, 229, 255, 0.12);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.35);
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.14);
    }
    .badge-lookalike {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-rejected {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-inconclusive {
        background: rgba(148, 163, 184, 0.10);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    .data-tag {
        display: inline-flex;
        align-items: center;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-family: var(--font-mono);
    }
    .tag-observed {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .tag-inferred {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .tag-predicted {
        background: rgba(139, 92, 246, 0.14);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.35);
    }
    .tag-simulated {
        background: rgba(239, 68, 68, 0.10);
        color: #f87171;
        border: 1px dashed rgba(239, 68, 68, 0.35);
    }
    .tag-official {
        background: rgba(0, 229, 255, 0.12);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.35);
    }

    /* ─── CANDIDATE VESSEL CARDS ─── */
    .vessel-card {
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .vessel-card-selected {
        border: 1px solid var(--accent-cyan) !important;
        background: rgba(14, 28, 56, 0.65) !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.22), inset 0 1px 0 0 rgba(255, 255, 255, 0.18) !important;
    }
    .vessel-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .vessel-name {
        font-size: 15px;
        font-weight: 700;
        color: var(--text-pure);
        letter-spacing: 0.2px;
    }
    .vessel-mmsi {
        font-size: 11px;
        color: var(--text-muted);
        font-family: var(--font-mono);
        letter-spacing: 0.5px;
    }
    .vessel-score {
        font-size: 20px;
        font-weight: 800;
        font-family: var(--font-mono);
        line-height: 1;
    }
    .vessel-score-high { color: var(--accent-red); }
    .vessel-score-med  { color: var(--accent-amber); }
    .vessel-score-low  { color: var(--text-secondary); }

    /* Telemetry Progress Bar */
    .bar-bg {
        background: rgba(255, 255, 255, 0.07);
        height: 5px;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 8px;
    }
    .bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* 4-column metric unit inside cards */
    .telemetry-grid-4 {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 14px;
        font-size: 12px;
    }
    .telemetry-metric-unit {
        display: flex;
        flex-direction: column;
    }

    /* ─── TACTICAL BUTTONS ─── */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 8px 16px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        font-family: var(--font-sans);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #0284c7 0%, #0369a1 100%) !important;
        border: 1px solid #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 12px rgba(2, 132, 199, 0.32), inset 0 1px 0 0 rgba(255, 255, 255, 0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #0ea5e9 0%, #0284c7 100%) !important;
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 4px 18px rgba(0, 229, 255, 0.38), inset 0 1px 0 0 rgba(255, 255, 255, 0.35) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background: rgba(14, 22, 42, 0.65) !important;
        border: 1px solid var(--border-glass) !important;
        color: #cbd5e1 !important;
        backdrop-filter: blur(12px) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(30, 41, 59, 0.75) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #f8fafc !important;
        transform: translateY(-1px);
    }

    /* ─── SIDEBAR TACTICAL RAIL ─── */
    section[data-testid="stSidebar"] {
        background: rgba(7, 10, 18, 0.90) !important;
        border-right: 1px solid var(--border-glass) !important;
        backdrop-filter: blur(28px) !important;
        -webkit-backdrop-filter: blur(28px) !important;
        box-shadow: 6px 0 32px rgba(0, 0, 0, 0.6) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
        margin: 14px 0 !important;
    }

    /* ─── STREAMLIT NATIVE INPUTS & WIDGETS ─── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(10, 15, 28, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
    }
    div[data-baseweb="select"]:hover > div,
    div[data-baseweb="input"]:focus-within > div {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.18) !important;
    }
    
    /* Radio Pill Controller */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background: rgba(10, 15, 28, 0.65);
        border: 1px solid var(--border-glass);
        backdrop-filter: blur(16px);
        border-radius: 6px;
        padding: 4px 6px;
        gap: 6px;
    }

    /* Folium Map Frame */
    iframe {
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.06) !important;
    }

    /* Tactical Map Header */
    .map-tactical-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(10, 15, 28, 0.85);
        border: 1px solid var(--border-glass);
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 7px 14px;
        font-size: 10px;
        font-family: var(--font-mono);
        color: var(--text-muted);
        letter-spacing: 0.5px;
    }
    .map-tactical-title {
        color: var(--text-secondary);
        font-weight: 700;
        letter-spacing: 0.8px;
    }

    /* Recent Analyses Cards */
    .recent-card {
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .recent-meta-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 10.5px;
        font-family: var(--font-mono);
        color: var(--text-muted);
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 10px;
        margin-top: 12px;
    }

    /* Responsive adjustments */
    @media (max-width: 900px) {
        .brand-bar {
            flex-direction: column;
            align-items: flex-start;
            gap: 12px;
        }
        .brand-meta-group {
            width: 100%;
            justify-content: space-between;
        }
        .telemetry-grid-4 {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    /* ── Phase 8 Command Palette Styles ── */
    .cmd-topbar-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(14, 22, 42, 0.65);
        border: 1px solid rgba(0, 229, 255, 0.35);
        border-radius: 6px;
        padding: 4px 10px;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    .cmd-topbar-pill:hover {
        background: rgba(0, 229, 255, 0.16);
        border-color: #00e5ff;
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.40);
        transform: translateY(-1px);
    }
    .cmd-pill-key {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 700;
        color: var(--accent-cyan);
        background: rgba(0, 229, 255, 0.15);
        border: 1px solid rgba(0, 229, 255, 0.45);
        border-radius: 3px;
        padding: 1px 5px;
        line-height: 1.2;
    }
    .cmd-pill-label {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 600;
        color: #f8fafc;
        letter-spacing: 0.04em;
    }
    @keyframes componentPulse {
        0% {
            box-shadow: 0 0 0 rgba(0, 229, 255, 0);
            border-color: rgba(0, 229, 255, 0.2);
        }
        25% {
            box-shadow: 0 0 32px rgba(0, 229, 255, 0.55), inset 0 0 16px rgba(0, 229, 255, 0.20);
            border-color: #00e5ff !important;
        }
        75% {
            box-shadow: 0 0 36px rgba(0, 229, 255, 0.40), inset 0 0 10px rgba(0, 229, 255, 0.12);
            border-color: #00e5ff !important;
        }
        100% {
            box-shadow: 0 0 0 rgba(0, 229, 255, 0);
            border-color: rgba(255, 255, 255, 0.08);
        }
    }
    .component-highlight-active {
        animation: componentPulse 2.2s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    }
</style>
""", unsafe_allow_html=True)

# Centralized design system injection
if os.path.exists("assets/theme.css"):
    try:
        with open("assets/theme.css", "r", encoding="utf-8") as _f_theme:
            st.markdown(f"<style>{_f_theme.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def render_tag(classification: str) -> str:
    """Render subtle, non-intrusive data classification tag."""
    c_lower = classification.lower()
    tag_class = f"tag-{c_lower}" if c_lower in ("observed", "inferred", "predicted", "simulated") else "tag-inferred"
    return f'<span class="data-tag {tag_class}">{classification}</span>'


def render_atmospheric_backdrop():
    """Render subtle, lightweight 40-particle ambient constellation and scanline layer."""
    st.markdown("""
    <div class="ambient-scanline" aria-hidden="true"></div>
    <div class="ambient-orbital-ring" aria-hidden="true"></div>
    <div class="ambient-orbital-ring-inner" aria-hidden="true"></div>
    """, unsafe_allow_html=True)
    
    # 40-particle canvas injection with prefers-reduced-motion and visibilitychange safety
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
      if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
      let targetDoc = document;
      try {
        if (window.parent && window.parent.document) {
          targetDoc = window.parent.document;
        }
      } catch(e) {}
      if (targetDoc.getElementById('jal-rakshak-atmosphere-canvas')) return;

      const canvas = targetDoc.createElement('canvas');
      canvas.id = 'jal-rakshak-atmosphere-canvas';
      canvas.style.position = 'fixed';
      canvas.style.top = '0';
      canvas.style.left = '0';
      canvas.style.width = '100vw';
      canvas.style.height = '100vh';
      canvas.style.pointerEvents = 'none';
      canvas.style.zIndex = '0';
      canvas.style.opacity = '0.45';
      targetDoc.body.appendChild(canvas);

      const ctx = canvas.getContext('2d', { alpha: true });
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      let w = window.innerWidth;
      let h = window.innerHeight;

      function resizeCanvas() {
        w = window.innerWidth;
        h = window.innerHeight;
        canvas.width = Math.floor(w * dpr);
        canvas.height = Math.floor(h * dpr);
        ctx.scale(dpr, dpr);
      }
      resizeCanvas();

      // Debounced resize listener with cleanup
      let resizeTimer;
      function handleResize() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(resizeCanvas, 100);
      }
      if (window.parent.__jalAtmosphereResize) {
        try {
          window.parent.removeEventListener('resize', window.parent.__jalAtmosphereResize);
        } catch(e) {}
      }
      window.parent.__jalAtmosphereResize = handleResize;
      try {
        window.parent.addEventListener('resize', handleResize);
      } catch(e) {}

      // Pre-allocated particle pool (Zero object creation in animation frame)
      const PARTICLE_COUNT = 32;
      const particles = new Array(PARTICLE_COUNT);
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles[i] = {
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.24,
          vy: (Math.random() - 0.5) * 0.24,
          radius: Math.random() * 1.2 + 0.8,
          color: Math.random() > 0.35 ? 'rgba(0, 229, 255,' : 'rgba(139, 92, 246,',
          alpha: Math.random() * 0.25 + 0.12
        };
      }

      let isVisible = !targetDoc.hidden;
      function onVisibility() {
        const nowVis = !targetDoc.hidden && !document.hidden;
        if (nowVis && !isVisible) {
          isVisible = true;
          requestAnimationFrame(loop);
        } else {
          isVisible = nowVis;
        }
      }
      targetDoc.addEventListener('visibilitychange', onVisibility);
      document.addEventListener('visibilitychange', onVisibility);

      let animId = null;
      function loop() {
        // Stop execution if canvas is detached or document is hidden
        if (!canvas.isConnected || !isVisible) {
          animId = null;
          return;
        }
        ctx.clearRect(0, 0, w, h);

        // Render network lines
        for (let i = 0; i < PARTICLE_COUNT; i++) {
          const pi = particles[i];
          for (let j = i + 1; j < PARTICLE_COUNT; j++) {
            const pj = particles[j];
            const dx = pi.x - pj.x;
            const dy = pi.y - pj.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 90) {
              ctx.beginPath();
              ctx.moveTo(pi.x, pi.y);
              ctx.lineTo(pj.x, pj.y);
              ctx.strokeStyle = `rgba(0, 229, 255, ${0.08 * (1 - dist / 90)})`;
              ctx.lineWidth = 0.5;
              ctx.stroke();
            }
          }
        }

        // Update positions and render particles
        for (let i = 0; i < PARTICLE_COUNT; i++) {
          const p = particles[i];
          p.x += p.vx;
          p.y += p.vy;

          if (p.x < 0) p.x = w;
          if (p.x > w) p.x = 0;
          if (p.y < 0) p.y = h;
          if (p.y > h) p.y = 0;

          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
          ctx.fillStyle = `${p.color} ${p.alpha})`;
          ctx.fill();
        }

        animId = requestAnimationFrame(loop);
      }

      animId = requestAnimationFrame(loop);
    })();
    </script>
    """, height=0, width=0)


def render_command_palette(spill_lat: float, spill_lon: float, source_lat: float, source_lon: float):
    """
    Render Phase 8 Premium Command Palette with real frontend action mapping.
    Keyboard shortcuts: '/' (when not typing) and 'Ctrl/Cmd + K'.
    """
    import streamlit.components.v1 as components

    template = """
    <script>
    (function() {
      let targetDoc = document;
      try {
        if (window.parent && window.parent.document) {
          targetDoc = window.parent.document;
        }
      } catch(e) {}

      // Store/Update latest coordinates globally on parent window
      window.parent.__jalRakshakCoords = {
        spillLat: __SPILL_LAT__,
        spillLon: __SPILL_LON__,
        sourceLat: __SOURCE_LAT__,
        sourceLon: __SOURCE_LON__
      };

      // 1. Ensure modal & toast styles exist in targetDoc.head
      if (!targetDoc.getElementById('jalrakshak-cmd-palette-styles')) {
        const styleEl = targetDoc.createElement('style');
        styleEl.id = 'jalrakshak-cmd-palette-styles';
        styleEl.textContent = `
          #cmd-palette-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(4, 7, 15, 0.82);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            z-index: 999999;
            display: none;
            align-items: flex-start;
            justify-content: center;
            padding-top: 11vh;
            opacity: 0;
            transition: opacity 0.18s cubic-bezier(0.16, 1, 0.3, 1);
          }
          #cmd-palette-backdrop.open {
            display: flex;
            opacity: 1;
          }
          #cmd-palette-modal {
            width: 660px;
            max-width: 92vw;
            background: rgba(10, 15, 28, 0.94);
            border: 1px solid rgba(0, 229, 255, 0.35);
            border-radius: 12px;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.80), 0 0 45px rgba(0, 229, 255, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.12);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transform: translateY(-10px) scale(0.98);
            transition: transform 0.18s cubic-bezier(0.16, 1, 0.3, 1);
          }
          #cmd-palette-backdrop.open #cmd-palette-modal {
            transform: translateY(0) scale(1);
          }
          .cmd-header {
            display: flex;
            align-items: center;
            padding: 14px 18px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(14, 22, 42, 0.55);
            gap: 12px;
          }
          .cmd-prompt-sym {
            font-family: 'JetBrains Mono', monospace;
            font-size: 18px;
            font-weight: 700;
            color: #00e5ff;
            user-select: none;
          }
          #cmd-palette-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            font-family: 'JetBrains Mono', monospace;
            font-size: 15px;
            font-weight: 500;
            color: #ffffff;
            caret-color: #00e5ff;
          }
          #cmd-palette-input::placeholder {
            color: #64748b;
            font-size: 14px;
          }
          .cmd-esc-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 600;
            color: #94a3b8;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 4px;
            padding: 3px 6px;
            cursor: pointer;
            transition: all 0.15s ease;
          }
          .cmd-esc-badge:hover {
            background: rgba(239, 68, 68, 0.2);
            border-color: rgba(239, 68, 68, 0.5);
            color: #fca5a5;
          }
          #cmd-palette-list {
            max-height: 380px;
            overflow-y: auto;
            padding: 8px;
          }
          #cmd-palette-list::-webkit-scrollbar {
            width: 6px;
          }
          #cmd-palette-list::-webkit-scrollbar-track {
            background: rgba(10, 15, 28, 0.4);
          }
          #cmd-palette-list::-webkit-scrollbar-thumb {
            background: rgba(0, 229, 255, 0.25);
            border-radius: 3px;
          }
          .cmd-group-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #64748b;
            padding: 8px 12px 4px 12px;
            text-transform: uppercase;
          }
          .cmd-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 2px;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.12s ease;
          }
          .cmd-item:hover, .cmd-item.selected {
            background: rgba(0, 229, 255, 0.12);
            border-color: rgba(0, 229, 255, 0.35);
          }
          .cmd-item-left {
            display: flex;
            align-items: center;
            gap: 12px;
          }
          .cmd-item-icon {
            font-size: 16px;
            width: 24px;
            text-align: center;
          }
          .cmd-item-title {
            font-family: 'Inter', sans-serif;
            font-size: 13.5px;
            font-weight: 600;
            color: #f8fafc;
          }
          .cmd-item.selected .cmd-item-title {
            color: #00e5ff;
          }
          .cmd-item-desc {
            font-size: 11.5px;
            color: #94a3b8;
            margin-top: 1px;
          }
          .cmd-item-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 600;
            color: #00e5ff;
            background: rgba(0, 229, 255, 0.12);
            border: 1px solid rgba(0, 229, 255, 0.28);
            border-radius: 4px;
            padding: 2px 7px;
            white-space: nowrap;
          }
          .cmd-no-results {
            padding: 28px;
            text-align: center;
            color: #64748b;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
          }
          .cmd-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 9px 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            background: rgba(7, 10, 18, 0.95);
            font-family: 'JetBrains Mono', monospace;
            font-size: 10.5px;
            color: #64748b;
          }
          .cmd-footer-shortcuts {
            display: flex;
            gap: 14px;
            align-items: center;
          }
          .cmd-footer-key {
            color: #94a3b8;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.10);
            padding: 1px 4px;
            border-radius: 3px;
            font-size: 9.5px;
          }
          #cmd-tactical-toast {
            position: fixed;
            top: 24px;
            right: 24px;
            background: rgba(10, 15, 28, 0.94);
            border: 1px solid rgba(0, 229, 255, 0.35);
            border-left: 4px solid #00e5ff;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.65), 0 0 24px rgba(0, 229, 255, 0.20);
            border-radius: 8px;
            padding: 12px 18px;
            z-index: 1000000;
            pointer-events: none;
            opacity: 0;
            transform: translateX(20px);
            transition: opacity 0.25s cubic-bezier(0.16, 1, 0.3, 1), transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            max-width: 440px;
          }
          #cmd-tactical-toast.cmd-toast-visible {
            opacity: 1;
            transform: translateX(0);
          }
          #cmd-tactical-toast.cmd-toast-hidden {
            opacity: 0;
            transform: translateX(20px);
          }
        `;
        targetDoc.head.appendChild(styleEl);
      }

      // 2. Helper Functions for Real Actions
      function showToast(title, detail) {
        let toast = targetDoc.getElementById('cmd-tactical-toast');
        if (!toast) {
          toast = targetDoc.createElement('div');
          toast.id = 'cmd-tactical-toast';
          targetDoc.body.appendChild(toast);
        }
        toast.innerHTML = `
          <div style="display:flex; align-items:flex-start; gap:10px;">
            <span style="background:#00e5ff; box-shadow:0 0 10px #00e5ff; margin-top:5px; display:inline-block; width:7px; height:7px; border-radius:50%;"></span>
            <div>
              <div style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:700; color:#00e5ff; letter-spacing:0.04em;">${title}</div>
              <div style="font-family:'Inter', sans-serif; font-size:12px; color:#e2e8f0; margin-top:2px;">${detail}</div>
            </div>
          </div>
        `;
        toast.className = 'cmd-toast-visible';
        clearTimeout(window.parent.__jalToastTimer);
        window.parent.__jalToastTimer = setTimeout(() => {
          toast.className = 'cmd-toast-hidden';
        }, 4000);
      }

      function animateComponent(selector) {
        try {
          const els = targetDoc.querySelectorAll(selector);
          els.forEach(el => {
            el.classList.remove('component-highlight-active');
            void el.offsetWidth;
            el.classList.add('component-highlight-active');
          });
        } catch(e) {}
      }

      function clickButtonByText(text) {
        try {
          const buttons = Array.from(targetDoc.querySelectorAll('button'));
          const btn = buttons.find(b => b.textContent && b.textContent.trim().toLowerCase().includes(text.toLowerCase()));
          if (btn) {
            btn.click();
            return true;
          }
        } catch(e) {}
        return false;
      }

      function clickMainTab(index) {
        try {
          const tabLists = targetDoc.querySelectorAll('div[data-baseweb="tab-list"]');
          if (tabLists.length > 0) {
            const tabs = tabLists[0].querySelectorAll('button[data-baseweb="tab"]');
            if (tabs[index]) {
              tabs[index].click();
              return true;
            }
          }
        } catch(e) {}
        return false;
      }

      function clickTabByText(tabText) {
        try {
          const tabs = Array.from(targetDoc.querySelectorAll('button[data-baseweb="tab"]'));
          const tab = tabs.find(t => t.textContent && t.textContent.trim().toLowerCase().includes(tabText.toLowerCase()));
          if (tab) {
            tab.click();
            return true;
          }
        } catch(e) {}
        return false;
      }

      function clickRadioByText(text) {
        try {
          const radios = Array.from(targetDoc.querySelectorAll('label[data-baseweb="radio"]'));
          const radio = radios.find(r => r.textContent && r.textContent.trim().toLowerCase().includes(text.toLowerCase()));
          if (radio) {
            radio.click();
            return true;
          }
        } catch(e) {}
        return false;
      }

      function flyLeafletMap(lat, lon, zoom) {
        let flown = false;
        try {
          for (const key in window.parent) {
            try {
              const obj = window.parent[key];
              if (obj && typeof obj.flyTo === 'function') {
                obj.flyTo([lat, lon], zoom || 12, { duration: 1.5 });
                flown = true;
                break;
              }
            } catch(e) {}
          }
        } catch(e) {}
        if (!flown) {
          try {
            const iframes = targetDoc.querySelectorAll('iframe');
            for (const ifr of iframes) {
              try {
                const win = ifr.contentWindow;
                if (!win) continue;
                for (const key in win) {
                  try {
                    const obj = win[key];
                    if (obj && typeof obj.flyTo === 'function') {
                      obj.flyTo([lat, lon], zoom || 12, { duration: 1.5 });
                      flown = true;
                      break;
                    }
                  } catch(e) {}
                }
              } catch(e) {}
              if (flown) break;
            }
          } catch(e) {}
        }
      }

      // 3. Command Definitions mapped directly to real frontend actions
      const COMMANDS = [
        {
          id: 'show_spill',
          icon: '🎯',
          title: 'Show Spill',
          desc: 'Center primary canvas on detected oil spill polygon & centroid',
          category: 'Geospatial Canvas',
          badge: 'LAYER',
          keywords: ['show spill', 'spill', 'slick', 'detect', 'focus spill', 'polygon', 'centroid'],
          action: () => {
            clickMainTab(0);
            const c = window.parent.__jalRakshakCoords || {};
            flyLeafletMap(c.spillLat, c.spillLon, 13);
            clickButtonByText('Focus Spill') || clickRadioByText('SPILL');
            animateComponent('.hud-pill');
            animateComponent('.leaflet-container');
            showToast('🎯 SHOW SPILL', `Primary canvas centered on detected slick [${c.spillLat.toFixed(4)}°N, ${c.spillLon.toFixed(4)}°E]`);
          }
        },
        {
          id: 'show_vessels',
          icon: '🚢',
          title: 'Show Vessels',
          desc: 'Activate correlated AIS fleet tracks and candidate positions layer',
          category: 'Geospatial Canvas',
          badge: 'AIS FLEET',
          keywords: ['show vessels', 'vessels', 'vessel', 'ais', 'fleet', 'ships', 'traffic'],
          action: () => {
            clickMainTab(0);
            clickRadioByText('AIS');
            animateComponent('.vessel-card');
            animateComponent('.leaflet-container');
            showToast('🚢 SHOW VESSELS', 'Correlated AIS fleet tracks & dynamic positions active');
          }
        },
        {
          id: 'trace_source',
          icon: '⚡',
          title: 'Trace Source',
          desc: 'Hindcast advection to origin, fly to source & engage candidate attribution panel',
          category: 'Actions & Analysis',
          badge: 'FORENSIC ACTION',
          keywords: ['trace source', 'trace', 'source', 'origin', 'candidate panel', 'attribution'],
          action: () => {
            clickMainTab(0);
            const c = window.parent.__jalRakshakCoords || {};
            flyLeafletMap(c.sourceLat, c.sourceLon, 12.5);
            clickButtonByText('TRACE SOURCE') || clickButtonByText('TRACING ACTIVE') || clickRadioByText('BACKTRACK');
            clickTabByText('AIS Candidate Vessel Attribution');
            animateComponent('.vessel-card');
            animateComponent('.evidence-box');
            animateComponent('.glass-panel');
            showToast('⚡ TRACE SOURCE', `Advection backtrack engaged → Map flown to source [${c.sourceLat.toFixed(4)}°N, ${c.sourceLon.toFixed(4)}°E] // Candidate Attribution Engaged`);
          }
        },
        {
          id: 'show_backtrack',
          icon: '⏱️',
          title: 'Show Backtrack',
          desc: 'Display historical reverse-advection hindcast trajectory (T-12h → T0)',
          category: 'Geospatial Canvas',
          badge: 'HINDCAST',
          keywords: ['show backtrack', 'backtrack', 'hindcast', 'history', 'origin', 'trail'],
          action: () => {
            clickMainTab(0);
            const c = window.parent.__jalRakshakCoords || {};
            flyLeafletMap(c.sourceLat, c.sourceLon, 12);
            clickButtonByText('TRACE SOURCE') || clickRadioByText('BACKTRACK');
            animateComponent('.glass-panel');
            showToast('⏱️ SHOW BACKTRACK', 'Historical reverse-drift backtrack trail (T-12h → T0) activated');
          }
        },
        {
          id: 'show_drift',
          icon: '🌊',
          title: 'Show Drift',
          desc: 'Simulate forward hydrodynamic drift trajectory and dispersion envelope',
          category: 'Actions & Analysis',
          badge: 'HYDRODYNAMIC',
          keywords: ['show drift', 'drift', 'simulate drift', 'dispersion', 'forecast', 'currents'],
          action: () => {
            clickMainTab(4) || clickButtonByText('SIMULATE DRIFT') || clickRadioByText('FORWARD DRIFT');
            animateComponent('.drift-disclosure-banner');
            animateComponent('.metric-card');
            showToast('🌊 SHOW DRIFT', 'Forward hydrodynamic trajectory & dispersion simulation active');
          }
        },
        {
          id: 'open_sar',
          icon: '🔬',
          title: 'Open SAR',
          desc: 'Navigate to Sentinel-1 C-Band SAR intelligence workspace & mask inspector',
          category: 'Workspaces',
          badge: 'WORKSPACE',
          keywords: ['open sar', 'sar', 'sentinel', 'radar', 'segmentation', 'mask', 'inspection'],
          action: () => {
            clickMainTab(1) || clickTabByText('SAR Intelligence');
            animateComponent('.glass-panel');
            showToast('🔬 OPEN SAR', 'Navigating to Sentinel-1 C-Band SAR intelligence workspace');
          }
        },
        {
          id: 'open_ais',
          icon: '🚢',
          title: 'Open AIS',
          desc: 'Navigate to correlated candidate vessel fleet intelligence workspace',
          category: 'Workspaces',
          badge: 'WORKSPACE',
          keywords: ['open ais', 'ais', 'vessel workspace', 'candidates', 'fleet', 'traffic'],
          action: () => {
            clickMainTab(3) || clickTabByText('AIS Intelligence');
            animateComponent('.vessel-card');
            showToast('🚢 OPEN AIS', 'Navigating to correlated AIS candidate fleet workspace');
          }
        },
        {
          id: 'open_reports',
          icon: '📋',
          title: 'Open Reports',
          desc: 'Access automated forensic incident dossier & intelligence reports',
          category: 'Workspaces',
          badge: 'WORKSPACE',
          keywords: ['open reports', 'reports', 'report', 'dossier', 'forensic', 'export', 'pdf'],
          action: () => {
            clickMainTab(6) || clickTabByText('Intelligence Reports');
            animateComponent('.glass-panel');
            showToast('📋 OPEN REPORTS', 'Opening automated forensic incident dossier & export console');
          }
        },
        {
          id: 'open_command_center',
          icon: '🛰️',
          title: 'Open Command Center',
          desc: 'Return to primary command center canvas and operational telemetry',
          category: 'Workspaces',
          badge: 'WORKSPACE',
          keywords: ['open command center', 'command center', 'overview', 'home', 'main'],
          action: () => {
            clickMainTab(0) || clickTabByText('Command Center');
            animateComponent('.hud-pill');
            showToast('🛰️ COMMAND CENTER', 'Primary operations canvas and situational telemetry active');
          }
        },
        {
          id: 'open_calibration',
          icon: '📡',
          title: 'Open Radar Calibration',
          desc: 'Inspect wind-wave dampening ratios and NESZ radar calibration matrix',
          category: 'Workspaces',
          badge: 'WORKSPACE',
          keywords: ['open radar calibration', 'calibration', 'radar', 'nesz', 'damping'],
          action: () => {
            clickMainTab(2) || clickTabByText('Radar Calibration');
            animateComponent('.glass-panel');
            showToast('📡 RADAR CALIBRATION', 'Wind-wave dampening & NESZ radar diagnostics workspace');
          }
        },
        {
          id: 'open_coastal',
          icon: '🛡️',
          title: 'Open Coastal Threat',
          desc: 'Assess shoreline vulnerability, sensitive assets, and arrival timeline',
          category: 'Workspaces',
          badge: 'WORKSPACE',
          keywords: ['open coastal threat', 'coastal', 'threat', 'shoreline', 'vulnerability', 'assets'],
          action: () => {
            clickMainTab(5) || clickTabByText('Coastal Threat');
            animateComponent('.glass-panel');
            showToast('🛡️ COASTAL THREAT', 'Shoreline vulnerability & sensitive infrastructure assessment');
          }
        },
        {
          id: 'execute_pipeline',
          icon: '⚡',
          title: 'Execute Pipeline',
          desc: 'Launch 11-node LangGraph autonomous maritime intelligence pipeline',
          category: 'Operations',
          badge: 'SYSTEM',
          keywords: ['execute pipeline', 'run pipeline', 'pipeline', 'run', 'execute', 'langgraph'],
          action: () => {
            clickButtonByText('EXECUTE PIPELINE');
            showToast('⚡ EXECUTE PIPELINE', 'Triggering 11-node LangGraph intelligence pipeline');
          }
        },
        {
          id: 'mount_chennai',
          icon: '⚓',
          title: 'Mount Chennai Scene',
          desc: 'Load calibrated Chennai Port outer anchorage verified spill scenario',
          category: 'Operations',
          badge: 'SCENE',
          keywords: ['mount chennai', 'chennai', 'anchorage', 'scenario'],
          action: () => {
            clickButtonByText('Mount Chennai') || clickButtonByText('Chennai Spill');
            showToast('⚓ MOUNT SCENE', 'Chennai Port Outer Anchorage scenario mounted');
          }
        },
        {
          id: 'mount_istanbul',
          icon: '🌊',
          title: 'Mount Istanbul Control Scene',
          desc: 'Load Istanbul Bosphorus strait high-traffic false-positive control scene',
          category: 'Operations',
          badge: 'SCENE',
          keywords: ['mount istanbul', 'istanbul', 'bosphorus', 'control'],
          action: () => {
            clickButtonByText('Mount Istanbul') || clickButtonByText('Istanbul Scene');
            showToast('🌊 MOUNT SCENE', 'Istanbul Bosphorus Strait control scene mounted');
          }
        },
        {
          id: 'reset_map',
          icon: '🔄',
          title: 'Reset Map View',
          desc: 'Reset primary canvas camera center and clear active layer isolations',
          category: 'Geospatial Canvas',
          badge: 'RESET',
          keywords: ['reset map', 'reset view', 'reset center', 'clear', 'default'],
          action: () => {
            clickButtonByText('Reset Center');
            const c = window.parent.__jalRakshakCoords || {};
            flyLeafletMap(c.spillLat, c.spillLon, 11);
            animateComponent('.leaflet-container');
            showToast('🔄 RESET VIEW', 'Map camera centered on default observation boundary');
          }
        },
        {
          id: 'focus_origin',
          icon: '📍',
          title: 'Focus Origin Centroid',
          desc: 'Center primary canvas camera on hindcast-derived source origin',
          category: 'Geospatial Canvas',
          badge: 'CAMERA',
          keywords: ['focus origin', 'origin', 'source centroid', 'camera origin'],
          action: () => {
            clickButtonByText('Focus Origin') || clickRadioByText('SOURCE');
            const c = window.parent.__jalRakshakCoords || {};
            flyLeafletMap(c.sourceLat, c.sourceLon, 13);
            showToast('📍 FOCUS ORIGIN', `Camera locked on estimated source origin [${c.sourceLat.toFixed(4)}°N, ${c.sourceLon.toFixed(4)}°E]`);
          }
        },
        {
          id: 'enter_live_mode',
          icon: '🛰️',
          title: 'Enter Live Operations',
          desc: 'Switch to Live Tactical Command Center with real-time map & AIS fleet radar',
          category: 'Navigation & Modes',
          badge: 'MODE',
          keywords: ['live', 'live operations', 'live ops', 'command center', 'realtime'],
          action: () => {
            clickButtonByText('Live Operations') || clickButtonByText('ENTER LIVE OPERATIONS');
            showToast('🛰️ LIVE OPERATIONS', 'Switching to Live Satellite Operations Center');
          }
        },
        {
          id: 'launch_demo_mode',
          icon: '🧪',
          title: 'Launch Guided Demo',
          desc: 'Switch to curated 5-step investigative narrative of Chennai spill',
          category: 'Navigation & Modes',
          badge: 'MODE',
          keywords: ['demo', 'guided demo', 'evaluation', 'walkthrough', 'chennai demo'],
          action: () => {
            clickButtonByText('Guided Demo') || clickButtonByText('LAUNCH GUIDED DEMO');
            showToast('🧪 GUIDED DEMO', 'Switching to 5-Step Guided Demo Evaluation Mode');
          }
        },
        {
          id: 'go_home',
          icon: '🏠',
          title: 'Go to Home Portal',
          desc: 'Return to minimal landing screen and operations portal',
          category: 'Navigation & Modes',
          badge: 'NAV',
          keywords: ['home', 'landing', 'portal', 'main', 'start'],
          action: () => {
            clickButtonByText('Home Portal');
            showToast('🏠 HOME PORTAL', 'Returning to main portal landing screen');
          }
        }
      ];

      // 4. Mount Command Palette DOM in targetDoc if not already mounted
      let backdrop = targetDoc.getElementById('cmd-palette-backdrop');
      if (!backdrop) {
        backdrop = targetDoc.createElement('div');
        backdrop.id = 'cmd-palette-backdrop';
        backdrop.innerHTML = `
          <div id="cmd-palette-modal" role="dialog" aria-modal="true" aria-label="Command Palette">
            <div class="cmd-header">
              <span class="cmd-prompt-sym">></span>
              <input type="text" id="cmd-palette-input" placeholder="Type a command or keyword... (e.g. 'trace source', 'open SAR', 'show spill')" autocomplete="off" spellcheck="false" />
              <span class="cmd-esc-badge" id="cmd-esc-btn" title="Close (ESC)">ESC</span>
            </div>
            <div id="cmd-palette-list" role="listbox"></div>
            <div class="cmd-footer">
              <div class="cmd-footer-shortcuts">
                <span><kbd class="cmd-footer-key">↑</kbd><kbd class="cmd-footer-key">↓</kbd> navigate</span>
                <span><kbd class="cmd-footer-key">↵</kbd> execute</span>
                <span><kbd class="cmd-footer-key">ESC</kbd> close</span>
                <span><kbd class="cmd-footer-key">/</kbd> search</span>
              </div>
              <div style="color:#00e5ff; font-weight:700; letter-spacing:0.06em;">JAL-RAKSHAK COMMAND SYSTEM</div>
            </div>
          </div>
        `;
        targetDoc.body.appendChild(backdrop);
      }

      const inputEl = targetDoc.getElementById('cmd-palette-input');
      const listEl = targetDoc.getElementById('cmd-palette-list');
      const escBtn = targetDoc.getElementById('cmd-esc-btn');

      let filteredCommands = [...COMMANDS];
      let selectedIndex = 0;

      function renderList() {
        listEl.innerHTML = '';
        if (filteredCommands.length === 0) {
          listEl.innerHTML = '<div class="cmd-no-results">No matching commands found. Try "trace", "spill", "ais", "sar", or "drift".</div>';
          return;
        }

        let lastCategory = '';
        filteredCommands.forEach((cmd, idx) => {
          if (cmd.category !== lastCategory) {
            lastCategory = cmd.category;
            const grp = targetDoc.createElement('div');
            grp.className = 'cmd-group-label';
            grp.textContent = lastCategory;
            listEl.appendChild(grp);
          }

          const item = targetDoc.createElement('div');
          item.className = 'cmd-item' + (idx === selectedIndex ? ' selected' : '');
          item.setAttribute('role', 'option');
          item.setAttribute('aria-selected', idx === selectedIndex ? 'true' : 'false');
          item.dataset.index = idx;
          item.innerHTML = `
            <div class="cmd-item-left">
              <span class="cmd-item-icon">${cmd.icon}</span>
              <div>
                <div class="cmd-item-title">${cmd.title}</div>
                <div class="cmd-item-desc">${cmd.desc}</div>
              </div>
            </div>
            <span class="cmd-item-badge">${cmd.badge}</span>
          `;

          item.addEventListener('mouseenter', () => {
            selectedIndex = idx;
            updateSelection();
          });

          item.addEventListener('click', (e) => {
            e.stopPropagation();
            executeCommand(idx);
          });

          listEl.appendChild(item);
        });
      }

      function updateSelection() {
        const items = listEl.querySelectorAll('.cmd-item');
        items.forEach((item, idx) => {
          if (idx === selectedIndex) {
            item.classList.add('selected');
            item.setAttribute('aria-selected', 'true');
            item.scrollIntoView({ block: 'nearest' });
          } else {
            item.classList.remove('selected');
            item.setAttribute('aria-selected', 'false');
          }
        });
      }

      function filterCommands(query) {
        const q = query.trim().toLowerCase();
        if (!q) {
          filteredCommands = [...COMMANDS];
        } else {
          filteredCommands = COMMANDS.filter(cmd => {
            if (cmd.title.toLowerCase().includes(q)) return true;
            if (cmd.desc.toLowerCase().includes(q)) return true;
            if (cmd.category.toLowerCase().includes(q)) return true;
            if (cmd.badge.toLowerCase().includes(q)) return true;
            return cmd.keywords.some(k => k.toLowerCase().includes(q));
          });
        }
        selectedIndex = 0;
        renderList();
      }

      function openPalette() {
        backdrop.classList.add('open');
        inputEl.value = '';
        filterCommands('');
        setTimeout(() => {
          inputEl.focus();
        }, 50);
      }

      function closePalette() {
        backdrop.classList.remove('open');
        inputEl.blur();
      }

      function executeCommand(idx) {
        const cmd = filteredCommands[idx];
        if (!cmd) return;
        closePalette();
        try {
          cmd.action();
        } catch(err) {
          console.error('Command execution error:', err);
        }
      }

      // Input event listeners (clean replace)
      inputEl.oninput = (e) => {
        filterCommands(e.target.value);
      };

      inputEl.onkeydown = (e) => {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          if (filteredCommands.length > 0) {
            selectedIndex = (selectedIndex + 1) % filteredCommands.length;
            updateSelection();
          }
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          if (filteredCommands.length > 0) {
            selectedIndex = (selectedIndex - 1 + filteredCommands.length) % filteredCommands.length;
            updateSelection();
          }
        } else if (e.key === 'Enter') {
          e.preventDefault();
          executeCommand(selectedIndex);
        } else if (e.key === 'Escape') {
          e.preventDefault();
          closePalette();
        }
      };

      escBtn.onclick = (e) => {
        e.stopPropagation();
        closePalette();
      };

      backdrop.onclick = (e) => {
        if (e.target === backdrop) {
          closePalette();
        }
      };

      // Global Keyboard Shortcuts
      function handleGlobalKeyDown(e) {
        if (e.key === 'Escape' && backdrop.classList.contains('open')) {
          e.preventDefault();
          closePalette();
          return;
        }

        // Ctrl+K or Cmd+K
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
          e.preventDefault();
          if (backdrop.classList.contains('open')) {
            closePalette();
          } else {
            openPalette();
          }
          return;
        }

        // '/' key when not focused in an input/textarea
        if (e.key === '/' && !backdrop.classList.contains('open')) {
          const activeEl = targetDoc.activeElement;
          const isInput = activeEl && (
            activeEl.tagName === 'INPUT' ||
            activeEl.tagName === 'TEXTAREA' ||
            activeEl.tagName === 'SELECT' ||
            activeEl.isContentEditable
          );
          if (!isInput) {
            e.preventDefault();
            openPalette();
          }
        }
      }

      if (window.parent.__jalCmdKeyHandler) {
        try {
          window.parent.document.removeEventListener('keydown', window.parent.__jalCmdKeyHandler);
          window.parent.removeEventListener('keydown', window.parent.__jalCmdKeyHandler);
        } catch(e) {}
      }
      window.parent.__jalCmdKeyHandler = handleGlobalKeyDown;
      try {
        window.parent.document.addEventListener('keydown', handleGlobalKeyDown);
        window.parent.addEventListener('keydown', handleGlobalKeyDown);
      } catch(e) {}
      document.addEventListener('keydown', handleGlobalKeyDown);

      // Custom Event listener for top bar button
      function handleCustomOpen() {
        openPalette();
      }
      if (window.parent.__jalCmdCustomOpen) {
        try {
          window.parent.removeEventListener('jalrakshak:open_palette', window.parent.__jalCmdCustomOpen);
        } catch(e) {}
      }
      window.parent.__jalCmdCustomOpen = handleCustomOpen;
      try {
        window.parent.addEventListener('jalrakshak:open_palette', handleCustomOpen);
      } catch(e) {}

    })();
    </script>
    """
    html_code = (
        template
        .replace("__SPILL_LAT__", f"{spill_lat:.6f}")
        .replace("__SPILL_LON__", f"{spill_lon:.6f}")
        .replace("__SOURCE_LAT__", f"{source_lat:.6f}")
        .replace("__SOURCE_LON__", f"{source_lon:.6f}")
    )
    components.html(html_code, height=0, width=0)


def render_telemetry_card(label: str, value: str, sub: str = "", tag: str = "", border_color: str = None, value_color: str = None) -> str:
    """Render standardized mission-control glassmorphic telemetry card."""
    tag_html = f'<span class="telemetry-chip">{tag}</span>' if tag else ""
    border_style = f"border-color: {border_color};" if border_color else ""
    val_style = f"color: {value_color};" if value_color else ""
    sub_html = f'<div class="telemetry-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card glass-panel" style="{border_style}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="telemetry-label">{label}</span>
            {tag_html}
        </div>
        <div class="telemetry-value" style="{val_style}">{value}</div>
        {sub_html}
    </div>
    """


def render_status_pill(status: str) -> str:
    """Render standardized validation status pill."""
    if not status:
        return '<span class="status-badge badge-inconclusive">UNKNOWN</span>'
    s_upper = status.upper()
    if "CONFIRMED" in s_upper:
        css = "badge-confirmed"
        icon = "●"
    elif "PROBABLE" in s_upper:
        css = "badge-probable"
        icon = "●"
    elif "LOOK-ALIKE" in s_upper:
        css = "badge-lookalike"
        icon = "▲"
    elif "REJECTED" in s_upper:
        css = "badge-rejected"
        icon = "✕"
    else:
        css = "badge-inconclusive"
        icon = "○"
    return f'<span class="status-badge {css}">{icon} {status}</span>'


def generate_sar_layer_image(image_path: str, layer: str, final_state: dict):
    """
    Generate requested SAR visual layer on demand with zero-flicker memoization caching:
    - composite: crisp detection overlay with contours, bounding box, centroid crosshairs
    - raw: original radar backscatter amplitude
    - yolo: YOLO segmentation mask in magenta/red over dark base
    - landmask: terrestrial land in ochre/brown, ocean in navy, cyan coastline
    - classical: classical K-Means and adaptive threshold extraction
    - final: final consensus mask (green validated slick or rejected stamp)
    - diagnostics: full 6-panel composite matrix
    """
    if not image_path or not os.path.exists(image_path):
        return None

    # Check memoized session cache to eliminate redundant computation and image flickering
    if "sar_layer_cache" not in st.session_state:
        st.session_state["sar_layer_cache"] = {}

    all_coords = final_state.get("all_spill_coords", []) if final_state else []
    if not all_coords and final_state and final_state.get("spill_coords"):
        all_coords = [final_state["spill_coords"]]
    val_status_str = final_state.get("validation_status", "PROBABLE") if final_state else "PROBABLE"
    cache_key = f"{image_path}:{layer}:{len(all_coords)}:{val_status_str}"

    if cache_key in st.session_state["sar_layer_cache"]:
        return st.session_state["sar_layer_cache"][cache_key]

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None
    h, w = img_gray.shape[:2]

    # Build YOLO mask from actual pipeline coordinates
    yolo_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in all_coords:
        if poly and len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(yolo_mask, [pts], 255)

    if layer == "raw":
        st.session_state["sar_layer_cache"][cache_key] = img_gray
        return img_gray

    # Extract land mask
    lm = extract_land_mask(img_gray)
    land_mask = lm.land_mask

    # Run classical consensus validation
    try:
        res = validate_consensus(yolo_mask=yolo_mask, image=img_gray, land_mask=land_mask)
    except Exception:
        res = None

    if layer == "landmask":
        p_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        p_bgr[:] = [60, 25, 15]  # Deep navy ocean
        p_bgr[land_mask > 0] = [40, 140, 190]  # Ochre/brown landmass
        contours_l, _ = cv2.findContours(land_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p_bgr, contours_l, -1, (100, 230, 255), 2)
        out = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["sar_layer_cache"][cache_key] = out
        return out

    if layer == "yolo":
        p_bgr = cv2.cvtColor(img_gray // 2, cv2.COLOR_GRAY2BGR)
        p_bgr[yolo_mask > 0] = [255, 50, 80]
        contours_y, _ = cv2.findContours(yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p_bgr, contours_y, -1, (255, 220, 100), 2)
        out = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["sar_layer_cache"][cache_key] = out
        return out

    if layer == "classical":
        p_bgr = cv2.cvtColor(img_gray // 2, cv2.COLOR_GRAY2BGR)
        class_m = res.validated_mask if res and hasattr(res, "validated_mask") else yolo_mask
        p_bgr[class_m > 0] = [0, 220, 255]
        contours_c, _ = cv2.findContours(class_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p_bgr, contours_c, -1, (255, 255, 255), 1)
        out = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["sar_layer_cache"][cache_key] = out
        return out

    if layer == "final":
        p_bgr = cv2.cvtColor(img_gray // 3, cv2.COLOR_GRAY2BGR)
        val_m = res.validated_mask if res and hasattr(res, "validated_mask") else np.zeros((h, w), dtype=np.uint8)
        if np.sum(val_m > 0) > 0 and (final_state and final_state.get("validation_status") != "REJECTED"):
            p_bgr[val_m > 0] = [50, 220, 50]
            contours_v, _ = cv2.findContours(val_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p_bgr, contours_v, -1, (255, 255, 255), 2)
        else:
            cv2.putText(p_bgr, "NO VALIDATED SPILL / REJECTED", (max(10, w // 8), h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2)
        out = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["sar_layer_cache"][cache_key] = out
        return out

    if layer == "diagnostics":
        diag = generate_diagnostic_panels(
            image=img_gray,
            yolo_mask=yolo_mask,
            classical_mask=res.validated_mask if res else yolo_mask,
            land_mask=land_mask,
            validated_mask=res.validated_mask if res else yolo_mask,
            validation_status=val_status_str,
        )
        out = cv2.cvtColor(diag, cv2.COLOR_BGR2RGB)
        st.session_state["sar_layer_cache"][cache_key] = out
        return out

    # Default: Composite Detection Overlay
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(vis)
    if np.sum(yolo_mask > 0) > 0:
        overlay[yolo_mask > 0] = [239, 68, 68]
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.40, 0)
        contours, _ = cv2.findContours(yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 229, 255), 2)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (56, 189, 248), 1)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawMarker(vis, (cx, cy), (0, 229, 255), cv2.MARKER_CROSS, 16, 2)
    out = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    st.session_state["sar_layer_cache"][cache_key] = out
    return out


def build_investigation_map(state, slider_minutes=0, selected_vessel_mmsi=None, mode="ALL", focus_target=None, drift_hours=None):
    """
    Build the forensic Folium geospatial intelligence map with purposeful motion:
    - Smooth camera flyTo interpolation
    - Progressive backtrack path animation with hydrodynamic waypoints (TRACE SOURCE)
    - Progressive forward forecast animation with animated AntPath (SIMULATE DRIFT)
    - Expanded target vessel marker with radar halo and highlighted animated track
    - Soft pulsing spill boundary with credible source zone activation
    - Stable cinematic Drift Intelligence mode with deterministic advection & dispersion
    """
    if not state:
        state = {}

    spill_lat = state.get("spill_lat", DEMO_SPILL_LAT)
    spill_lon = state.get("spill_lon", DEMO_SPILL_LON)
    source_lat = state.get("source_lat", spill_lat)
    source_lon = state.get("source_lon", spill_lon)
    uncertainty_km = state.get("source_uncertainty_km", 5.0)

    # Ocean hydrodynamic parameters from hindcast_result
    hindcast = state.get("hindcast_result", {}) if state else {}
    current_speed = hindcast.get("current_speed_ms", 0.48)
    current_bearing = hindcast.get("current_bearing_deg", 118.0)
    wind_speed = hindcast.get("wind_speed_ms", 6.2)
    wind_bearing = hindcast.get("wind_bearing_deg", 135.0)
    wind_factor = hindcast.get("wind_factor", 0.03)

    # Combined drift vector (Euler numerical integration match)
    cx = current_speed * math.sin(math.radians(current_bearing))
    cy = current_speed * math.cos(math.radians(current_bearing))
    wx = wind_factor * wind_speed * math.sin(math.radians(wind_bearing))
    wy = wind_factor * wind_speed * math.cos(math.radians(wind_bearing))
    vx = cx + wx
    vy = cy + wy
    net_drift_speed = math.sqrt(vx**2 + vy**2)
    net_drift_bearing = math.degrees(math.atan2(vx, vy)) % 360
    reverse_bearing = (net_drift_bearing + 180) % 360
    net_speed_knots = net_drift_speed * 1.94384

    # Corridor endpoints: T-12h origin and T+24h forecast
    d_m12 = net_drift_speed * (12.0 * 3600) / 1000.0
    lat_m12, lon_m12 = destination_point(spill_lat, spill_lon, reverse_bearing, d_m12)
    d_p24 = net_drift_speed * (24.0 * 3600) / 1000.0
    lat_p24, lon_p24 = destination_point(spill_lat, spill_lon, net_drift_bearing, d_p24)
    mid_lat = (lat_m12 + lat_p24) / 2.0
    mid_lon = (lon_m12 + lon_p24) / 2.0

    # Temporal reference for dynamic positioning
    ts = state.get("detection_timestamp")
    base_time = datetime.fromisoformat(ts) if ts else datetime(2026, 9, 14, 15, 30, tzinfo=timezone.utc)
    slider_time = base_time + timedelta(minutes=slider_minutes)

    # Determine candidate vessel positions for dynamic camera tracking
    tracks_data = state.get("ais_tracks", {}) or {}
    candidates = state.get("candidate_scores", []) or []
    ranking_map = {c["mmsi"]: c for c in candidates}

    target_vessel_pos = None
    if selected_vessel_mmsi and tracks_data:
        recs = []
        if isinstance(tracks_data, dict) and selected_vessel_mmsi in tracks_data:
            recs = tracks_data[selected_vessel_mmsi]
        elif isinstance(tracks_data, list):
            for t in tracks_data:
                if str(t.get("mmsi")) == str(selected_vessel_mmsi):
                    recs = t.get("points", [])
                    break
        if recs and isinstance(recs[0], dict):
            target_vessel_pos = recs[0]
            min_dt = float("inf")
            for r in recs:
                if isinstance(r, dict) and "timestamp" in r:
                    try:
                        r_time = datetime.fromisoformat(r["timestamp"])
                        dt = abs((r_time - slider_time).total_seconds())
                        if dt < min_dt:
                            min_dt = dt
                            target_vessel_pos = r
                    except Exception:
                        pass

    # Determine intelligent camera centering and zoom level
    center_lat, center_lon = source_lat, source_lon
    zoom_val = 11

    if mode == "DRIFT":
        center_lat, center_lon = mid_lat, mid_lon
        zoom_val = 11
    elif target_vessel_pos and (focus_target == "vessel" or (focus_target is None and selected_vessel_mmsi)):
        center_lat, center_lon = float(target_vessel_pos["lat"]), float(target_vessel_pos["lon"])
        zoom_val = 13
    elif focus_target == "spill" or mode == "SPILL":
        center_lat, center_lon = spill_lat, spill_lon
        zoom_val = 13
    elif focus_target == "source" or mode in ("SOURCE", "BACKTRACK"):
        center_lat, center_lon = source_lat, source_lon
        zoom_val = 12

    # Reliable basemap with zero watermark errors
    carto_key = CARTO_API_KEY or os.getenv("CARTO_API_KEY", "")
    use_carto = bool(carto_key) and (MAP_BASEMAP.lower() in ("cartodb_dark", "cartodb dark_matter", "cartodb"))

    if use_carto:
        tiles_url = f"https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png?api_key={carto_key}"
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_val, tiles=None, control_scale=True)
        folium.TileLayer(
            tiles=tiles_url,
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name="CartoDB Dark Matter",
        ).add_to(fmap)
    else:
        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_val,
            tiles="OpenStreetMap",
            control_scale=True,
        )

    Fullscreen(position="topright").add_to(fmap)

    # Interactive polygon marking plugin for operator region queries
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": True,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
                "shapeOptions": {
                    "color": "#00e5ff",
                    "fillColor": "#00e5ff",
                    "fillOpacity": 0.20,
                    "weight": 2,
                },
            },
        },
        edit_options={"edit": False, "remove": True},
    ).add_to(fmap)

    # ──────────────────────────────────────────────────────────
    # MODE DRIFT: CINEMATIC BUT SCIENTIFICALLY RESTRAINED DISPLAY
    # ──────────────────────────────────────────────────────────
    if mode == "DRIFT":
        fg_drift_origin = folium.FeatureGroup(name="📍 Spill Origin (T-12h)", show=True)
        fg_drift_history = folium.FeatureGroup(name="⏱️ Historical Trajectory (T-12h → NOW)", show=True)
        fg_drift_forecast = folium.FeatureGroup(name="🌊 Predicted Trajectory (NOW → T+24h)", show=True)
        fg_drift_vector = folium.FeatureGroup(name="🧭 Current Drift Vector", show=True)
        fg_drift_active = folium.FeatureGroup(name="🎯 Active Slick & Dispersion Region", show=True)
        fg_coastal = folium.FeatureGroup(name="🛡️ Coastal Protection Assets", show=True)

        h_val = float(drift_hours if drift_hours is not None else (slider_minutes / 60.0))

        # 1. Spill Origin at T-12h
        unc_m12 = 0.8 + d_m12 * 0.35
        folium.Circle(
            location=[lat_m12, lon_m12],
            radius=unc_m12 * 1000,
            color="#f59e0b",
            fill=True,
            fill_color="#f59e0b",
            fill_opacity=0.14,
            dash_array="6 4",
            tooltip=f"Spill Origin Boundary (T-12h Hindcast Uncertainty: ±{unc_m12:.1f} km)",
            popup=f"<b>Spill Origin Region</b><br>Horizon: T-12h<br>Coordinates: [{lat_m12:.4f}°N, {lon_m12:.4f}°E]<br>Uncertainty: ±{unc_m12:.1f} km",
        ).add_to(fg_drift_origin)

        folium.CircleMarker(
            location=[lat_m12, lon_m12],
            radius=7,
            color="#f59e0b",
            weight=2,
            fill=True,
            fill_color="#00e5ff",
            fill_opacity=0.9,
            tooltip=f"Spill Origin Centroid (T-12h): {lat_m12:.4f}°N, {lon_m12:.4f}°E",
        ).add_to(fg_drift_origin)

        folium.Marker(
            location=[lat_m12, lon_m12],
            icon=folium.DivIcon(
                html='<div style="font-size:10px; font-weight:700; font-family:monospace; color:#f59e0b; background:rgba(10,15,28,0.92); border:1px solid #f59e0b; padding:2px 7px; border-radius:4px; margin-top:-32px; margin-left:-55px; white-space:nowrap; box-shadow:0 0 10px rgba(245,158,11,0.35);">▲ SPILL ORIGIN (T-12h)</div>'
            ),
        ).add_to(fg_drift_origin)

        # Also show model estimated source if distinct
        if abs(source_lat - lat_m12) > 0.005 or abs(source_lon - lon_m12) > 0.005:
            folium.CircleMarker(
                location=[source_lat, source_lon],
                radius=5,
                color="#fbbf24",
                fill=True,
                fill_color="#fbbf24",
                fill_opacity=0.6,
                tooltip=f"Estimated Source (Hindcast Model): {source_lat:.4f}°N, {source_lon:.4f}°E",
            ).add_to(fg_drift_origin)

        # 2. Historical / Backtracked Trajectory (T-12h -> NOW)
        hist_steps = 30
        hist_pts = []
        for s in range(hist_steps + 1):
            frac = s / hist_steps
            h_step = -12.0 * (1.0 - frac)
            d_step = net_drift_speed * (abs(h_step) * 3600) / 1000.0
            pt_lat, pt_lon = destination_point(spill_lat, spill_lon, reverse_bearing, d_step)
            hist_pts.append([pt_lat, pt_lon])

        AntPath(
            locations=hist_pts,
            color="#f59e0b",
            pulse_color="#00e5ff",
            weight=4,
            delay=650,
            dash_array=[8, 14],
            opacity=0.90,
            tooltip="Historical Backtracked Advection Trail (T-12h → NOW)",
        ).add_to(fg_drift_history)

        # Intermediate Milestone: T-6h
        d_m6 = net_drift_speed * (6.0 * 3600) / 1000.0
        lat_m6, lon_m6 = destination_point(spill_lat, spill_lon, reverse_bearing, d_m6)
        folium.CircleMarker(
            location=[lat_m6, lon_m6],
            radius=5.5,
            color="#f59e0b",
            weight=2,
            fill=True,
            fill_color="#38bdf8",
            fill_opacity=0.9,
            tooltip=f"Hindcast Milestone T-6h: {lat_m6:.4f}°N, {lon_m6:.4f}°E (Dist: {d_m6:.1f} km)",
        ).add_to(fg_drift_history)
        folium.Marker(
            location=[lat_m6, lon_m6],
            icon=folium.DivIcon(
                html='<div style="font-size:9.5px; font-weight:700; font-family:monospace; color:#38bdf8; background:rgba(10,15,28,0.85); border:1px solid rgba(56,189,248,0.4); padding:1px 5px; border-radius:3px; margin-top:-26px; margin-left:-18px; white-space:nowrap;">T-6h</div>'
            ),
        ).add_to(fg_drift_history)

        # Detection Point: NOW (T=0)
        folium.CircleMarker(
            location=[spill_lat, spill_lon],
            radius=7.5,
            color="#ef4444",
            weight=2.5,
            fill=True,
            fill_color="#ef4444",
            fill_opacity=1.0,
            tooltip=f"Observation Point: NOW (SAR Sensor Acquisition) [{spill_lat:.4f}°N, {spill_lon:.4f}°E]",
        ).add_to(fg_drift_history)
        folium.Marker(
            location=[spill_lat, spill_lon],
            icon=folium.DivIcon(
                html='<div style="font-size:10px; font-weight:700; font-family:monospace; color:#ef4444; background:rgba(10,15,28,0.92); border:1px solid #ef4444; padding:2px 6px; border-radius:4px; margin-top:-30px; margin-left:-32px; white-space:nowrap; box-shadow:0 0 10px rgba(239,68,68,0.4);">● NOW</div>'
            ),
        ).add_to(fg_drift_history)

        # 3. Predicted Trajectory (NOW -> T+24h)
        fc_steps = 48
        fc_pts = []
        for s in range(fc_steps + 1):
            frac = s / fc_steps
            h_step = 24.0 * frac
            d_step = net_drift_speed * (h_step * 3600) / 1000.0
            pt_lat, pt_lon = destination_point(spill_lat, spill_lon, net_drift_bearing, d_step)
            fc_pts.append([pt_lat, pt_lon])

        AntPath(
            locations=fc_pts,
            color="#8b5cf6",
            pulse_color="#ffffff",
            weight=4,
            delay=650,
            dash_array=[8, 16],
            opacity=0.90,
            tooltip="Predicted Forward Drift Trajectory (NOW → T+24h)",
        ).add_to(fg_drift_forecast)

        # Forecast Milestones: T+6h, T+12h, T+24h
        fc_milestones = [
            (6.0, "#0ea5e9", "T+6h"),
            (12.0, "#8b5cf6", "T+12h"),
            (24.0, "#f43f5e", "T+24h"),
        ]
        for f_h, f_col, f_lbl in fc_milestones:
            f_dist = net_drift_speed * (f_h * 3600) / 1000.0
            f_lat, f_lon = destination_point(spill_lat, spill_lon, net_drift_bearing, f_dist)
            f_unc = 0.8 + f_dist * 0.35

            folium.Circle(
                location=[f_lat, f_lon],
                radius=f_unc * 1000,
                color=f_col,
                fill=True,
                fill_color=f_col,
                fill_opacity=0.10,
                dash_array="4 4",
                tooltip=f"Projected Uncertainty Envelope at {f_lbl} (±{f_unc:.1f} km)",
            ).add_to(fg_drift_forecast)

            folium.CircleMarker(
                location=[f_lat, f_lon],
                radius=5.5 if f_h < 24 else 7,
                color=f_col,
                weight=2,
                fill=True,
                fill_color="#ffffff",
                fill_opacity=0.9,
                tooltip=f"Forecast Milestone {f_lbl}: {f_lat:.4f}°N, {f_lon:.4f}°E (Dist: {f_dist:.1f} km, Unc: ±{f_unc:.1f} km)",
            ).add_to(fg_drift_forecast)

            folium.Marker(
                location=[f_lat, f_lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:9.5px; font-weight:700; font-family:monospace; color:{f_col}; background:rgba(10,15,28,0.85); border:1px solid {f_col}; padding:1px 5px; border-radius:3px; margin-top:-26px; margin-left:-18px; white-space:nowrap;">{f_lbl}</div>'
                ),
            ).add_to(fg_drift_forecast)

        # 4. Active Slick Centroid & Projected Dispersion Region at current scrubbed hour
        if abs(h_val) < 0.01:
            active_lat, active_lon = spill_lat, spill_lon
            active_cum_dist = 0.0
            active_uncertainty = 0.8
            color_active = "#ef4444"
        elif h_val < 0.0:
            active_cum_dist = net_drift_speed * (abs(h_val) * 3600) / 1000.0
            active_lat, active_lon = destination_point(spill_lat, spill_lon, reverse_bearing, active_cum_dist)
            active_uncertainty = 0.8 + active_cum_dist * 0.35
            color_active = "#f59e0b"
        else:
            active_cum_dist = net_drift_speed * (h_val * 3600) / 1000.0
            active_lat, active_lon = destination_point(spill_lat, spill_lon, net_drift_bearing, active_cum_dist)
            active_uncertainty = 0.8 + active_cum_dist * 0.35
            color_active = "#8b5cf6"

        # Projected dispersion region circle
        folium.Circle(
            location=[active_lat, active_lon],
            radius=active_uncertainty * 1000,
            color=color_active,
            fill=True,
            fill_color=color_active,
            fill_opacity=0.22,
            weight=2,
            dash_array="5 4",
            tooltip=f"Projected Dispersion Envelope at T{h_val:+0.1f}h (Radius: ±{active_uncertainty:.1f} km)",
            popup=f"<b>Projected Slick Dispersion Zone</b><br>Horizon: T{h_val:+0.1f}h<br>Coordinates: [{active_lat:.4f}°N, {active_lon:.4f}°E]<br>Dispersion Radius: ±{active_uncertainty:.1f} km",
        ).add_to(fg_drift_active)

        # Outer pulsing radar halo
        folium.CircleMarker(
            location=[active_lat, active_lon],
            radius=22,
            color="#00e5ff",
            weight=2,
            dash_array="3 3",
            fill=True,
            fill_color="#00e5ff",
            fill_opacity=0.12,
            tooltip="Active Slick Radar Halo",
        ).add_to(fg_drift_active)

        # Core slick marker
        folium.CircleMarker(
            location=[active_lat, active_lon],
            radius=9,
            color="#ffffff",
            weight=2.5,
            fill=True,
            fill_color="#00e5ff" if h_val > 0 else ("#ef4444" if abs(h_val) < 0.01 else "#f59e0b"),
            fill_opacity=1.0,
            tooltip=f"Active Slick Centroid (T{h_val:+0.1f}h): {active_lat:.4f}°N, {active_lon:.4f}°E",
        ).add_to(fg_drift_active)

        # Tactical callout label
        folium.Marker(
            location=[active_lat, active_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px; font-weight:700; font-family:monospace; color:#00e5ff; background:rgba(10,15,28,0.92); border:1px solid #00e5ff; padding:2px 7px; border-radius:4px; margin-top:-36px; margin-left:-52px; white-space:nowrap; box-shadow:0 0 12px rgba(0,229,255,0.45);">T{h_val:+0.1f}h SLICK [{active_lat:.3f}°N, {active_lon:.3f}°E] (±{active_uncertainty:.1f}km)</div>'
            ),
        ).add_to(fg_drift_active)

        # 5. Current Drift Vector Arrow
        vec_len_km = 3.2
        vec_end_lat, vec_end_lon = destination_point(active_lat, active_lon, net_drift_bearing, vec_len_km)

        folium.PolyLine(
            locations=[[active_lat, active_lon], [vec_end_lat, vec_end_lon]],
            color="#00e5ff",
            weight=4,
            opacity=0.95,
            tooltip=f"Current Drift Vector: {net_drift_speed:.2f} m/s ({net_speed_knots:.2f} kn) @ {net_drift_bearing:.0f}° True",
        ).add_to(fg_drift_vector)

        # Tactical vector arrowhead
        folium.Marker(
            location=[vec_end_lat, vec_end_lon],
            icon=folium.DivIcon(
                html=f'''
                <div style="transform: rotate({net_drift_bearing}deg); transform-origin: center; margin-top:-12px; margin-left:-12px;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2L19 19L12 15L5 19L12 2Z" fill="#00e5ff" stroke="#ffffff" stroke-width="1.5"/>
                    </svg>
                </div>
                '''
            ),
        ).add_to(fg_drift_vector)

        # Tactical vector label
        folium.Marker(
            location=[vec_end_lat, vec_end_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9.5px; font-weight:700; font-family:monospace; color:#38bdf8; background:rgba(10,15,28,0.85); border:1px solid rgba(56,189,248,0.4); padding:1px 5px; border-radius:3px; margin-top:14px; margin-left:-38px; white-space:nowrap;">V_NET: {net_drift_speed:.2f} m/s ({net_speed_knots:.2f} kn) @ {net_drift_bearing:.0f}°T</div>'
            ),
        ).add_to(fg_drift_vector)

        # 6. Sensitive Coastal Infrastructure
        for sa in (CHENNAI_SCENARIO.sensitive_areas or []):
            icon_sym = "🐦" if sa.get("type") == "wildlife" else ("⚓" if sa.get("type") == "port" else "🎣")
            folium.Marker(
                location=[sa["lat"], sa["lon"]],
                icon=folium.DivIcon(html=f'<div style="font-size:15px;">{icon_sym}</div>'),
                tooltip=f"{sa['name']} ({sa.get('type', 'Asset')})",
                popup=f"<b>{sa['name']}</b><br>Type: {sa.get('type')}<br>Coordinates: [{sa['lat']}°N, {sa['lon']}°E]",
            ).add_to(fg_coastal)

        fg_drift_origin.add_to(fmap)
        fg_drift_history.add_to(fmap)
        fg_drift_forecast.add_to(fmap)
        fg_drift_vector.add_to(fmap)
        fg_drift_active.add_to(fmap)
        fg_coastal.add_to(fmap)

        folium.LayerControl(position="topright", collapsed=True).add_to(fmap)

        # Stable camera: camera does NOT fly or jump on scrubber movements
        return fmap

    # Feature Groups for clean layer control and mode filtering (All other modes)
    fg_spill = folium.FeatureGroup(name="🛢️ Detected Spill Slick", show=(mode in ["ALL", "SPILL", "RISK"]))
    fg_source = folium.FeatureGroup(name="🎯 Probable Source Region", show=(mode in ["ALL", "SOURCE", "BACKTRACK"]))
    fg_hindcast = folium.FeatureGroup(name="⏱️ Hindcast Backtrack", show=(mode in ["ALL", "BACKTRACK", "SOURCE"]))
    fg_forecast = folium.FeatureGroup(name="🌊 Forward Drift Forecast", show=(mode in ["ALL", "FORWARD DRIFT", "RISK"]))
    fg_ais = folium.FeatureGroup(name="🚢 AIS Fleet Tracks", show=(mode in ["ALL", "AIS", "SOURCE"]))
    fg_vessels = folium.FeatureGroup(name="⚓ Vessel Positions", show=(mode in ["ALL", "AIS"]))
    fg_coastal = folium.FeatureGroup(name="🛡️ Coastal Protection Assets", show=(mode in ["ALL", "RISK"]))

    # 1. Detected Spill Polygon & Centroid
    spill_detected = state.get("spill_detected", False)
    all_coords = state.get("all_spill_coords", [])
    sar_meta = state.get("sar_metadata", {})
    bbox = sar_meta.get("bbox") if sar_meta else None
    dims = sar_meta.get("dimensions", [512, 512]) if sar_meta else [512, 512]

    if spill_detected:
        polygon_drawn = False
        if all_coords and bbox and len(bbox) == 4:
            min_lat, min_lon, max_lat, max_lon = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            h, w = dims[0], dims[1]
            for poly in all_coords:
                if poly and len(poly) >= 3:
                    geo_poly = []
                    for pt in poly:
                        px_x, px_y = float(pt[0]), float(pt[1])
                        pt_lat = max_lat - (px_y / h) * (max_lat - min_lat)
                        pt_lon = min_lon + (px_x / w) * (max_lon - min_lon)
                        geo_poly.append([pt_lat, pt_lon])
                    folium.Polygon(
                        locations=geo_poly,
                        color="#ef4444",
                        weight=2,
                        fill=True,
                        fill_color="#ef4444",
                        fill_opacity=0.45,
                        tooltip="Detected Oil Spill Polygon",
                        popup=f"<b>Detected Oil Slick</b><br>Centroid: {spill_lat:.4f}°N, {spill_lon:.4f}°E",
                    ).add_to(fg_spill)
                    polygon_drawn = True

        if not polygon_drawn:
            folium.CircleMarker(
                location=[spill_lat, spill_lon],
                radius=14,
                color="#ef4444",
                fill=True,
                fill_color="#ef4444",
                fill_opacity=0.40,
                tooltip="Detected Oil Spill Boundary",
                popup=f"<b>Detected Oil Spill</b><br>Centroid: {spill_lat:.4f}°N, {spill_lon:.4f}°E",
            ).add_to(fg_spill)

        # Soft pulsing outer ring around spill centroid
        folium.CircleMarker(
            location=[spill_lat, spill_lon],
            radius=26,
            color="#ef4444",
            weight=1.5,
            dash_array="4 4",
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.12,
            tooltip="Slick Centroid Soft Pulse",
        ).add_to(fg_spill)

        folium.Marker(
            location=[spill_lat, spill_lon],
            icon=folium.DivIcon(html='<div style="font-size:18px;">🛢️</div>'),
            tooltip=f"Spill Location: {spill_lat:.4f}°N, {spill_lon:.4f}°E",
            popup=f"<b>Oil Spill Centroid</b><br>[{spill_lat:.4f}°N, {spill_lon:.4f}°E]",
        ).add_to(fg_spill)

    # 2. Origin Likelihood Surface & Credible Zones
    source_prob = state.get("source_probability", {})
    credible_zones = source_prob.get("credible_zones", [])
    if credible_zones:
        for cz in reversed(credible_zones):
            pts = cz.get("polygon_points", [])
            if pts:
                folium.Polygon(
                    locations=pts,
                    color=cz.get("color_hex", "#f59e0b"),
                    weight=1.5,
                    fill=True,
                    fill_color=cz.get("color_hex", "#f59e0b"),
                    fill_opacity=cz.get("fill_opacity", 0.16),
                    tooltip=f"{cz.get('name')}: {cz.get('description')}",
                    popup=f"<b>{cz.get('name')}</b><br>{cz.get('description')}",
                ).add_to(fg_source)
    else:
        folium.Circle(
            location=[source_lat, source_lon],
            radius=uncertainty_km * 1000,
            color="#f59e0b",
            fill=True,
            fill_color="#f59e0b",
            fill_opacity=0.15,
            dash_array="8 5",
            tooltip=f"Estimated Origin Zone (±{uncertainty_km:.1f} km)",
            popup=f"<b>Estimated Origin Uncertainty Zone</b><br>Radius: ±{uncertainty_km:.1f} km",
        ).add_to(fg_source)

    folium.Marker(
        location=[source_lat, source_lon],
        icon=folium.DivIcon(
            html='<div style="font-size:11px; color:#f59e0b; font-weight:700; white-space:nowrap; background:rgba(11,17,32,0.85); border:1px solid rgba(245,158,11,0.4); padding:2px 6px; border-radius:4px;">▲ ORIGIN ESTIMATE</div>'
        ),
        popup=f"<b>Estimated Origin</b><br>[{source_lat:.4f}°N, {source_lon:.4f}°E]<br>Uncertainty: ±{uncertainty_km:.1f} km",
    ).add_to(fg_source)

    # 3. Progressive Hindcast Backtrack Vector (Spill -> Origin with Interpolated Waypoints)
    backtrack_waypoints = []
    num_steps = 15
    for s in range(num_steps + 1):
        frac = s / num_steps
        # Hydrodynamic drift curvature
        curv = np.sin(frac * np.pi) * 0.0035
        pt_lat = spill_lat + frac * (source_lat - spill_lat) + curv
        pt_lon = spill_lon + frac * (source_lon - spill_lon) - curv * 0.4
        backtrack_waypoints.append([pt_lat, pt_lon])

    AntPath(
        locations=backtrack_waypoints,
        color="#f59e0b",
        pulse_color="#00e5ff",
        weight=4,
        delay=600,
        dash_array=[10, 16],
        opacity=0.95,
        tooltip="Progressive Reverse Drift Backtrack Trail (Spill → Origin)",
    ).add_to(fg_hindcast)

    prog_times = [
        ("T-45m", backtrack_waypoints[4], "Estimated Slick Position (-45 min)"),
        ("T-90m", backtrack_waypoints[8], "Estimated Slick Position (-90 min)"),
        ("T-135m", backtrack_waypoints[11], "Estimated Slick Position (-135 min)"),
        ("T-180m (Origin)", [source_lat, source_lon], "Estimated Origin Slick Position (-180 min)"),
    ]
    for lbl, pos, desc in prog_times:
        folium.CircleMarker(
            location=pos,
            radius=4.5,
            color="#f59e0b",
            fill=True,
            fill_color="#00e5ff",
            fill_opacity=0.9,
            tooltip=f"{lbl}: {desc}",
        ).add_to(fg_hindcast)

    # 4. Progressive Forward Forecast Cones with Animated AntPath
    forecasts = state.get("forecast_results", []) or []
    fc_colors = ["#0ea5e9", "#8b5cf6", "#f43f5e"]
    for i, fc in enumerate(forecasts):
        dest_lat = fc.get("destination_lat")
        dest_lon = fc.get("destination_lon")
        if dest_lat and dest_lon:
            color = fc_colors[i % len(fc_colors)]
            fc_waypoints = []
            f_steps = 15
            for s in range(f_steps + 1):
                frac = s / f_steps
                curv = np.sin(frac * np.pi) * 0.003
                pt_lat = spill_lat + frac * (dest_lat - spill_lat) + curv
                pt_lon = spill_lon + frac * (dest_lon - spill_lon) + curv * 0.5
                fc_waypoints.append([pt_lat, pt_lon])

            AntPath(
                locations=fc_waypoints,
                color=color,
                pulse_color="#ffffff",
                weight=3.5,
                delay=750,
                dash_array=[8, 16],
                opacity=0.90,
                tooltip=f"Forward Drift Simulation: {fc.get('direction', '').replace('_', ' ').title()} ({fc.get('hours', 0)}h Horizon)",
            ).add_to(fg_forecast)

            folium.CircleMarker(
                location=[dest_lat, dest_lon],
                radius=6, color=color, fill=True, fill_opacity=0.75,
                tooltip=f"Projected Position at +{fc.get('hours', 0)}h Horizon: {dest_lat:.4f}°N, {dest_lon:.4f}°E",
                popup=f"<b>Predicted Slick Position</b><br>+{fc.get('hours', 0)}h Horizon<br>[{dest_lat:.4f}°N, {dest_lon:.4f}°E]<br>ETA: +{fc.get('hours', 0)} hrs",
            ).add_to(fg_forecast)

    # 5. Sensitive Coastal Infrastructure
    for sa in (CHENNAI_SCENARIO.sensitive_areas or []):
        icon_sym = "🐦" if sa.get("type") == "wildlife" else ("⚓" if sa.get("type") == "port" else "🎣")
        folium.Marker(
            location=[sa["lat"], sa["lon"]],
            icon=folium.DivIcon(html=f'<div style="font-size:15px;">{icon_sym}</div>'),
            tooltip=f"{sa['name']} ({sa.get('type', 'Asset')})",
            popup=f"<b>{sa['name']}</b><br>Type: {sa.get('type')}<br>Coordinates: [{sa['lat']}°N, {sa['lon']}°E]",
        ).add_to(fg_coastal)

    # 6. Candidate Vessel Tracks & Dynamic Positioning
    ts = state.get("detection_timestamp")
    base_time = datetime.fromisoformat(ts) if ts else datetime(2026, 9, 14, 15, 30, tzinfo=timezone.utc)
    slider_time = base_time + timedelta(minutes=slider_minutes)

    v_palette = ["#38bdf8", "#fbbf24", "#f87171", "#a78bfa", "#34d399"]

    if isinstance(tracks_data, dict):
        track_items = tracks_data.items()
    elif isinstance(tracks_data, list):
        track_items = [(t.get("mmsi", str(i)), t.get("points", [])) for i, t in enumerate(tracks_data)]
    else:
        track_items = []

    def _simplify_track_pts(pts, max_pts=60):
        """Subsample dense polyline coordinates to prevent SVG/DOM bloat while preserving geometry."""
        if not pts or len(pts) <= max_pts:
            return pts
        step = (len(pts) - 1) / (max_pts - 1)
        indices = [int(round(i * step)) for i in range(max_pts)]
        seen = set()
        clean = []
        for idx in indices:
            if idx not in seen and idx < len(pts):
                seen.add(idx)
                clean.append(pts[idx])
        return clean

    dense_fleet = len(track_items) > 15
    fleet_cluster = MarkerCluster(name="🚢 Dense Fleet Clusters", overlay=True, control=False) if dense_fleet else None
    if fleet_cluster:
        fleet_cluster.add_to(fg_vessels)

    for idx, (mmsi, recs) in enumerate(track_items):
        if not recs:
            continue
        v_color = v_palette[idx % len(v_palette)]
        is_selected = (str(mmsi) == str(selected_vessel_mmsi))
        rank_info = ranking_map.get(mmsi, {})
        score = rank_info.get("score", 0)
        vessel_name = recs[0].get("name", mmsi) if isinstance(recs[0], dict) else mmsi

        raw_track_pts = [[r["lat"], r["lon"]] for r in recs if isinstance(r, dict) and "lat" in r and "lon" in r]
        track_pts = _simplify_track_pts(raw_track_pts, max_pts=60)
        if track_pts:
            # Interpolate position nearest to slider_time
            v_pos = recs[0]
            min_dt = float("inf")
            for r in recs:
                if isinstance(r, dict) and "timestamp" in r:
                    try:
                        r_time = datetime.fromisoformat(r["timestamp"])
                        dt = abs((r_time - slider_time).total_seconds())
                        if dt < min_dt:
                            min_dt = dt
                            v_pos = r
                    except Exception:
                        pass

            if is_selected:
                # Expanded marker with outer pulsing radar halo (Always unclustered for focus)
                folium.CircleMarker(
                    location=[v_pos["lat"], v_pos["lon"]],
                    radius=22,
                    color="#00e5ff",
                    weight=2,
                    dash_array="3 3",
                    fill=True,
                    fill_color="#00e5ff",
                    fill_opacity=0.18,
                    tooltip=f"CANDIDATE TARGET: {vessel_name}",
                ).add_to(fg_vessels)

                folium.CircleMarker(
                    location=[v_pos["lat"], v_pos["lon"]],
                    radius=12,
                    color="#ffffff",
                    weight=3,
                    fill=True,
                    fill_color="#00e5ff",
                    fill_opacity=1.0,
                    tooltip=f"🚢 {vessel_name} (MMSI: {mmsi}) | Speed: {v_pos.get('speed_knots', 0)} kn | Association: {score:.0f}/100",
                    popup=f"<b>PRIMARY CANDIDATE VESSEL</b><br><b>{vessel_name}</b><br>MMSI: {mmsi}<br>Speed: {v_pos.get('speed_knots', 0)} kn<br>Course: {v_pos.get('heading', v_pos.get('course_over_ground', 'N/A'))}°<br>Association Score: {score:.0f}/100",
                ).add_to(fg_vessels)

                folium.Marker(
                    location=[v_pos["lat"], v_pos["lon"]],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:10px; font-weight:700; font-family:monospace; color:#00e5ff; background:rgba(10,15,28,0.92); border:1px solid #00e5ff; padding:2px 6px; border-radius:4px; margin-top:-34px; margin-left:-30px; white-space:nowrap; box-shadow:0 0 12px rgba(0,229,255,0.45);">TARGET: {vessel_name} ({score:.0f}/100)</div>'
                    )
                ).add_to(fg_vessels)

                # Animated highlighted track for selected vessel
                AntPath(
                    locations=track_pts,
                    color="#00e5ff",
                    pulse_color="#ffffff",
                    weight=5,
                    delay=800,
                    opacity=1.0,
                    tooltip=f"Candidate Trajectory: {vessel_name} (Association: {score:.0f}/100)",
                ).add_to(fg_ais)

                # Origin connection vector to probable source
                folium.PolyLine(
                    locations=[[v_pos["lat"], v_pos["lon"]], [source_lat, source_lon]],
                    color="#10b981", weight=2.5, dash_array="4 4", opacity=0.90,
                    tooltip=f"Source Proximity Vector: {haversine_km(v_pos['lat'], v_pos['lon'], source_lat, source_lon):.1f} km",
                ).add_to(fg_vessels)
            elif selected_vessel_mmsi:
                # Dim unrelated vessels when a candidate is selected
                target_group = fleet_cluster if (dense_fleet and fleet_cluster) else fg_vessels
                folium.CircleMarker(
                    location=[v_pos["lat"], v_pos["lon"]],
                    radius=4,
                    color="#64748b",
                    weight=1,
                    fill=True,
                    fill_color="#334155",
                    fill_opacity=0.20,
                    tooltip=f"🚢 {vessel_name} (Dimmed) | MMSI: {mmsi} | Association: {score:.0f}/100",
                    popup=f"<b>{vessel_name}</b><br>MMSI: {mmsi}<br>Association Score: {score:.0f}/100<br><i>(Non-selected candidate)</i>",
                ).add_to(target_group)

                folium.PolyLine(
                    locations=track_pts,
                    color="#475569",
                    weight=1.2,
                    opacity=0.14,
                    dash_array="4 6",
                    tooltip=f"{vessel_name} (Dimmed Trajectory)",
                ).add_to(fg_ais)
            else:
                # Normal fleet display when no candidate is specifically isolated
                target_group = fleet_cluster if (dense_fleet and fleet_cluster) else fg_vessels
                folium.CircleMarker(
                    location=[v_pos["lat"], v_pos["lon"]],
                    radius=6,
                    color=v_color,
                    fill=True,
                    fill_color=v_color,
                    fill_opacity=0.75,
                    tooltip=f"🚢 {vessel_name} | {v_pos.get('speed_knots', 0)} kn | Association: {score:.0f}/100",
                    popup=f"<b>{vessel_name}</b><br>MMSI: {mmsi}<br>Speed: {v_pos.get('speed_knots', 0)} kn<br>Association Score: {score:.0f}/100",
                ).add_to(target_group)

                folium.PolyLine(
                    locations=track_pts,
                    color=v_color,
                    weight=2,
                    opacity=0.45,
                    dash_array="6 3",
                    tooltip=f"{vessel_name} (Association: {score:.0f}/100)",
                ).add_to(fg_ais)

    fg_spill.add_to(fmap)
    fg_source.add_to(fmap)
    fg_hindcast.add_to(fmap)
    fg_forecast.add_to(fmap)
    fg_ais.add_to(fmap)
    fg_vessels.add_to(fmap)
    fg_coastal.add_to(fmap)

    # Render active operator drawn query region if present
    active_poly = state.get("drawn_polygon")
    if not active_poly and "drawn_polygon" in st.session_state:
        active_poly = st.session_state.get("drawn_polygon")
    if active_poly and isinstance(active_poly, dict) and "coordinates" in active_poly:
        ring = active_poly["coordinates"]
        poly_pts = [[p[1], p[0]] for p in ring]
        fg_query_region = folium.FeatureGroup(name="📐 Active Selected Region", show=True)
        folium.Polygon(
            locations=poly_pts,
            color="#00e5ff",
            weight=2.5,
            fill=True,
            fill_color="#00e5ff",
            fill_opacity=0.22,
            dash_array="5 5",
            tooltip=f"Selected Region ({active_poly.get('area_km2', 0):.2f} km²)",
            popup=f"<b>Selected Region</b><br>Area: {active_poly.get('area_km2', 0):.2f} km²<br>Bounds: [{active_poly.get('min_lat', 0):.4f}, {active_poly.get('min_lon', 0):.4f}] to [{active_poly.get('max_lat', 0):.4f}, {active_poly.get('max_lon', 0):.4f}]",
        ).add_to(fg_query_region)
        fg_query_region.add_to(fmap)

    folium.LayerControl(position="topright", collapsed=True).add_to(fmap)

    # Smooth camera flyTo transition script
    fly_script = f"""
    <script>
    (function() {{
        function tryFly() {{
            try {{
                var mapEl = document.querySelector('.folium-map');
                if (mapEl && window[mapEl.id]) {{
                    var map = window[mapEl.id];
                    map.flyTo([{center_lat}, {center_lon}], {zoom_val}, {{
                        duration: 1.2,
                        easeLinearity: 0.25
                    }});
                }} else {{
                    setTimeout(tryFly, 80);
                }}
            }} catch(e) {{}}
        }}
        if (document.readyState === 'complete') {{
            setTimeout(tryFly, 100);
        }} else {{
            window.addEventListener('load', function() {{ setTimeout(tryFly, 100); }});
        }}
    }})();
    </script>
    """
    fmap.get_root().html.add_child(folium.Element(fly_script))

    return fmap


def render_evidence_panel(final_state, selected_mmsi=None, key_prefix=""):
    """Render the synchronized evidence panel for candidate vessels or oil spill consensus."""
    st.markdown("#### ⚖️ Evidence & Consensus Panel")

    if not final_state:
        st.info("Execute pipeline to compute multi-signal evidence metrics.")
        if st.button("Run Pipeline Now", key=f"{key_prefix}_btn_run_empty", type="primary", use_container_width=True):
            st.session_state["trigger_pipeline_run"] = True
            st.rerun()
        return

    candidates = final_state.get("candidate_scores", [])
    selected_cand = None
    if selected_mmsi:
        for c in candidates:
            if str(c.get("mmsi")) == str(selected_mmsi):
                selected_cand = c
                break

    # If a specific vessel is selected, prioritize Vessel Evidence Inspector
    if selected_cand:
        score = selected_cand.get("score", 0.0)
        score_css = "vessel-score-high" if score >= 70 else ("vessel-score-med" if score >= 40 else "vessel-score-low")
        bar_color = "#ef4444" if score >= 70 else ("#f59e0b" if score >= 40 else "#64748b")
        bdown = selected_cand.get("breakdown", {})

        st.markdown(f"""
        <div class="vessel-card glass-panel" style="margin-bottom:14px; border:1px solid rgba(0,229,255,0.40); animation: slide-in-evidence 0.32s cubic-bezier(0.16, 1, 0.3, 1);">
            <div class="vessel-header">
                <div>
                    <span class="telemetry-micro-label">CANDIDATE TARGET</span>
                    <div class="vessel-name">🚢 {selected_cand.get('name', 'UNKNOWN')}</div>
                    <span class="vessel-mmsi">MMSI: {selected_cand.get('mmsi')}</span>
                </div>
                <div style="text-align:right;">
                    <span class="telemetry-micro-label">ASSOCIATION SCORE</span>
                    <div class="vessel-score {score_css}">{score:.0f}<span style="font-size:12px; color:#64748b;">/100</span></div>
                </div>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{score}%; background:{bar_color};"></div>
            </div>
            <div class="telemetry-grid-4">
                <div class="telemetry-metric-unit">
                    <span class="telemetry-label">SOURCE PROXIMITY</span>
                    <strong class="telemetry-value-sm">{selected_cand.get('min_distance_km', 0.0):.1f} KM</strong>
                </div>
                <div class="telemetry-metric-unit">
                    <span class="telemetry-label">TEMPORAL WINDOW</span>
                    <strong class="telemetry-value-sm" style="color:{'#34d399' if selected_cand.get('time_match') else '#f87171'};">
                        {'COINCIDENT' if selected_cand.get('time_match') else 'OUTSIDE WINDOW'}
                    </strong>
                </div>
                <div class="telemetry-metric-unit">
                    <span class="telemetry-label">TRAJECTORY CONSISTENCY</span>
                    <strong class="telemetry-value-sm">{bdown.get('trajectory', 0.0):.0f}%</strong>
                </div>
                <div class="telemetry-metric-unit">
                    <span class="telemetry-label">DRIFT ALIGNMENT</span>
                    <strong class="telemetry-value-sm">{bdown.get('drift_consistency', 0.0):.0f}%</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✕ Deselect Vessel (View Spill Evidence)", key=f"{key_prefix}_btn_deselect", use_container_width=True):
            st.session_state["selected_vessel_mmsi"] = None
            st.rerun()

    # Main Spill Status Box
    val_status = final_state.get("validation_status", "PROBABLE" if final_state.get("spill_detected") else "REJECTED")
    val_res = normalize_validation_result(final_state.get("validation_result"))

    st.markdown(f"""
    <div class="glass-panel" style="padding:16px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="telemetry-label">FINAL VALIDATION CONSENSUS</span>
            {render_tag('VALIDATED')}
        </div>
        <div style="margin:8px 0 10px 0;">{render_status_pill(val_status)}</div>
        <div style="font-size:12px; color:#cbd5e1; line-height:1.45;">
            {val_res.get('explanation', 'Awaiting consensus evaluation.')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quantitative Evidence Gauges
    st.markdown("##### Multi-Signal Evidence")
    e1, e2 = st.columns(2)
    with e1:
        yolo_conf = final_state.get("detection_confidence", 0.0)
        st.markdown(f"""
        <div class="metric-card glass-panel">
            <div class="telemetry-label">AI SEGMENTATION</div>
            <div class="metric-value">{yolo_conf:.1%}</div>
            <div class="metric-sub">MODEL: YOLOV8N-SEG // MARITIME</div>
        </div>
        """, unsafe_allow_html=True)

        c_ratio = val_res.get("contrast_ratio", 1.0)
        damping_str = "Strong Damping" if c_ratio < 0.7 else ("Moderate" if c_ratio < 0.9 else "Low / Land")
        st.markdown(f"""
        <div class="metric-card glass-panel">
            <div class="telemetry-label">RADAR DAMPING</div>
            <div class="metric-value">{c_ratio:.2f}</div>
            <div class="metric-sub">RATIO: μ_SLICK / μ_SEA ({damping_str.upper()})</div>
        </div>
        """, unsafe_allow_html=True)

    with e2:
        c_agree = val_res.get("classical_agreement", 0.0)
        st.markdown(f"""
        <div class="metric-card glass-panel">
            <div class="telemetry-label">CLASSICAL SUPPORT</div>
            <div class="metric-value">{c_agree:.1%}</div>
            <div class="metric-sub">ALGORITHMS: ADAPTIVE + K-MEANS + DARK</div>
        </div>
        """, unsafe_allow_html=True)

        look_risk = val_res.get("look_alike_risk", 0.0)
        risk_tag = "Low" if look_risk < 0.3 else ("Elevated" if look_risk < 0.6 else "High Risk")
        st.markdown(f"""
        <div class="metric-card glass-panel">
            <div class="telemetry-label">LOOK-ALIKE RISK</div>
            <div class="metric-value">{look_risk:.1%}</div>
            <div class="metric-sub">PROBABILITY: {risk_tag.upper()} (WIND-CALM / FILM)</div>
        </div>
        """, unsafe_allow_html=True)

    # Physical Characterization
    if final_state.get("spill_detected"):
        st.markdown("##### Geometric & Physical Properties")
        char = final_state.get("characterization", {})
        area_sq_km = char.get("area_sq_km", final_state.get("spill_area_sq_km", 0.0))
        vol_tons = char.get("estimated_volume_tons", 0.0)
        perimeter = char.get("perimeter_km", 0.0)

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">EXTENT AREA</div>
                <div class="metric-value" style="font-size:19px;">{area_sq_km:.3f}</div>
                <div class="metric-sub">KM²</div>
            </div>
            """, unsafe_allow_html=True)
        with p2:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">ESTIMATED VOLUME</div>
                <div class="metric-value" style="font-size:19px;">{vol_tons:.1f}</div>
                <div class="metric-sub">METRIC TONS (FAY/BLOKKER)</div>
            </div>
            """, unsafe_allow_html=True)
        with p3:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">PERIMETER</div>
                <div class="metric-value" style="font-size:19px;">{perimeter:.2f}</div>
                <div class="metric-sub">KM</div>
            </div>
            """, unsafe_allow_html=True)

        # Weathering Age
        age_data = final_state.get("age_estimation", {})
        if age_data and age_data.get("status") == "ESTIMATED":
            age_rng = age_data.get("estimated_age_range_hours", [0, 0])
            regime = age_data.get("fay_regime", "N/A").replace("_", " ").title()
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">SPILL WEATHERING AGE (FAY SPREADING REGIME)</div>
                <div class="metric-value" style="font-size:19px; color:#00e5ff;">{age_rng[0]:.1f} – {age_rng[1]:.1f} HRS</div>
                <div class="metric-sub">REGIME: {regime.upper()} • CONFIDENCE: {age_data.get('confidence', 0):.0%}</div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# APPLICATION MODE RESOLUTION
# ──────────────────────────────────────────────────────────────
query_mode = st.query_params.get("mode")
if query_mode in ("landing", "live", "demo"):
    st.session_state["app_mode"] = query_mode
elif "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "landing"

app_mode = st.session_state.get("app_mode", "landing")

default_lat = float(st.session_state.get('spill_lat', DEMO_SPILL_LAT))
default_lon = float(st.session_state.get('spill_lon', DEMO_SPILL_LON))
spill_lat = default_lat
spill_lon = default_lon
is_demo = (app_mode == "demo")

# Render subtle background atmosphere & command palette
render_atmospheric_backdrop()

cur_spill_lat = default_lat
cur_spill_lon = default_lon
final_state_ref = st.session_state.get("pipeline_result")
cur_source_lat = float(final_state_ref.get('hindcast_result', {}).get('estimated_source_lat', cur_spill_lat) if final_state_ref else cur_spill_lat)
cur_source_lon = float(final_state_ref.get('hindcast_result', {}).get('estimated_source_lon', cur_spill_lon) if final_state_ref else cur_spill_lon)

render_command_palette(cur_spill_lat, cur_spill_lon, cur_source_lat, cur_source_lon)

# ──────────────────────────────────────────────────────────────
# TOP BRAND BAR & GLOBAL MODE NAVIGATION
# ──────────────────────────────────────────────────────────────
if app_mode != "landing":
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""
    <div class="brand-bar glass-panel tactical-reticle">
        <div class="brand-title-group">
            <span class="status-pulse" title="System Status: Sentinel-1 Telemetry Engine Active"></span>
            <div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <h1 class="brand-title">JAL-RAKSHAK</h1>
                    <span class="telemetry-chip">TACTICAL C2 // EPSG:4326</span>
                </div>
                <div class="brand-subtitle">MARITIME SATELLITE SURVEILLANCE & RECONNAISSANCE INTELLIGENCE SYSTEM</div>
            </div>
        </div>
        <div class="brand-meta-group">
            <div class="telemetry-readout">
                <span class="readout-label">SYSTEM EPOCH</span>
                <span class="readout-val" style="color:#ffffff;">{now_utc}</span>
            </div>
            <div class="telemetry-readout">
                <span class="readout-label">SENSOR PLATFORM</span>
                <span class="readout-val">SENTINEL-1A [C-SAR // VV+VH]</span>
            </div>
            <div class="telemetry-readout">
                <span class="readout-label">OBSERVATION ANCHOR</span>
                <span class="readout-val">{cur_spill_lat:.4f}°N, {cur_spill_lon:.4f}°E</span>
            </div>
            <div class="telemetry-readout">
                <span class="readout-label">TELEMETRY LINK</span>
                <span class="data-tag tag-observed">● SYNCHRONIZED</span>
            </div>
            <button id="cmd-palette-topbar-btn" class="cmd-topbar-pill" onclick="window.parent.dispatchEvent(new CustomEvent('jalrakshak:open_palette'))" title="Open Command Palette (/ or ⌘K)">
                <span class="cmd-pill-key">⌘K</span>
                <span class="cmd-pill-label">COMMANDS</span>
                <span class="cmd-pill-key">/</span>
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Global Mode Ribbon
    col_nav_brand, col_nav_home, col_nav_live, col_nav_demo = st.columns([6, 2, 2, 2])
    with col_nav_brand:
        mode_badge_text = "🟢 LIVE OPERATIONS CENTER" if app_mode == "live" else "🔶 GUIDED DEMO EVALUATION // CHENNAI"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; height:100%; padding-top:6px;">
            <span class="telemetry-micro-label" style="margin:0;">SYSTEM MODE:</span>
            <strong style="font-family:'JetBrains Mono'; font-size:12px; color:#f8fafc;">{mode_badge_text}</strong>
        </div>
        """, unsafe_allow_html=True)
    with col_nav_home:
        if st.button("🏠 Home Portal", key="global_btn_home", use_container_width=True):
            st.session_state["app_mode"] = "landing"
            st.query_params["mode"] = "landing"
            st.rerun()
    with col_nav_live:
        is_live_act = (app_mode == "live")
        if st.button("🛰️ Live Operations", key="global_btn_live", type="primary" if is_live_act else "secondary", use_container_width=True):
            st.session_state["app_mode"] = "live"
            st.query_params["mode"] = "live"
            st.rerun()
    with col_nav_demo:
        is_demo_act = (app_mode == "demo")
        if st.button("🧪 Guided Demo", key="global_btn_demo", type="primary" if is_demo_act else "secondary", use_container_width=True):
            st.session_state["app_mode"] = "demo"
            st.query_params["mode"] = "demo"
            st.session_state["demo_step"] = 1
            if not st.session_state.get("pipeline_result"):
                st.session_state["active_image_path"] = get_or_create_demo_sar_patch()
                st.session_state["spill_lat"] = CHENNAI_SCENARIO.spill_lat
                st.session_state["spill_lon"] = CHENNAI_SCENARIO.spill_lon
                st.session_state["current_scene_name"] = "Chennai Port Outer Anchorage (512x512)"
                st.session_state["auto_run"] = True
            st.rerun()


# ──────────────────────────────────────────────────────────────
# SIDEBAR OPERATIONS PANEL
# ──────────────────────────────────────────────────────────────
if app_mode != "landing":
    with st.sidebar:
        st.markdown("### 🎛️ Operations Control")

        mode_selection = st.radio(
            "Sensor Operating Mode",
            ["🔶 Demo Simulation Mode", "🟢 Live Telemetry Ingestion"],
            index=0 if is_demo else 1,
            help="Select between deterministic calibrated demo dataset and live external telemetry feeds.",
        )
        is_demo = (app_mode == "demo") or ("Demo" in mode_selection)

        st.markdown("---")

    st.markdown("#### ⚡ Quick-Load Operations")
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        if st.button("Chennai Spill", use_container_width=True, type="secondary"):
            demo_patch = get_or_create_demo_sar_patch()
            st.session_state["active_image_path"] = demo_patch
            st.session_state["spill_lat"] = CHENNAI_SCENARIO.spill_lat
            st.session_state["spill_lon"] = CHENNAI_SCENARIO.spill_lon
            st.session_state["auto_run"] = True
            st.session_state["current_scene_name"] = "Chennai Port Outer Anchorage (512x512)"
            st.rerun()
    with col_demo2:
        if st.button("Istanbul Scene", use_container_width=True, type="secondary"):
            st.session_state["active_image_path"] = "data/test_sar_scene.jpg"
            st.session_state["spill_lat"] = 41.1100
            st.session_state["spill_lon"] = 29.0500
            st.session_state["auto_run"] = True
            st.session_state["current_scene_name"] = "Istanbul Bosphorus Strait (1222x1600)"
            st.rerun()

    st.markdown("---")

    st.markdown("#### 📁 Sensor Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload SAR imagery (GeoTIFF / PNG / JPG)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
    )
    if uploaded_file is not None:
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["active_image_path"] = temp_path
        st.session_state["current_scene_name"] = uploaded_file.name

    uploaded_ais = st.file_uploader(
        "Historical AIS Archive (CSV / JSON)",
        type=["csv", "json", "geojson"],
        key="ais_uploader",
    )
    if uploaded_ais is not None:
        temp_ais_path = "temp_historical_ais.csv"
        with open(temp_ais_path, "wb") as f:
            f.write(uploaded_ais.getbuffer())
        st.session_state["active_ais_path"] = temp_ais_path
        st.caption("✅ Real AIS Archive Mounted")
    else:
        st.session_state.pop("active_ais_path", None)

    st.markdown("---")

    st.markdown("#### 📍 Observation Anchor")
    default_lat = st.session_state.get("spill_lat", DEMO_SPILL_LAT)
    default_lon = st.session_state.get("spill_lon", DEMO_SPILL_LON)
    spill_lat = st.number_input("Latitude (°N)", value=default_lat, format="%.4f")
    spill_lon = st.number_input("Longitude (°E)", value=default_lon, format="%.4f")

    st.markdown("---")

    active_image = st.session_state.get("active_image_path")
    if active_image and os.path.exists(active_image):
        if st.button("⚡ EXECUTE PIPELINE", type="primary", use_container_width=True):
            st.session_state["trigger_pipeline_run"] = True
            st.rerun()


# ──────────────────────────────────────────────────────────────
# EXECUTION CONTROLLER
# ──────────────────────────────────────────────────────────────
active_image = st.session_state.get("active_image_path")
should_run = st.session_state.pop("trigger_pipeline_run", False) or st.session_state.pop("auto_run", False)

if should_run and active_image and os.path.exists(active_image):
    with st.status("⚙️ Executing 11-Node LangGraph Intelligence Pipeline...", expanded=False) as status:
        stages = [
            "SAR Preprocessing & Speckle Reduction",
            "YOLOv8 Segmentation & Mask Extraction",
            "Classical Consensus & Land Masking Layer",
            "Geometric & Weathering Characterization",
            "Ocean Current Euler Hindcast",
            "AIS Fleet Spatiotemporal Filtering",
            "Candidate Vessel Association Scoring",
            "Forward Drift Trajectory Forecast",
            "Coastal Threat & Shoreline Assessment",
            "Automated Early Warning Dispatch",
            "Forensic Incident Dossier Generation",
        ]
        progress_bar = st.progress(0)
        for i, s in enumerate(stages):
            st.caption(f"Stage {i+1}/11: {s}")
            progress_bar.progress((i + 1) / len(stages))
            time.sleep(0.08)

        try:
            ais_file = st.session_state.get("active_ais_path")
            pipeline_out = run_pipeline(
                image_path=active_image,
                spill_lat=spill_lat,
                spill_lon=spill_lon,
                app_mode="demo" if is_demo else "live",
                ais_file_path=ais_file,
            )
            st.session_state["pipeline_result"] = pipeline_out
            status.update(label="✅ 11-Node Pipeline Execution Complete", state="complete")
        except Exception as e:
            status.update(label=f"❌ Execution Failure: {e}", state="error")
            st.error(f"Pipeline Execution Failed: {e}")

final_state = st.session_state.get("pipeline_result")


# =========================================================================
# SECTION 1: OVERVIEW SCREEN (TACTICAL MAP & RADAR CANVAS)
# =========================================================================
def render_overview_tab(final_state, is_demo, spill_lat, spill_lon):
    # ──────────────────────────────────────────────────────────
    # 1. RAPID OPERATIONS RIBBON
    # ──────────────────────────────────────────────────────────
    col_rib1, col_rib2, col_rib3, col_rib4 = st.columns([5, 3, 3, 4])
    with col_rib1:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; height:100%; padding-top:6px;">
            <span class="telemetry-micro-label" style="margin:0;">ACTIVE SCENE:</span>
            <span style="font-family:'JetBrains Mono'; font-size:12px; color:#f8fafc; font-weight:600;">{st.session_state.get('current_scene_name', 'Chennai Outer Anchorage')}</span>
            {render_tag('SIMULATED' if is_demo else 'OBSERVED')}
        </div>
        """, unsafe_allow_html=True)
    with col_rib2:
        if st.button("⚡ Mount Chennai (Confirmed)", key="ribbon_chennai", use_container_width=True, type="secondary"):
            st.session_state["active_image_path"] = get_or_create_demo_sar_patch()
            st.session_state["spill_lat"] = CHENNAI_SCENARIO.spill_lat
            st.session_state["spill_lon"] = CHENNAI_SCENARIO.spill_lon
            st.session_state["current_scene_name"] = "Chennai Port Outer Anchorage (512x512)"
            st.session_state["auto_run"] = True
            st.rerun()
    with col_rib3:
        if st.button("🛡️ Mount Istanbul (Control)", key="ribbon_istanbul", use_container_width=True, type="secondary"):
            st.session_state["active_image_path"] = "data/test_sar_scene.jpg"
            st.session_state["spill_lat"] = 41.1100
            st.session_state["spill_lon"] = 29.0500
            st.session_state["current_scene_name"] = "Istanbul Bosphorus Strait (1222x1600)"
            st.session_state["auto_run"] = True
            st.rerun()
    with col_rib4:
        if st.button("⚡ EXECUTE PIPELINE", key="ribbon_execute", use_container_width=True, type="primary"):
            st.session_state["trigger_pipeline_run"] = True
            st.rerun()

    # ──────────────────────────────────────────────────────────
    # 2. FLOATING HUD TELEMETRY STRIP (ABOVE PRIMARY MAP CANVAS)
    # ──────────────────────────────────────────────────────────
    cmd_spill_lat = final_state.get("spill_lat", spill_lat) if final_state else spill_lat
    cmd_spill_lon = final_state.get("spill_lon", spill_lon) if final_state else spill_lon
    source_lat = float(final_state.get("source_lat", final_state.get("hindcast_result", {}).get("estimated_source_lat", final_state.get("hindcast_result", {}).get("origin_lat", cmd_spill_lat))) if final_state else cmd_spill_lat)
    source_lon = float(final_state.get("source_lon", final_state.get("hindcast_result", {}).get("estimated_source_lon", final_state.get("hindcast_result", {}).get("origin_lon", cmd_spill_lon))) if final_state else cmd_spill_lon)
    cmd_status = final_state.get("validation_status", "AWAITING SENSOR PASS") if final_state else "STANDBY // NO SCAN"
    char_data = final_state.get("characterization", {}) if final_state else {}
    area_val = char_data.get("area_sq_km", final_state.get("spill_area_sq_km", 0.0) if final_state else 0.0)
    vol_val = char_data.get("estimated_volume_tons", 0.0)
    yolo_conf_val = final_state.get("detection_confidence", 0.0) if final_state else 0.0
    val_res_obj = normalize_validation_result(final_state.get("validation_result") if final_state else None)
    c_ratio_val = val_res_obj.get("contrast_ratio", 1.0)
    c_agree_val = val_res_obj.get("classical_agreement", 0.0)
    cand_list = final_state.get("candidate_scores", []) if final_state else []

    if "cmd_map_mode" not in st.session_state:
        st.session_state["cmd_map_mode"] = "🌐 ALL"

    hud_c1, hud_c2, hud_c3 = st.columns([7, 10, 7])
    with hud_c1:
        st.markdown(f"""
        <div class="hud-pill glass-panel">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="telemetry-label" style="margin:0 !important;">INCIDENT DISPOSITION</span>
                {render_tag('CONSENSUS')}
            </div>
            <div style="margin:4px 0 2px 0;">{render_status_pill(cmd_status)}</div>
            <div class="telemetry-coords">CENTROID: {cmd_spill_lat:.4f}°N, {cmd_spill_lon:.4f}°E • AREA: {area_val:.2f} KM²</div>
        </div>
        """, unsafe_allow_html=True)

    with hud_c2:
        cmd_modes = ["🌐 ALL", "🛢️ SPILL", "🚢 AIS", "🎯 SOURCE", "⏱️ BACKTRACK", "🌊 FORWARD DRIFT", "🛡️ RISK"]
        cur_m_idx = cmd_modes.index(st.session_state["cmd_map_mode"]) if st.session_state["cmd_map_mode"] in cmd_modes else 0
        sel_cmd_mode = st.radio(
            "Primary Canvas Mode",
            cmd_modes,
            index=cur_m_idx,
            horizontal=True,
            key="cmd_mode_radio",
            label_visibility="collapsed",
        )
        st.session_state["cmd_map_mode"] = sel_cmd_mode
        active_cmd_mode = sel_cmd_mode.replace("🌐 ", "").replace("🛢️ ", "").replace("🚢 ", "").replace("🎯 ", "").replace("⏱️ ", "").replace("🌊 ", "").replace("🛡️ ", "")

    with hud_c3:
        st.markdown(f"""
        <div class="hud-pill glass-panel">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="telemetry-label" style="margin:0 !important;">AI CONF</span>
                    <div class="telemetry-value" style="font-size:16px;">{yolo_conf_val:.1%}</div>
                </div>
                <div style="border-left:1px solid rgba(255,255,255,0.08); padding-left:10px;">
                    <span class="telemetry-label" style="margin:0 !important;">DAMPING</span>
                    <div class="telemetry-value" style="font-size:16px;">{c_ratio_val:.2f}</div>
                </div>
                <div style="border-left:1px solid rgba(255,255,255,0.08); padding-left:10px;">
                    <span class="telemetry-label" style="margin:0 !important;">FLEET</span>
                    <div class="telemetry-value" style="font-size:16px;">{len(cand_list)}</div>
                </div>
            </div>
            <div class="telemetry-sub" style="font-size:9.5px; margin:3px 0 0 0;">CONSENSUS: {c_agree_val:.0%} // 6 ALGORITHMS</div>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # 3. PRIMARY CANVAS: GEOSPATIAL MAP (65-75% OF SCREEN)
    # ──────────────────────────────────────────────────────────
    cmd_epoch_ts = final_state.get("detection_timestamp", "2026-09-14T15:30:00+00:00") if final_state else "2026-09-14T15:30:00+00:00"
    try:
        cmd_sim_time = datetime.fromisoformat(cmd_epoch_ts) + timedelta(minutes=st.session_state.get("timeline_min", 0))
        cmd_epoch_str = cmd_sim_time.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        cmd_epoch_str = "2026-09-14 15:30 UTC"

    st.markdown(f"""
    <div class="map-tactical-header tactical-reticle">
        <div>
            <span class="status-pulse-sm"></span>
            <span class="map-tactical-title">SATELLITE C2 // PRIMARY GEOSPATIAL INTELLIGENCE CANVAS [{active_cmd_mode}]</span>
        </div>
        <div>
            <span class="tech-spec-label">DATUM:</span> <span class="tech-spec-val">EPSG:4326</span> • 
            <span class="tech-spec-label">EPOCH:</span> <span class="tech-spec-val">{cmd_epoch_str}</span> • 
            <span class="tech-spec-label">DELTA:</span> <span class="tech-spec-val">{st.session_state.get('timeline_min', 0):+d}m</span> • 
            <span class="tech-spec-label">GSD:</span> <span class="tech-spec-val">10.0M</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    map_center_state = final_state if final_state else {
        "spill_lat": spill_lat,
        "spill_lon": spill_lon,
        "source_lat": spill_lat,
        "source_lon": spill_lon,
        "source_uncertainty_km": 5.0,
    }

    fmap_cmd = build_investigation_map(
        map_center_state,
        slider_minutes=st.session_state.get("timeline_min", 0),
        selected_vessel_mmsi=st.session_state.get("selected_vessel_mmsi"),
        mode=active_cmd_mode,
        focus_target=st.session_state.get("map_focus"),
    )
    # Primary Canvas: Height 580px gives dominant 65-75% visual weight with interactive polygon drawing
    map_output = st_folium(
        fmap_cmd,
        height=580,
        use_container_width=True,
        key="command_center_hero_map",
        returned_objects=["last_active_drawing", "all_drawings"],
    )

    # Process live operator drawn polygon if created
    if map_output:
        drawing = map_output.get("last_active_drawing")
        if not drawing and map_output.get("all_drawings"):
            d_list = map_output.get("all_drawings")
            if isinstance(d_list, list) and len(d_list) > 0:
                drawing = d_list[-1]
        if drawing and isinstance(drawing, dict) and "geometry" in drawing:
            coords = drawing["geometry"].get("coordinates", [])
            if coords and len(coords) > 0 and len(coords[0]) >= 3:
                ring = coords[0]
                lats = [float(p[1]) for p in ring]
                lons = [float(p[0]) for p in ring]
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                calc_area = polygon_area_km2(ring)
                curr_poly = st.session_state.get("drawn_polygon")
                if not curr_poly or abs(curr_poly.get("area_km2", 0) - calc_area) > 0.01:
                    st.session_state["drawn_polygon"] = {
                        "coordinates": ring,
                        "min_lat": min_lat,
                        "max_lat": max_lat,
                        "min_lon": min_lon,
                        "max_lon": max_lon,
                        "area_km2": calc_area,
                        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                    }
                    st.session_state["region_query_result"] = None
                    st.rerun()

    # Contextual Selected Region Toolbar & Action Panel
    poly = st.session_state.get("drawn_polygon")
    if poly:
        st.markdown(f"""
        <div class="glass-panel" style="padding:12px 18px; margin:10px 0 8px 0; border:1px solid rgba(0, 229, 255, 0.45); border-left:4px solid #00e5ff !important;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span class="status-pulse-sm" style="background:#00e5ff; box-shadow:0 0 10px #00e5ff;"></span>
                    <strong style="color:#00e5ff; font-family:'JetBrains Mono'; font-size:12px; letter-spacing:0.06em;">SELECTED GEOSPATIAL REGION:</strong>
                    <span style="font-family:'JetBrains Mono'; font-size:12px; color:#f8fafc; font-weight:700;">{poly.get('area_km2', 0):.2f} KM²</span>
                </div>
                <div style="font-family:'JetBrains Mono'; font-size:11px; color:#94a3b8;">
                    BOUNDS: [{poly['min_lat']:.4f}°N, {poly['min_lon']:.4f}°E] to [{poly['max_lat']:.4f}°N, {poly['max_lon']:.4f}°E]
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        qcol1, qcol2, qcol3, qcol4 = st.columns([3, 3, 3, 2])
        with qcol1:
            if st.button("🚢 Query Ships in Region", key="btn_query_ships", use_container_width=True, type="secondary"):
                ships_in_poly = []
                tracks = final_state.get("ais_tracks", {}) if final_state else {}
                if isinstance(tracks, dict):
                    t_items = tracks.items()
                elif isinstance(tracks, list):
                    t_items = [(t.get("mmsi", str(i)), t.get("points", [])) for i, t in enumerate(tracks)]
                else:
                    t_items = []
                for mmsi, recs in t_items:
                    if recs and isinstance(recs[0], dict):
                        for r in recs:
                            if isinstance(r, dict) and "lat" in r and "lon" in r:
                                lat_p, lon_p = float(r["lat"]), float(r["lon"])
                                if poly["min_lat"] <= lat_p <= poly["max_lat"] and poly["min_lon"] <= lon_p <= poly["max_lon"]:
                                    v_name = r.get("name", recs[0].get("name", mmsi))
                                    ships_in_poly.append({
                                        "mmsi": mmsi,
                                        "name": v_name,
                                        "lat": lat_p,
                                        "lon": lon_p,
                                        "speed": r.get("speed_knots", 0.0),
                                        "heading": r.get("heading", 0.0),
                                        "type": r.get("vessel_type", "Cargo/Tanker"),
                                    })
                                    break
                st.session_state["region_query_result"] = {
                    "type": "ships",
                    "count": len(ships_in_poly),
                    "items": ships_in_poly,
                }
                st.rerun()

        with qcol2:
            if st.button("🎯 Query Spill Activity", key="btn_query_spill", use_container_width=True, type="secondary"):
                h_res = final_state.get("hindcast_result", {}) if final_state else {}
                s_lat = float(final_state.get("source_lat", h_res.get("estimated_source_lat", h_res.get("origin_lat", spill_lat))) if final_state else spill_lat)
                s_lon = float(final_state.get("source_lon", h_res.get("estimated_source_lon", h_res.get("origin_lon", spill_lon))) if final_state else spill_lon)
                in_spill = (poly["min_lat"] <= spill_lat <= poly["max_lat"] and poly["min_lon"] <= spill_lon <= poly["max_lon"])
                in_source = (poly["min_lat"] <= s_lat <= poly["max_lat"] and poly["min_lon"] <= s_lon <= poly["max_lon"])
                char = final_state.get("characterization", {}) if final_state else {}
                st.session_state["region_query_result"] = {
                    "type": "spill",
                    "centroid_inside": in_spill,
                    "source_inside": in_source,
                    "spill_detected": final_state.get("spill_detected", False) if final_state else False,
                    "spill_area_sq_km": char.get("area_sq_km", final_state.get("spill_area_sq_km", 0.0) if final_state else 0.0),
                    "validation_status": final_state.get("validation_status", "PROBABLE") if final_state else "STANDBY",
                }
                st.rerun()

        with qcol3:
            if st.button("📊 Analyze Region", key="btn_analyze_region", use_container_width=True, type="secondary"):
                hind = final_state.get("hindcast_result", {}) if final_state else {}
                coast = final_state.get("coastal_impact", {}) if final_state else {}
                st.session_state["region_query_result"] = {
                    "type": "analysis",
                    "current_speed_ms": hind.get("current_speed_ms", 0.48),
                    "current_bearing_deg": hind.get("current_bearing_deg", 118.0),
                    "wind_speed_ms": hind.get("wind_speed_ms", 6.2),
                    "shoreline_dist_km": coast.get("shortest_distance_to_coast_km", 8.2),
                    "risk_tier": coast.get("risk_tier", "HIGH"),
                }
                st.rerun()

        with qcol4:
            if st.button("✕ Clear Region", key="btn_clear_region", use_container_width=True):
                st.session_state["drawn_polygon"] = None
                st.session_state["region_query_result"] = None
                st.rerun()

        # Render Query Results Box
        res_data = st.session_state.get("region_query_result")
        if res_data:
            q_type = res_data.get("type")
            if q_type == "ships":
                v_count = res_data.get("count", 0)
                items = res_data.get("items", [])
                st.markdown(f"""
                <div class="glass-panel" style="padding:12px 16px; margin:8px 0; border-left:3px solid #38bdf8 !important;">
                    <div style="font-family:'JetBrains Mono'; font-size:12px; color:#38bdf8; font-weight:700;">
                        IDENTIFIED {v_count} VESSELS WITHIN BOUNDING CORRIDOR
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if items:
                    for ship in items[:6]:
                        st.markdown(f"""
                        <div style="background:rgba(15, 23, 42, 0.7); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px 14px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <strong style="color:#f8fafc; font-size:12px;">🚢 {ship['name']}</strong>
                                <span style="font-family:'JetBrains Mono'; font-size:11px; color:#94a3b8; margin-left:8px;">MMSI: {ship['mmsi']}</span>
                            </div>
                            <div style="font-family:'JetBrains Mono'; font-size:11px; color:#38bdf8;">
                                [{ship['lat']:.4f}°N, {ship['lon']:.4f}°E] • {ship['speed']:.1f} kn
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            elif q_type == "spill":
                c_in = res_data.get("centroid_inside")
                s_in = res_data.get("source_inside")
                stat_spill = "YES // INTERSECTS CORRIDOR" if c_in else "NO // OUTSIDE CORRIDOR"
                stat_src = "YES // ORIGIN IN CORRIDOR" if s_in else "NO // OUTSIDE CORRIDOR"
                st.markdown(f"""
                <div class="glass-panel" style="padding:12px 16px; margin:8px 0; border-left:3px solid #ef4444 !important;">
                    <div style="font-family:'JetBrains Mono'; font-size:12px; color:#ef4444; font-weight:700; margin-bottom:6px;">
                        SPILL INTERSECTION QUERY RESULTS
                    </div>
                    <div style="display:flex; gap:24px; font-size:12px; color:#e2e8f0; font-family:'JetBrains Mono';">
                        <div>SPILL CENTROID: <strong style="color:#f8fafc;">{stat_spill}</strong></div>
                        <div>REVERSE SOURCE: <strong style="color:#f8fafc;">{stat_src}</strong></div>
                        <div>AREA: <strong style="color:#f8fafc;">{res_data.get('spill_area_sq_km', 0):.2f} KM²</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif q_type == "analysis":
                st.markdown(f"""
                <div class="glass-panel" style="padding:12px 16px; margin:8px 0; border-left:3px solid #a855f7 !important;">
                    <div style="font-family:'JetBrains Mono'; font-size:12px; color:#a855f7; font-weight:700; margin-bottom:6px;">
                        GEOSPATIAL & OCEANOGRAPHIC REGIONAL ASSESSMENT
                    </div>
                    <div style="display:flex; gap:20px; font-size:12px; color:#e2e8f0; font-family:'JetBrains Mono';">
                        <div>CURRENT: <strong style="color:#f8fafc;">{res_data.get('current_speed_ms', 0):.2f} m/s @ {res_data.get('current_bearing_deg', 0):.0f}°</strong></div>
                        <div>WIND: <strong style="color:#f8fafc;">{res_data.get('wind_speed_ms', 0):.1f} m/s</strong></div>
                        <div>COAST PROXIMITY: <strong style="color:#f8fafc;">{res_data.get('shoreline_dist_km', 0):.1f} KM</strong></div>
                        <div>RISK TIER: <strong style="color:#f8fafc;">{res_data.get('risk_tier', 'N/A')}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Contextual Candidate Vessel Drawer
    sel_vessel_mmsi = st.session_state.get("selected_vessel_mmsi")
    if sel_vessel_mmsi:
        v_rec = None
        v_track = []
        tracks = final_state.get("ais_tracks", {}) if final_state else {}
        if isinstance(tracks, dict) and sel_vessel_mmsi in tracks:
            v_track = tracks[sel_vessel_mmsi]
        elif isinstance(tracks, list):
            for t in tracks:
                if str(t.get("mmsi")) == str(sel_vessel_mmsi):
                    v_track = t.get("points", [])
                    break
        if v_track and isinstance(v_track[0], dict):
            v_rec = v_track[0]

        cand_scores = final_state.get("candidate_scores", []) if final_state else []
        c_info = next((c for c in cand_scores if str(c.get("mmsi")) == str(sel_vessel_mmsi)), {})
        v_score = c_info.get("score", 0.0)
        v_name = v_rec.get("name", f"VESSEL {sel_vessel_mmsi}") if v_rec else f"VESSEL {sel_vessel_mmsi}"
        v_lat = v_rec.get("lat", 0.0) if v_rec else 0.0
        v_lon = v_rec.get("lon", 0.0) if v_rec else 0.0
        v_spd = v_rec.get("speed_knots", 0.0) if v_rec else 0.0
        v_hdg = v_rec.get("heading", 0.0) if v_rec else 0.0
        v_type = v_rec.get("vessel_type", "Cargo/Tanker") if v_rec else "Cargo/Tanker"
        v_time = v_rec.get("timestamp", "2026-09-14 15:30:00 UTC") if v_rec else "2026-09-14 15:30:00 UTC"

        st.markdown(f"""
        <div class="glass-panel" style="padding:14px 18px; margin:10px 0; border:1px solid rgba(56, 189, 248, 0.45); border-left:4px solid #38bdf8 !important;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="telemetry-label" style="margin:0 !important; color:#38bdf8;">CANDIDATE VESSEL INTELLIGENCE</span>
                    <div style="font-size:18px; font-weight:800; color:#f8fafc; font-family:'JetBrains Mono'; margin-top:2px;">
                        🚢 {v_name} <span style="font-size:12px; color:#94a3b8; font-weight:400;">(MMSI: {sel_vessel_mmsi})</span>
                    </div>
                </div>
                <div style="text-align:right;">
                    <span class="telemetry-label" style="margin:0 !important;">TRAJECTORY CONSISTENCY</span>
                    <div style="font-size:20px; font-weight:800; color:#38bdf8; font-family:'JetBrains Mono';">{v_score:.0f}/100</div>
                </div>
            </div>
            <div class="telemetry-grid-4" style="margin-top:10px;">
                <div class="telemetry-metric-unit">
                    <span class="telemetry-label">POSITION</span>
                    <strong class="telemetry-value-sm">[{v_lat:.4f}°N, {v_lon:.4f}°E]</strong>
                </div>
                <div class="telemetry-metric-unit">
                    <span class="telemetry-label">SPEED // HEADING</span>
                    <strong class="telemetry-value-sm">{v_spd:.1f} KN // {v_hdg:.0f}°</strong>
                </div>
                <div class="telemetry-metric-unit">
                    <span class="telemetry-label">VESSEL TYPE</span>
                    <strong class="telemetry-value-sm">{v_type}</strong>
                </div>
                <div class="telemetry-metric-unit">
                    <span class="telemetry-label">LAST TELEMETRY</span>
                    <strong class="telemetry-value-sm" style="font-size:11px;">{str(v_time)[:19].replace('T', ' ')} UTC</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        vbtn_col1, vbtn_col2, vbtn_col3 = st.columns([3, 3, 6])
        with vbtn_col1:
            if st.button("📍 Focus on Vessel", key="btn_focus_sel_vessel", use_container_width=True, type="primary"):
                st.session_state["map_focus"] = "vessel"
                st.rerun()
        with vbtn_col2:
            if st.button("✕ Close Vessel Drawer", key="btn_close_vessel_drawer", use_container_width=True):
                st.session_state["selected_vessel_mmsi"] = None
                st.session_state["map_focus"] = None
                st.rerun()

    # ──────────────────────────────────────────────────────────
    # 4. FLOATING FOCUS & FORENSIC SCRUBBER BAR (BELOW MAP)
    # ──────────────────────────────────────────────────────────
    # Dynamic simulation status banner
    if st.session_state.get("trace_active"):
        hind_data = final_state.get("hindcast_result", {}) if final_state else {}
        st.markdown(f"""
        <div class="glass-panel" style="padding:8px 14px; margin-bottom:8px; border-left:3px solid #f59e0b !important; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span class="status-pulse-sm" style="background:#f59e0b; box-shadow:0 0 8px #f59e0b;"></span>
                <strong style="color:#f8fafc; font-size:12px; font-family:'JetBrains Mono';">PROGRESSIVE BACKTRACK ACTIVE:</strong>
                <span style="color:#94a3b8; font-size:12px;"> Traced 180 min back to estimated origin (Bearing: {hind_data.get('current_bearing_deg', 118):.0f}° • Current: {hind_data.get('current_speed_ms', 0.48):.2f} m/s)</span>
            </div>
            {render_tag('HINDCAST')}
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.get("drift_sim_active"):
        coast_data = final_state.get("coastal_impact", {}) if final_state else {}
        st.markdown(f"""
        <div class="glass-panel" style="padding:8px 14px; margin-bottom:8px; border-left:3px solid #0ea5e9 !important; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span class="status-pulse-sm" style="background:#0ea5e9; box-shadow:0 0 8px #0ea5e9;"></span>
                <strong style="color:#f8fafc; font-size:12px; font-family:'JetBrains Mono';">FORWARD DRIFT SIMULATION ACTIVE:</strong>
                <span style="color:#94a3b8; font-size:12px;"> Shoreline Distance: {coast_data.get('shortest_distance_to_coast_km', 8.2):.1f} km • Landfall ETA: {coast_data.get('eta_to_coast_hours', 14.5):.1f} hrs</span>
            </div>
            {render_tag('PREDICTED')}
        </div>
        """, unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns([2, 2, 2, 2.5, 2.5, 4.5])
    with ctrl_col1:
        if st.button("🎯 Focus Spill", key="cmd_foc_spill", use_container_width=True):
            st.session_state["map_focus"] = "spill"
            st.session_state["cmd_map_mode"] = "🛢️ SPILL"
            st.session_state["trace_active"] = False
            st.session_state["drift_sim_active"] = False
            st.rerun()
    with ctrl_col2:
        if st.button("📍 Focus Origin", key="cmd_foc_origin", use_container_width=True):
            st.session_state["map_focus"] = "source"
            st.session_state["cmd_map_mode"] = "🎯 SOURCE"
            st.session_state["trace_active"] = False
            st.session_state["drift_sim_active"] = False
            st.rerun()
    with ctrl_col3:
        if st.button("🔄 Reset Center", key="cmd_foc_reset", use_container_width=True):
            st.session_state["map_focus"] = None
            st.session_state["selected_vessel_mmsi"] = None
            st.session_state["cmd_map_mode"] = "🌐 ALL"
            st.session_state["trace_active"] = False
            st.session_state["drift_sim_active"] = False
            st.rerun()
    with ctrl_col4:
        is_trace = st.session_state.get("trace_active", False)
        trace_lbl = "✓ TRACING ACTIVE" if is_trace else "⚡ TRACE SOURCE"
        if st.button(trace_lbl, key="cmd_trace_source", use_container_width=True, type="primary" if is_trace else "secondary"):
            st.session_state["trace_active"] = True
            st.session_state["drift_sim_active"] = False
            st.session_state["cmd_map_mode"] = "⏱️ BACKTRACK"
            st.session_state["map_focus"] = "source"
            st.session_state["timeline_min"] = -180
            st.rerun()
    with ctrl_col5:
        is_drift = st.session_state.get("drift_sim_active", False)
        drift_lbl = "✓ DRIFT ACTIVE" if is_drift else "🌊 SIMULATE DRIFT"
        if st.button(drift_lbl, key="cmd_drift_sim", use_container_width=True, type="primary" if is_drift else "secondary"):
            st.session_state["drift_sim_active"] = True
            st.session_state["trace_active"] = False
            st.session_state["cmd_map_mode"] = "🌊 FORWARD DRIFT"
            st.session_state["map_focus"] = "spill"
            st.session_state["timeline_min"] = 60
            st.rerun()
    with ctrl_col6:
        c_names = ["All Fleet (Unselected)"] + [f"{c['name']} (MMSI: {c['mmsi']})" for c in cand_list]
        sel_c_idx = 0
        if st.session_state.get("selected_vessel_mmsi"):
            for idx, c in enumerate(cand_list):
                if str(c["mmsi"]) == str(st.session_state.get("selected_vessel_mmsi")):
                    sel_c_idx = idx + 1
                    break
        chosen_v = st.selectbox(
            "Target Vessel Selector",
            c_names,
            index=sel_c_idx,
            key="cmd_vessel_selector",
            label_visibility="collapsed",
        )
        if chosen_v != "All Fleet (Unselected)":
            m_v = next((c for c in cand_list if f"{c['name']} (MMSI: {c['mmsi']})" == chosen_v), None)
            if m_v:
                st.session_state["selected_vessel_mmsi"] = m_v["mmsi"]
        else:
            st.session_state["selected_vessel_mmsi"] = None

    # Scrub buttons and simulation scrubber
    sc_c1, sc_c2, sc_c3, sc_c4, sc_c5 = st.columns([1, 1, 1, 1, 4])
    with sc_c1:
        if st.button("↺ -180m", key="cmd_scrub_180", use_container_width=True):
            st.session_state["timeline_min"] = -180
            st.rerun()
    with sc_c2:
        if st.button("◀ -15m", key="cmd_scrub_m15", use_container_width=True):
            st.session_state["timeline_min"] = max(-180, st.session_state.get("timeline_min", 0) - 15)
            st.rerun()
    with sc_c3:
        if st.button("▶ +15m", key="cmd_scrub_p15", use_container_width=True):
            st.session_state["timeline_min"] = min(60, st.session_state.get("timeline_min", 0) + 15)
            st.rerun()
    with sc_c4:
        if st.button("🎯 At Detection", key="cmd_scrub_zero", use_container_width=True):
            st.session_state["timeline_min"] = 0
            st.rerun()
    with sc_c5:
        st.session_state["timeline_min"] = st.slider(
            "Forensic Timeline",
            min_value=-180,
            max_value=60,
            value=st.session_state.get("timeline_min", 0),
            step=5,
            format="%d min",
            key="slider_cmd_playback",
            label_visibility="collapsed",
        )

    # ──────────────────────────────────────────────────────────
    # 5. DOCKED BOTTOM INTELLIGENCE CONSOLE
    # ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📡 Intelligence Console")

    console_tabs = st.tabs([
        "⚖️ Multi-Signal Evidence & Consensus",
        "🚢 AIS Candidate Vessel Attribution",
        "⏱️ Ocean Drift & Origin Hindcast",
        "🛡️ Shoreline Threat & Sensitive Assets",
        "📋 Calibrated Reference Cases",
    ])

    with console_tabs[0]:
        render_evidence_panel(final_state, selected_mmsi=st.session_state.get("selected_vessel_mmsi"), key_prefix="cmd_console")

    with console_tabs[1]:
        if final_state and final_state.get("candidates_done") and cand_list:
            st.markdown(f"**AIS Feed:** {render_tag(final_state.get('ais_data_mode', 'ARCHIVE'))} • **Correlated Fleet Size:** `{len(cand_list)} Vessels`")
            for i, cand in enumerate(cand_list):
                score = cand.get("score", 0.0)
                score_css = "vessel-score-high" if score >= 70 else ("vessel-score-med" if score >= 40 else "vessel-score-low")
                bar_color = "#ef4444" if score >= 70 else ("#f59e0b" if score >= 40 else "#64748b")
                bdown = cand.get("breakdown", {})
                is_active_vessel = (str(cand.get("mmsi")) == str(st.session_state.get("selected_vessel_mmsi")))
                border_style = "border:2px solid #00e5ff;" if is_active_vessel else ""

                st.markdown(f"""
                <div class="vessel-card glass-panel" style="{border_style}">
                    <div class="vessel-header">
                        <div>
                            <span class="telemetry-micro-label">CANDIDATE TARGET #{i+1}</span>
                            <div class="vessel-name">🚢 {cand.get('name', 'UNKNOWN')}</div>
                            <span class="vessel-mmsi">MMSI: {cand.get('mmsi')}</span>
                        </div>
                        <div style="text-align:right;">
                            <span class="telemetry-micro-label">ASSOCIATION SCORE</span>
                            <div class="vessel-score {score_css}">{score:.0f}<span style="font-size:12px; color:#64748b;">/100</span></div>
                        </div>
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width:{score}%; background:{bar_color};"></div>
                    </div>
                    <div class="telemetry-grid-4">
                        <div class="telemetry-metric-unit">
                            <span class="telemetry-label">SOURCE PROXIMITY</span>
                            <strong class="telemetry-value-sm">{cand.get('min_distance_km', 0.0):.1f} KM</strong>
                        </div>
                        <div class="telemetry-metric-unit">
                            <span class="telemetry-label">TEMPORAL WINDOW</span>
                            <strong class="telemetry-value-sm" style="color:{'#34d399' if cand.get('time_match') else '#f87171'};">
                                {'COINCIDENT' if cand.get('time_match') else 'OUTSIDE WINDOW'}
                            </strong>
                        </div>
                        <div class="telemetry-metric-unit">
                            <span class="telemetry-label">TRAJECTORY CONSISTENCY</span>
                            <strong class="telemetry-value-sm">{bdown.get('trajectory', 0.0):.0f}%</strong>
                        </div>
                        <div class="telemetry-metric-unit">
                            <span class="telemetry-label">DRIFT ALIGNMENT</span>
                            <strong class="telemetry-value-sm">{bdown.get('drift_consistency', 0.0):.0f}%</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                btn_lbl = "✓ Active Target on Primary Canvas" if is_active_vessel else f"🎯 Highlight {cand.get('name', cand.get('mmsi'))} on Map"
                if st.button(btn_lbl, key=f"cmd_btn_ais_{cand['mmsi']}", use_container_width=True, type="primary" if is_active_vessel else "secondary"):
                    st.session_state["selected_vessel_mmsi"] = cand["mmsi"]
                    st.rerun()
        else:
            st.info("Execute pipeline to compute spatiotemporal AIS vessel candidate rankings.")

    with console_tabs[2]:
        if final_state and final_state.get("hindcast_done"):
            hindcast = final_state.get("hindcast_result", {})
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.markdown(f"""
                <div class="metric-card glass-panel">
                    <div class="telemetry-label">ESTIMATED ORIGIN</div>
                    <div class="metric-value" style="font-size:18px;">{hindcast.get('origin_lat', 0.0):.4f}°N</div>
                    <div class="metric-sub">{hindcast.get('origin_lon', 0.0):.4f}°E // HINDCAST</div>
                </div>
                """, unsafe_allow_html=True)
            with h2:
                st.markdown(f"""
                <div class="metric-card glass-panel">
                    <div class="telemetry-label">SPATIAL UNCERTAINTY</div>
                    <div class="metric-value">±{final_state.get('source_uncertainty_km', 0.0):.1f}</div>
                    <div class="metric-sub">KM RADIUS (P95 CONFIDENCE)</div>
                </div>
                """, unsafe_allow_html=True)
            with h3:
                st.markdown(f"""
                <div class="metric-card glass-panel">
                    <div class="telemetry-label">OCEAN CURRENT</div>
                    <div class="metric-value">{hindcast.get('current_speed_ms', 0.0):.2f} M/S</div>
                    <div class="metric-sub">BEARING: {hindcast.get('current_bearing_deg', 0.0):.0f}°</div>
                </div>
                """, unsafe_allow_html=True)
            with h4:
                st.markdown(f"""
                <div class="metric-card glass-panel">
                    <div class="telemetry-label">WIND DRIFT FACTOR</div>
                    <div class="metric-value">3.0%</div>
                    <div class="metric-sub">EKMAN CURRENT TRANSFER</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Execute pipeline to generate ocean drift backtrack vectors and origin uncertainty bounds.")

    with console_tabs[3]:
        if final_state and final_state.get("risk_done"):
            coastal = final_state.get("coastal_impact", {})
            risk_obj = final_state.get("risk_assessment", {})
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card glass-panel">
                    <div class="telemetry-label">SHORELINE PROXIMITY</div>
                    <div class="metric-value">{coastal.get('shortest_distance_to_coast_km', 0.0):.1f} KM</div>
                    <div class="metric-sub">NEAREST: {coastal.get('nearest_shoreline_point', {}).get('name', 'N/A').upper()}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                eta = coastal.get("eta_to_coast_hours")
                eta_str = f"{eta:.1f} HRS" if eta else "NO LANDFALL"
                st.markdown(f"""
                <div class="metric-card glass-panel">
                    <div class="telemetry-label">LANDFALL ETA</div>
                    <div class="metric-value">{eta_str}</div>
                    <div class="metric-sub">TRAJECTORY PROJECTION</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card glass-panel">
                    <div class="telemetry-label">COASTAL VULNERABILITY</div>
                    <div class="metric-value">{coastal.get('coastal_vulnerability_score', 0.0):.0f}/100</div>
                    <div class="metric-sub">TIER: {coastal.get('risk_tier', 'LOW')}</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card glass-panel">
                    <div class="telemetry-label">THREATENED ASSETS</div>
                    <div class="metric-value">{coastal.get('threatened_assets_count', 0)}</div>
                    <div class="metric-sub">ECOLOGICAL & INFRASTRUCTURE</div>
                </div>
                """, unsafe_allow_html=True)

            threatened = coastal.get("threatened_assets", [])
            if threatened:
                st.markdown("##### 🛡️ Protected Marine & Shoreline Assets in Threat Corridor")
                for t in threatened:
                    t_level = t.get("threat_level", "MONITOR")
                    t_color = "#ef4444" if t_level == "IMMINENT" else ("#f59e0b" if t_level == "HIGH_RISK" else "#00e5ff")
                    st.markdown(f"""
                    <div class="glass-panel" style="border-left:3px solid {t_color} !important; padding:12px 16px; margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="color:#f8fafc; font-size:13px;">{t['name']} ({t.get('category', 'Asset').upper()})</strong>
                            <span style="color:{t_color}; font-weight:700; font-size:10px; font-family:'JetBrains Mono';">{t_level} • ESI {t.get('esi', 5)}/10</span>
                        </div>
                        <div style="font-size:11px; font-family:'JetBrains Mono'; color:#94a3b8; margin:4px 0;">
                            DISTANCE: {t.get('distance_from_spill_km', 0):.1f} KM | AUTHORITY: {t.get('contact_authority', 'Port Trust').upper()}
                        </div>
                        <div style="font-size:12px; color:#34d399;">
                            STRATEGY: {t.get('recommended_strategy', 'Deploy containment booms')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Execute pipeline to compute coastal shoreline approach vectors and environmental sensitivity rankings.")

    with console_tabs[4]:
        st.markdown("##### 📋 Audited Reference Analyses")
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            st.markdown(f"""
            <div class="recent-card glass-panel">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <span class="telemetry-micro-label">CASE: BAY-OF-BENGAL-01</span>
                        <strong style="font-size:16px; color:#f8fafc;">Chennai Port Outer Anchorage</strong>
                        <div class="telemetry-coords">13.1250°N, 80.3850°E // ANCHORAGE SECTOR</div>
                    </div>
                    {render_status_pill('CONFIRMED BY MULTIPLE SIGNALS')}
                </div>
                <p style="font-size:12px; color:#94a3b8; margin:12px 0 14px 0; line-height:1.5;">
                    Genuine maritime mineral oil slick. Primary YOLOv8 segmentation (48% conf) independently verified by K-Means dark cluster extraction and local adaptive thresholding. Radar damping contrast ratio 0.30 in open water.
                </p>
                <div class="recent-meta-bar">
                    <span>ACQUIRED: 2026-09-14 15:30 UTC</span>
                    <span>SCENE: demo_sar_patch.png</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Mount & Analyze Chennai Scene", key="cmd_btn_rec_chennai", use_container_width=True):
                st.session_state["active_image_path"] = get_or_create_demo_sar_patch()
                st.session_state["spill_lat"] = CHENNAI_SCENARIO.spill_lat
                st.session_state["spill_lon"] = CHENNAI_SCENARIO.spill_lon
                st.session_state["current_scene_name"] = "Chennai Port Outer Anchorage (512x512)"
                st.session_state["auto_run"] = True
                st.rerun()

        with col_rec2:
            st.markdown(f"""
            <div class="recent-card glass-panel">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <span class="telemetry-micro-label">CASE: BOSPHORUS-STRAIT-02</span>
                        <strong style="font-size:16px; color:#f8fafc;">Istanbul Bosphorus Strait</strong>
                        <div class="telemetry-coords">41.1100°N, 29.0500°E // COASTAL SWATH</div>
                    </div>
                    {render_status_pill('REJECTED')}
                </div>
                <p style="font-size:12px; color:#94a3b8; margin:12px 0 14px 0; line-height:1.5;">
                    Terrestrial topography negative control. Raw YOLO proposed a candidate polygon over the European landmass; correctly rejected by marine domain constraint (90.9% land overlap) and positive backscatter contrast (1.84).
                </p>
                <div class="recent-meta-bar">
                    <span>ACQUIRED: 2026-09-28 12:20 UTC</span>
                    <span>SCENE: test_sar_scene.jpg</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Mount & Analyze Istanbul Scene", key="cmd_btn_rec_istanbul", use_container_width=True):
                st.session_state["active_image_path"] = "data/test_sar_scene.jpg"
                st.session_state["spill_lat"] = 41.1100
                st.session_state["spill_lon"] = 29.0500
                st.session_state["current_scene_name"] = "Istanbul Bosphorus Strait (1222x1600)"
                st.session_state["auto_run"] = True
                st.rerun()


# =========================================================================
# SECTION 2: SAR INTELLIGENCE (DETECTION & CONSENSUS)
# =========================================================================
def render_sar_tab(final_state, is_demo, spill_lat, spill_lon, active_image=None):
    active_image = active_image or st.session_state.get("active_image_path")
    if not active_image or not os.path.exists(active_image):
        st.info("ℹ️ No SAR imagery loaded. Select a quick scenario from the Command Center or sidebar to begin.")
    else:
        # Header banner
        st.markdown(f"""
        <div class="sar-viewer-header">
            <div>
                <span class="status-pulse-sm"></span>
                <strong style="font-size:14px; letter-spacing:0.8px; color:#f8fafc; font-family:'JetBrains Mono';">SAR INTELLIGENCE WORKSPACE</strong>
                <span style="color:#64748b; font-size:12px; margin-left:8px;">// {st.session_state.get('current_scene_name', 'Active SAR Scene')}</span>
            </div>
            <div>
                {render_tag('SIMULATED' if is_demo else 'OBSERVED')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main 2-Column Workspace: LEFT (65% Viewer) & RIGHT (35% Analysis Inspector)
        sar_left_col, sar_right_col = st.columns([13, 8])

        # ──────────────────────────────────────────────────────────
        # LEFT: LARGE SAR IMAGE VIEWER (PAN, ZOOM, LAYER TOGGLES)
        # ──────────────────────────────────────────────────────────
        with sar_left_col:
            st.markdown("##### 🛰️ Sensor Swath & Visual Layers")

            # Layer selector pills
            layer_options = [
                "🛰️ Composite Overlay",
                "📷 Raw SAR Amplitude",
                "🎯 YOLO Detection Mask",
                "🌊 Land/Sea Domain Mask",
                "⚖️ Classical Validation",
                "✅ Final Consensus",
                "📊 6-Panel Diagnostic Matrix",
            ]
            selected_layer = st.radio(
                "Active SAR Intelligence Layer",
                layer_options,
                index=0,
                horizontal=True,
                key="sar_tab2_layer_sel",
                label_visibility="collapsed",
            )

            # Map layer string to internal key
            layer_key_map = {
                "🛰️ Composite Overlay": "composite",
                "📷 Raw SAR Amplitude": "raw",
                "🎯 YOLO Detection Mask": "yolo",
                "🌊 Land/Sea Domain Mask": "landmask",
                "⚖️ Classical Validation": "classical",
                "✅ Final Consensus": "final",
                "📊 6-Panel Diagnostic Matrix": "diagnostics",
            }
            active_layer_key = layer_key_map.get(selected_layer, "composite")

            # Controls: Viewer engine + Zoom presets
            col_v_eng, col_v_zoom = st.columns([1, 1])
            with col_v_eng:
                viewer_engine = st.radio(
                    "Viewer Engine",
                    ["🗺️ Interactive Canvas (Pan / Zoom / Inspect)", "🔬 High-Resolution Raster"],
                    index=0,
                    horizontal=True,
                    key="sar_tab2_engine_sel",
                )
            with col_v_zoom:
                zoom_choice = st.select_slider(
                    "Magnification Presets",
                    options=["100% (Fit)", "150% (Standard)", "200% (High Detail)", "300% (Pixel Forensics)"],
                    value="100% (Fit)",
                    key="sar_tab2_zoom_slider",
                )

            # Render Image based on chosen engine
            if viewer_engine == "🗺️ Interactive Canvas (Pan / Zoom / Inspect)":
                sar_meta_data = final_state.get("sar_metadata", {}) if final_state else {}
                bbox = sar_meta_data.get("bbox") if sar_meta_data else None
                s_lat = final_state.get("spill_lat", spill_lat) if final_state else spill_lat
                s_lon = final_state.get("spill_lon", spill_lon) if final_state else spill_lon
                char_data = final_state.get("characterization", {}) if final_state else {}
                area_val = char_data.get("area_sq_km", final_state.get("spill_area_sq_km", 0.0) if final_state else 0.0)
                y_conf = final_state.get("detection_confidence", 0.0) if final_state else 0.0
                all_c = final_state.get("all_spill_coords", []) if final_state else []

                # Build Leaflet Map for SAR Swath with Pan and Zoom
                sar_map = folium.Map(
                    location=[s_lat, s_lon],
                    zoom_start=13,
                    tiles="OpenStreetMap",
                    control_scale=True,
                )
                Fullscreen(position="topright").add_to(sar_map)

                if bbox and len(bbox) == 4:
                    min_lat, min_lon, max_lat, max_lon = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    # Add ImageOverlay
                    folium.raster_layers.ImageOverlay(
                        image=active_image,
                        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                        opacity=0.88,
                        name="SAR Amplitude Swath",
                    ).add_to(sar_map)

                    # Project polygons with interactive hover and click inspection
                    h_r, w_r = (512, 512)
                    if sar_meta_data and "dimensions" in sar_meta_data:
                        h_r, w_r = sar_meta_data["dimensions"][0], sar_meta_data["dimensions"][1]

                    for p_idx, poly in enumerate(all_c):
                        if poly and len(poly) >= 3:
                            geo_p = []
                            for pt in poly:
                                px_x, px_y = float(pt[0]), float(pt[1])
                                pt_lat = max_lat - (px_y / h_r) * (max_lat - min_lat)
                                pt_lon = min_lon + (px_x / w_r) * (max_lon - min_lon)
                                geo_p.append([pt_lat, pt_lon])

                            folium.Polygon(
                                locations=geo_p,
                                color="#00e5ff",
                                weight=3,
                                fill=True,
                                fill_color="#ef4444",
                                fill_opacity=0.45,
                                tooltip=f"🔍 Hover: Candidate Polygon #{p_idx+1} | Confidence: {y_conf:.1%} | Area: {area_val:.2f} km²",
                                popup=f"""<div style="font-family:'JetBrains Mono',monospace; font-size:12px; min-width:210px;">
                                    <strong style="color:#00e5ff;">DETECTED OIL SLICK #{p_idx+1}</strong><br>
                                    <b>YOLOv8 Confidence:</b> {y_conf:.1%}<br>
                                    <b>Slick Area:</b> {area_val:.3f} km²<br>
                                    <b>Centroid:</b> {s_lat:.4f}°N, {s_lon:.4f}°E<br>
                                    <b>Vertices:</b> {len(poly)} points
                                </div>""",
                            ).add_to(sar_map)

                    folium.Marker(
                        location=[s_lat, s_lon],
                        icon=folium.DivIcon(
                            html='<div style="font-size:11px; color:#ef4444; font-weight:700; white-space:nowrap; background:rgba(11,17,32,0.9); border:1px solid #ef4444; padding:2px 6px; border-radius:4px; margin-top:-25px; margin-left:-20px;">🛢️ SLICK CENTROID</div>'
                        ),
                    ).add_to(sar_map)

                st_folium(sar_map, height=520, use_container_width=True, key="sar_intelligence_canvas_map", returned_objects=[])
                st.caption("🖱️ **Interaction:** Scroll wheel to zoom (9x-18x) • Click & drag to pan • Hover over polygon for live telemetry • Click polygon for forensic popup.")
            else:
                # High Resolution Raster with Layer Switcher
                layer_img = generate_sar_layer_image(active_image, active_layer_key, final_state or {})
                if layer_img is not None:
                    zoom_factor = 1.0
                    if "150%" in zoom_choice:
                        zoom_factor = 1.5
                    elif "200%" in zoom_choice:
                        zoom_factor = 2.0
                    elif "300%" in zoom_choice:
                        zoom_factor = 3.0

                    if zoom_factor > 1.0:
                        h_orig, w_orig = layer_img.shape[:2]
                        layer_img = cv2.resize(layer_img, (int(w_orig * zoom_factor), int(h_orig * zoom_factor)), interpolation=cv2.INTER_LINEAR)

                    st.image(layer_img, use_container_width=True, caption=f"SAR Intelligence Layer: {selected_layer} ({zoom_choice})")

            # Interactive Polygon List & Inspector
            all_c = final_state.get("all_spill_coords", []) if final_state else []
            if all_c:
                st.markdown("###### 📐 Detected Polygon Geometries")
                for p_idx, poly in enumerate(all_c):
                    pts_np = np.array(poly)
                    min_x, min_y = pts_np.min(axis=0)
                    max_x, max_y = pts_np.max(axis=0)
                    st.markdown(f"""
                    <div class="polygon-chip">
                        <span style="color:#00e5ff; font-weight:700;">POLYGON #{p_idx+1}</span>
                        <span>VERTICES: {len(poly)}</span>
                        <span>BBOX: [{int(min_x)}, {int(min_y)}, {int(max_x)}, {int(max_y)}]</span>
                        <span>SPAN: {int(max_x - min_x)}×{int(max_y - min_y)} PX</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────
        # RIGHT: RIGOROUS ANALYSIS INSPECTOR (GROUNDED IN REAL DATA)
        # ──────────────────────────────────────────────────────────
        with sar_right_col:
            st.markdown("##### 🔬 Analysis Inspector")

            sar_meta_data = final_state.get("sar_metadata", {}) if final_state else {}
            dims = sar_meta_data.get("dimensions", [512, 512]) if sar_meta_data else [512, 512]
            res_m = sar_meta_data.get("pixel_resolution_m", 10.0) if sar_meta_data else 10.0
            bbox = sar_meta_data.get("bbox") if sar_meta_data else None
            acq_time = sar_meta_data.get("acquisition_timestamp") or (final_state.get("detection_timestamp", "2026-09-14 15:30:00 UTC") if final_state else "2026-09-14 15:30:00 UTC")
            sensor_name = sar_meta_data.get("sensor", "Sentinel-1A [C-Band SAR]") if sar_meta_data else "Sentinel-1A [C-Band SAR]"

            char_data = final_state.get("characterization", {}) if final_state else {}
            area_val = char_data.get("area_sq_km", final_state.get("spill_area_sq_km", 0.0) if final_state else 0.0)
            y_conf = final_state.get("detection_confidence", 0.0) if final_state else 0.0
            val_status = final_state.get("validation_status", "AWAITING SENSOR PASS") if final_state else "STANDBY"
            val_res = normalize_validation_result(final_state.get("validation_result") if final_state else None)
            c_agree = val_res.get("classical_agreement", 0.0)
            contrast_val = val_res.get("contrast_ratio", 1.0)
            land_frac = val_res.get("method_results", {}).get("land_mask", {}).get("overlap_fraction", 0.0)
            look_risk = val_res.get("look_alike_risk", 0.0)
            spill_detected_flag = final_state.get("spill_detected", False) if final_state else False

            # 1. SAR SCENE HIERARCHY
            bbox_str = f"[{bbox[0]:.4f}, {bbox[1]:.4f}] to [{bbox[2]:.4f}, {bbox[3]:.4f}]" if (bbox and len(bbox) == 4) else f"{spill_lat:.4f}°N, {spill_lon:.4f}°E"
            st.markdown(f"""
            <div class="glass-panel" style="padding:14px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="telemetry-label" style="margin:0 !important;">SAR SCENE TELEMETRY</span>
                    {render_tag('SENTINEL-1')}
                </div>
                <div style="margin-top:8px;" class="telemetry-grid-4">
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">SENSOR PLATFORM</span>
                        <strong class="telemetry-value-sm">{sensor_name}</strong>
                    </div>
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">ACQUISITION TIME</span>
                        <strong class="telemetry-value-sm" style="font-size:11px;">{acq_time[:19].replace('T', ' ')} UTC</strong>
                    </div>
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">SPATIAL GSD</span>
                        <strong class="telemetry-value-sm">{res_m:.1f} M/PX</strong>
                    </div>
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">RASTER EXTENT</span>
                        <strong class="telemetry-value-sm">{dims[1]} × {dims[0]} PX</strong>
                    </div>
                </div>
                <div class="telemetry-coords" style="margin-top:8px;">BBOX: {bbox_str} // CRS: EPSG:4326</div>
            </div>
            """, unsafe_allow_html=True)

            # 2. DETECTION HIERARCHY
            det_status_str = "VALIDATED OIL SLICK" if (spill_detected_flag and val_status != "REJECTED") else ("REJECTED / NO ANOMALY" if val_status == "REJECTED" else "STANDBY")
            det_badge_class = "tag-observed" if (spill_detected_flag and val_status != "REJECTED") else "tag-simulated"
            c_px = char_data.get("centroid_px", [0.0, 0.0])
            st.markdown(f"""
            <div class="glass-panel" style="padding:14px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="telemetry-label" style="margin:0 !important;">YOLOv8 DEEP DETECTION</span>
                    <span class="data-tag {det_badge_class}">{det_status_str}</span>
                </div>
                <div style="margin-top:8px;" class="telemetry-grid-4">
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">CONFIDENCE</span>
                        <strong class="telemetry-value-sm">{y_conf:.1%}</strong>
                    </div>
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">SLICK AREA</span>
                        <strong class="telemetry-value-sm">{area_val:.3f} KM²</strong>
                    </div>
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">PIXEL COUNT</span>
                        <strong class="telemetry-value-sm">{char_data.get('area_px', 0):.0f} PX</strong>
                    </div>
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">ASPECT RATIO</span>
                        <strong class="telemetry-value-sm">{char_data.get('aspect_ratio', 1.0):.2f}</strong>
                    </div>
                </div>
                <div class="telemetry-coords" style="margin-top:8px;">
                    CENTROID: ({c_px[0]:.1f}, {c_px[1]:.1f}) PX • ORIENTATION: {char_data.get('orientation_deg', 0):.1f}° • SOLIDITY: {char_data.get('solidity', 1.0):.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3. VALIDATION HIERARCHY
            st.markdown(f"""
            <div class="glass-panel" style="padding:14px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="telemetry-label" style="margin:0 !important;">MULTI-SIGNAL VALIDATION</span>
                    {render_tag('CONSENSUS')}
                </div>
                <div style="margin-top:8px;" class="telemetry-grid-4">
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">CONSENSUS</span>
                        <strong class="telemetry-value-sm">{c_agree:.0%}</strong>
                    </div>
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">DAMPING RATIO</span>
                        <strong class="telemetry-value-sm">{contrast_val:.2f}</strong>
                    </div>
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">LAND OVERLAP</span>
                        <strong class="telemetry-value-sm" style="color:{'#34d399' if land_frac < 0.20 else '#f87171'};">{land_frac:.1%}</strong>
                    </div>
                    <div class="telemetry-metric-unit">
                        <span class="telemetry-label">LOOK-ALIKE RISK</span>
                        <strong class="telemetry-value-sm">{look_risk:.0%}</strong>
                    </div>
                </div>
                <div style="margin-top:10px;">
                    {render_status_pill(val_status)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4. EVIDENCE TIMELINE (RAW -> DETECTION -> LAND/SEA -> CLASSICAL -> FINAL)
            land_pass = land_frac < 0.20
            class_pass = c_agree >= 0.15 or contrast_val < 0.75
            final_pass = (val_status in ("CONFIRMED BY MULTIPLE SIGNALS", "PROBABLE"))

            st.markdown(f"""
            <div class="evidence-timeline glass-panel">
                <span class="telemetry-label" style="margin-bottom:10px !important; display:block;">EVIDENCE TIMELINE</span>

                <!-- 1. RAW -->
                <div class="timeline-step">
                    <div class="timeline-step-badge">1. RAW</div>
                    <div class="timeline-step-content">
                        <div class="timeline-step-title">SENSOR ACQUISITION</div>
                        <div class="timeline-step-desc">{sensor_name} • {dims[1]}×{dims[0]} px • {res_m:.1f}m GSD</div>
                    </div>
                    <div class="timeline-step-status status-pass">INGESTED</div>
                </div>
                <div class="timeline-arrow">↓</div>

                <!-- 2. DETECTION -->
                <div class="timeline-step">
                    <div class="timeline-step-badge">2. DETECTION</div>
                    <div class="timeline-step-content">
                        <div class="timeline-step-title">YOLOv8 DEEP SEGMENTATION</div>
                        <div class="timeline-step-desc">Confidence: {y_conf:.1%} • Polygons: {len(all_c)} • Area: {area_val:.2f} km²</div>
                    </div>
                    <div class="timeline-step-status {'status-pass' if spill_detected_flag else 'status-fail'}">
                        {'DETECTED' if spill_detected_flag else 'NO ANOMALY'}
                    </div>
                </div>
                <div class="timeline-arrow">↓</div>

                <!-- 3. LAND/SEA FILTER -->
                <div class="timeline-step">
                    <div class="timeline-step-badge">3. LAND/SEA FILTER</div>
                    <div class="timeline-step-content">
                        <div class="timeline-step-title">MARINE DOMAIN CONSTRAINT</div>
                        <div class="timeline-step-desc">Land Overlap: {land_frac:.1%} • Constraint: {'PASSED (OPEN WATER)' if land_pass else 'FAILED (TERRESTRIAL)'}</div>
                    </div>
                    <div class="timeline-step-status {'status-pass' if land_pass else 'status-fail'}">
                        {'VALID MARINE' if land_pass else 'REJECTED LAND'}
                    </div>
                </div>
                <div class="timeline-arrow">↓</div>

                <!-- 4. CLASSICAL VALIDATION -->
                <div class="timeline-step">
                    <div class="timeline-step-badge">4. CLASSICAL VALIDATION</div>
                    <div class="timeline-step-content">
                        <div class="timeline-step-title">6-ALGORITHM CONSENSUS</div>
                        <div class="timeline-step-desc">Agreement: {c_agree:.0%} • Damping Contrast: {contrast_val:.2f}</div>
                    </div>
                    <div class="timeline-step-status {'status-pass' if class_pass else 'status-warn'}">
                        {'CONFIRMED' if class_pass else 'DISCORDANT'}
                    </div>
                </div>
                <div class="timeline-arrow">↓</div>

                <!-- 5. FINAL CONSENSUS -->
                <div class="timeline-step" style="border-left:3px solid {'#10b981' if final_pass else '#ef4444'} !important;">
                    <div class="timeline-step-badge" style="color:{'#10b981' if final_pass else '#ef4444'};">5. FINAL</div>
                    <div class="timeline-step-content">
                        <div class="timeline-step-title">INCIDENT DISPOSITION</div>
                        <div class="timeline-step-desc">{val_res.get('explanation', 'Awaiting consensus evaluation.')}</div>
                    </div>
                    <div class="timeline-step-status {'status-pass' if final_pass else 'status-fail'}">
                        {val_status}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================================
# SECTION 3: RADAR CALIBRATION SCREEN
# =========================================================================
def render_calibration_tab(final_state, is_demo, active_image=None):
    active_image = active_image or st.session_state.get("active_image_path")
    st.markdown("#### 🛰️ SAR Preprocessing & Sensor Calibration")
    st.caption("Inspect raw sensor values, speckle reduction filters, and land/sea domain separation.")

    if active_image and os.path.exists(active_image):
        img_raw = cv2.imread(active_image, cv2.IMREAD_GRAYSCALE)
        h, w = img_raw.shape[:2]

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">RASTER EXTENT</div>
                <div class="metric-value">{w} × {h}</div>
                <div class="metric-sub">PIXELS (GRD)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">SPATIAL GSD</div>
                <div class="metric-value">2.0</div>
                <div class="metric-sub">METERS / PIXEL</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">INTENSITY DYNAMICS</div>
                <div class="metric-value">{img_raw.min()} – {img_raw.max()}</div>
                <div class="metric-sub">8-BIT RADAR DN (0-255)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">MEAN BACKSCATTER</div>
                <div class="metric-value">{img_raw.mean():.1f}</div>
                <div class="metric-sub">STD DEV: {img_raw.std():.1f} DN</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Interactive Preprocessing Filter Preview
        st.markdown("##### Filter Comparison")
        f_tabs = st.tabs(["Raw Input", "Median Despeckle", "Lee Filter (7x7)", "CLAHE Enhancement"])

        with f_tabs[0]:
            st.image(img_raw, use_container_width=True, caption="Raw Sentinel-1 Amplitude Swath")
        with f_tabs[1]:
            med = cv2.medianBlur(img_raw, 5)
            st.image(med, use_container_width=True, caption="Median Filter (5x5 kernel) — Suppresses salt-and-pepper noise")
        with f_tabs[2]:
            from sar.preprocessing import lee_filter
            lee = lee_filter(img_raw, kernel_size=7)
            st.image(lee, use_container_width=True, caption="Lee Speckle Filter (7x7 kernel) — Preserves edges while attenuating multiplicative speckle")
        with f_tabs[3]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl_img = clahe.apply(img_raw)
            st.image(cl_img, use_container_width=True, caption="CLAHE (Clip Limit 2.0) — Enhances local dark-spot contrast")
    else:
        st.info("No active SAR image loaded.")


# =========================================================================
# SECTION 4: AIS CORRELATION & CANDIDATE VESSELS SCREEN
# =========================================================================
def render_ais_tab(final_state, is_demo):
    st.markdown("#### 🚢 AIS Candidate Vessel Intelligence & Spatiotemporal Correlation")
    st.caption("Operational multi-factor spatiotemporal correlation fusing commercial AIS fleet telemetry with estimated slick origin.")

    # Methodology and Non-Liability Disclaimer
    st.markdown("""
    <div class="glass-panel" style="border-left:3px solid #00e5ff !important; padding:12px 16px; margin-bottom:14px; font-size:12px; color:#94a3b8;">
        ⚖️ <strong style="color:#f8fafc;">LEGAL DISCLOSURE:</strong> Candidate vessel associations reflect mathematical spatiotemporal alignment between vessel trajectories and estimated spill origin zones. 
        They do <strong>NOT</strong> constitute legal proof of liability, operational negligence, or regulatory sanction.
    </div>
    """, unsafe_allow_html=True)

    if final_state and final_state.get("candidates_done"):
        candidates = final_state.get("candidate_scores", [])
        tracks_data = final_state.get("ais_tracks", {}) or {}
        ais_mode = final_state.get("ais_data_mode", "UNKNOWN")
        spill_lat = final_state.get("spill_lat", DEMO_SPILL_LAT)
        spill_lon = final_state.get("spill_lon", DEMO_SPILL_LON)
        source_lat = final_state.get("source_lat", spill_lat)
        source_lon = final_state.get("source_lon", spill_lon)
        det_ts = final_state.get("detection_timestamp")
        base_time = datetime.fromisoformat(det_ts) if det_ts else datetime(2026, 9, 14, 15, 30, tzinfo=timezone.utc)

        # Extract available vessel types dynamically from actual track records
        all_v_types = set()
        if isinstance(tracks_data, dict):
            for m_k, r_list in tracks_data.items():
                if r_list and isinstance(r_list[0], dict) and r_list[0].get("vessel_type"):
                    all_v_types.add(r_list[0].get("vessel_type"))
        elif isinstance(tracks_data, list):
            for t in tracks_data:
                pts = t.get("points", [])
                if pts and isinstance(pts[0], dict) and pts[0].get("vessel_type"):
                    all_v_types.add(pts[0].get("vessel_type"))
        v_type_options = ["All Vessel Types"] + sorted(list(all_v_types))

        # Helper: Extract interpolated vessel telemetry at current scrubber time
        def get_vessel_telemetry_at_time(mmsi_val, target_time):
            recs = []
            if isinstance(tracks_data, dict) and mmsi_val in tracks_data:
                recs = tracks_data[mmsi_val]
            elif isinstance(tracks_data, list):
                for t in tracks_data:
                    if str(t.get("mmsi")) == str(mmsi_val):
                        recs = t.get("points", [])
                        break
            if not recs or not isinstance(recs[0], dict):
                return {
                    "lat": spill_lat, "lon": spill_lon, "speed_knots": 0.0, "course": 0.0,
                    "timestamp": "N/A", "vessel_type": "Commercial Vessel", "nav_status": "Underway"
                }
            best_r = recs[0]
            min_dt = float("inf")
            for r in recs:
                if isinstance(r, dict) and "timestamp" in r:
                    try:
                        r_t = datetime.fromisoformat(r["timestamp"])
                        dt = abs((r_t - target_time).total_seconds())
                        if dt < min_dt:
                            min_dt = dt
                            best_r = r
                    except Exception:
                        pass
            return {
                "lat": float(best_r.get("lat", spill_lat)),
                "lon": float(best_r.get("lon", spill_lon)),
                "speed_knots": float(best_r.get("speed_knots", 0.0)),
                "course": float(best_r.get("heading", best_r.get("course_over_ground", 0.0))),
                "timestamp": str(best_r.get("timestamp", "")),
                "vessel_type": str(best_r.get("vessel_type", "Commercial")),
                "nav_status": str(best_r.get("navigation_status", "Underway")),
            }

        # ── Control & Filter Bar (Only supported filters) ──
        st.markdown("""
        <div class="glass-panel" style="padding:10px 14px; margin-bottom:12px;">
            <div style="font-size:10px; font-weight:700; color:#38bdf8; font-family:'JetBrains Mono'; letter-spacing:0.8px; margin-bottom:6px;">
                🛰️ AIS INTELLIGENCE FILTER & TEMPORAL CORRIDOR CONTROLS
            </div>
        </div>
        """, unsafe_allow_html=True)

        fc1, fc2, fc3, fc4 = st.columns([3, 2, 2, 3])
        with fc1:
            st.caption("🔍 SEARCH CANDIDATE (NAME / MMSI)")
            search_kw = st.text_input("Search Vessel", placeholder="e.g. Falcon or 419000", key="ais_search_filter_input", label_visibility="collapsed")
        with fc2:
            st.caption("🚢 VESSEL TYPE")
            sel_type = st.selectbox("Vessel Type", v_type_options, key="ais_type_filter_input", label_visibility="collapsed")
        with fc3:
            st.caption("⚖️ ASSOCIATION TIER")
            sel_tier = st.selectbox("Association Tier", ["All Association Tiers", "High Association (≥70%)", "Moderate Association (≥40%)", "Low Association (<40%)"], key="ais_tier_filter_input", label_visibility="collapsed")
        with fc4:
            st.caption("⏱️ TIME RANGE / SCRUBBER")
            time_scrub = st.slider("Time Range", min_value=-180, max_value=60, value=st.session_state.get("timeline_min", 0), step=15, format="%+d min", key="ais_timeline_scrub_input", label_visibility="collapsed")
            st.session_state["timeline_min"] = time_scrub

        current_scrub_time = base_time + timedelta(minutes=time_scrub)

        # Apply Filters
        filtered_candidates = []
        for cand in candidates:
            mmsi = str(cand.get("mmsi", ""))
            name = str(cand.get("name", "Unknown"))
            score = float(cand.get("score", 0.0))

            v_telemetry = get_vessel_telemetry_at_time(mmsi, current_scrub_time)
            v_type = v_telemetry.get("vessel_type", "Commercial")

            # Search filter
            if search_kw.strip():
                kw = search_kw.strip().lower()
                if kw not in name.lower() and kw not in mmsi.lower():
                    continue

            # Vessel type filter
            if sel_type != "All Vessel Types" and v_type.lower() != sel_type.lower():
                continue

            # Association score tier filter
            if sel_tier == "High Association (≥70%)" and score < 70:
                continue
            elif sel_tier == "Moderate Association (≥40%)" and (score < 40 or score >= 70):
                continue
            elif sel_tier == "Low Association (<40%)" and score >= 40:
                continue

            filtered_candidates.append(cand)

        col_ais_map, col_ais_panel = st.columns([13, 11])

        # ── LEFT COLUMN: MAIN VIEW (MAP + AIS TRACKS) ──
        with col_ais_map:
            st.markdown("##### 🗺️ Main View: Interactive Map & AIS Trajectory Corridor")

            # Direct Target Selector / Camera Flight dropdown
            map_c_opts = ["Full Fleet Overview (All Tracks)"] + [f"{c['name']} (MMSI: {c['mmsi']} | {c.get('score', 0):.0f}%)" for c in filtered_candidates]
            sel_idx = 0
            if st.session_state.get("selected_vessel_mmsi"):
                for idx, c in enumerate(filtered_candidates):
                    if str(c["mmsi"]) == str(st.session_state.get("selected_vessel_mmsi")):
                        sel_idx = idx + 1
                        break

            m_pick = st.selectbox("Direct Target Acquisition", map_c_opts, index=sel_idx, key="ais_tab_quick_selector", label_visibility="collapsed")
            if m_pick != "Full Fleet Overview (All Tracks)":
                m_match = next((c for c in filtered_candidates if f"{c['name']} (MMSI: {c['mmsi']} | {c.get('score', 0):.0f}%)" == m_pick), None)
                if m_match and str(st.session_state.get("selected_vessel_mmsi")) != str(m_match["mmsi"]):
                    st.session_state["selected_vessel_mmsi"] = m_match["mmsi"]
                    st.session_state["focus_target"] = "vessel"
                    st.rerun()
            elif m_pick == "Full Fleet Overview (All Tracks)" and st.session_state.get("selected_vessel_mmsi") is not None:
                st.session_state["selected_vessel_mmsi"] = None
                st.session_state["focus_target"] = None
                st.rerun()

            active_mmsi_label = f"TARGET: {st.session_state.get('selected_vessel_mmsi')}" if st.session_state.get("selected_vessel_mmsi") else "FLEET SURVEY"
            st.markdown(f"""
            <div class="map-tactical-header">
                <div><span class="status-pulse-sm"></span><span class="map-tactical-title">AIS COMMERCIAL FLEET TRACKING CORRIDOR // EPSG:4326</span></div>
                <div>MATCHING CANDIDATES: {len(filtered_candidates)} OF {len(candidates)} • FEED: {ais_mode} • {active_mmsi_label}</div>
            </div>
            """, unsafe_allow_html=True)

            fmap_ais = build_investigation_map(
                final_state,
                slider_minutes=time_scrub,
                selected_vessel_mmsi=st.session_state.get("selected_vessel_mmsi"),
                mode="AIS",
            )
            st_folium(fmap_ais, height=600, use_container_width=True, key="ais_workspace_folium_map", returned_objects=[])

            # Map tactical indicators & telemetry strip
            st.markdown(f"""
            <div class="glass-panel" style="padding:10px 14px; margin-top:8px; display:flex; justify-content:space-between; align-items:center; font-size:11px; font-family:'JetBrains Mono';">
                <div><span style="color:#00e5ff; font-weight:700;">● PRIMARY TARGET:</span> Pulsing Halo + AntPath Trajectory</div>
                <div><span style="color:#64748b; font-weight:700;">○ UNRELATED FLEET:</span> Dimmed (14% Opacity)</div>
                <div><span style="color:#ef4444; font-weight:700;">■ SPILL SLICK:</span> SAR Detection Centroid</div>
                <div><span style="color:#10b981; font-weight:700;">⇢ SOURCE VECTOR:</span> Backtrack Proximity</div>
            </div>
            """, unsafe_allow_html=True)

        # ── RIGHT COLUMN: SECONDARY VIEW (CANDIDATE VESSEL PANEL) ──
        with col_ais_panel:
            st.markdown("##### 📋 Secondary View: Candidate Vessel Panel & Evidence Dossier")
            st.caption(f"Showing {len(filtered_candidates)} matching candidate vessels ranked by multi-factor association evidence.")

            if not filtered_candidates:
                st.warning("No candidate vessels match the current search or filter criteria.")
                if st.button("Reset All Filters", key="ais_reset_filters_btn", type="secondary", use_container_width=True):
                    st.session_state["ais_search_filter_input"] = ""
                    st.session_state["ais_type_filter_input"] = "All Vessel Types"
                    st.session_state["ais_tier_filter_input"] = "All Association Tiers"
                    st.rerun()
            else:
                # If a specific vessel is currently selected, display its comprehensive Evidence Dossier first
                selected_cand = None
                if st.session_state.get("selected_vessel_mmsi"):
                    selected_cand = next((c for c in filtered_candidates if str(c.get("mmsi")) == str(st.session_state.get("selected_vessel_mmsi"))), None)
                    if not selected_cand:
                        selected_cand = next((c for c in candidates if str(c.get("mmsi")) == str(st.session_state.get("selected_vessel_mmsi"))), None)

                if selected_cand:
                    s_mmsi = str(selected_cand.get("mmsi", ""))
                    s_name = selected_cand.get("name", "Unknown Vessel")
                    s_score = selected_cand.get("score", 0.0)
                    s_bdown = selected_cand.get("breakdown", {})
                    s_telemetry = get_vessel_telemetry_at_time(s_mmsi, current_scrub_time)
                    s_lat, s_lon = s_telemetry["lat"], s_telemetry["lon"]
                    s_speed = s_telemetry["speed_knots"]
                    s_course = s_telemetry["course"]
                    s_type = s_telemetry["vessel_type"]
                    s_ts_str = s_telemetry["timestamp"]
                    s_dist_spill = haversine_km(s_lat, s_lon, spill_lat, spill_lon)
                    s_dist_source = selected_cand.get("min_distance_km", haversine_km(s_lat, s_lon, source_lat, source_lon))
                    s_time_match = selected_cand.get("time_match", False)

                    score_css = "vessel-score-high" if s_score >= 70 else ("vessel-score-med" if s_score >= 40 else "vessel-score-low")
                    bar_color = "#f87171" if s_score >= 70 else ("#fbbf24" if s_score >= 40 else "#64748b")

                    st.markdown(f"""
                    <div class="vessel-card vessel-card-active glass-panel">
                        <div class="vessel-header">
                            <div>
                                <span class="telemetry-micro-label">🎯 TARGET ACQUIRED // PRIMARY CANDIDATE VESSEL</span>
                                <div class="vessel-name" style="font-size:17px; margin-top:2px;">🚢 {s_name}</div>
                                <div style="display:flex; gap:8px; align-items:center; margin-top:4px;">
                                    <span class="vessel-mmsi">MMSI: {s_mmsi}</span>
                                    <span class="vessel-type-tag">{s_type.upper()}</span>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <span class="telemetry-micro-label">ASSOCIATION SCORE</span>
                                <div class="vessel-score {score_css}">{s_score:.0f}<span style="font-size:12px; color:#64748b;">/100</span></div>
                            </div>
                        </div>
                        <div class="bar-bg" style="height:7px;">
                            <div class="bar-fill" style="width:{s_score}%; background:{bar_color};"></div>
                        </div>

                        <div style="margin-top:14px; font-size:10px; font-weight:700; color:#38bdf8; font-family:'JetBrains Mono'; letter-spacing:0.8px;">
                            TELEMETRY SNAPSHOT (T{time_scrub:+d} MIN RELATIVE TO SLICK ACQUISITION)
                        </div>

                        <div class="telemetry-grid-4" style="margin-top:8px;">
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">POSITION</span>
                                <strong class="telemetry-value-sm" style="font-size:11px;">{s_lat:.4f}°N, {s_lon:.4f}°E</strong>
                            </div>
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">TIMESTAMP</span>
                                <strong class="telemetry-value-sm" style="font-size:11px;">{s_ts_str[11:19] if len(s_ts_str) >= 19 else s_ts_str} UTC</strong>
                            </div>
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">SPEED</span>
                                <strong class="telemetry-value-sm">{s_speed:.1f} KN</strong>
                            </div>
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">COURSE</span>
                                <strong class="telemetry-value-sm">{s_course:.0f}° TRUE</strong>
                            </div>
                        </div>

                        <div class="telemetry-grid-4" style="margin-top:8px;">
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">DISTANCE FROM SPILL</span>
                                <strong class="telemetry-value-sm">{s_dist_spill:.1f} KM</strong>
                            </div>
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">SOURCE PROXIMITY</span>
                                <strong class="telemetry-value-sm">{s_dist_source:.1f} KM</strong>
                            </div>
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">TEMPORAL RELATION</span>
                                <strong class="telemetry-value-sm" style="color:{'#34d399' if s_time_match else '#f87171'};">
                                    {'COINCIDENT' if s_time_match else 'DISCORDANT'}
                                </strong>
                            </div>
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">TRAJECTORY CONSISTENCY</span>
                                <strong class="telemetry-value-sm">{s_bdown.get('trajectory', 0.0):.0f}%</strong>
                            </div>
                        </div>

                        <div style="margin-top:16px; border-top:1px solid rgba(255,255,255,0.08); padding-top:12px;">
                            <span class="telemetry-micro-label">ASSOCIATION EVIDENCE BREAKDOWN</span>
                            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; margin-top:6px;">
                                <div style="background:rgba(14,22,42,0.6); padding:6px 8px; border-radius:4px; border:1px solid rgba(255,255,255,0.06); font-family:'JetBrains Mono'; font-size:10px;">
                                    <span style="color:#94a3b8;">SPATIAL:</span> <strong style="color:#f8fafc;">{s_bdown.get('spatial', 0.0):.0f}%</strong>
                                </div>
                                <div style="background:rgba(14,22,42,0.6); padding:6px 8px; border-radius:4px; border:1px solid rgba(255,255,255,0.06); font-family:'JetBrains Mono'; font-size:10px;">
                                    <span style="color:#94a3b8;">TEMPORAL:</span> <strong style="color:#f8fafc;">{s_bdown.get('temporal', 0.0):.0f}%</strong>
                                </div>
                                <div style="background:rgba(14,22,42,0.6); padding:6px 8px; border-radius:4px; border:1px solid rgba(255,255,255,0.06); font-family:'JetBrains Mono'; font-size:10px;">
                                    <span style="color:#94a3b8;">TRAJECTORY:</span> <strong style="color:#f8fafc;">{s_bdown.get('trajectory', 0.0):.0f}%</strong>
                                </div>
                                <div style="background:rgba(14,22,42,0.6); padding:6px 8px; border-radius:4px; border:1px solid rgba(255,255,255,0.06); font-family:'JetBrains Mono'; font-size:10px;">
                                    <span style="color:#94a3b8;">HEADING:</span> <strong style="color:#f8fafc;">{s_bdown.get('heading', 0.0):.0f}%</strong>
                                </div>
                                <div style="background:rgba(14,22,42,0.6); padding:6px 8px; border-radius:4px; border:1px solid rgba(255,255,255,0.06); font-family:'JetBrains Mono'; font-size:10px;">
                                    <span style="color:#94a3b8;">SPEED:</span> <strong style="color:#f8fafc;">{s_bdown.get('speed', 0.0):.0f}%</strong>
                                </div>
                                <div style="background:rgba(14,22,42,0.6); padding:6px 8px; border-radius:4px; border:1px solid rgba(255,255,255,0.06); font-family:'JetBrains Mono'; font-size:10px;">
                                    <span style="color:#94a3b8;">DRIFT:</span> <strong style="color:#f8fafc;">{s_bdown.get('drift_consistency', 0.0):.0f}%</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Evidence Bullets & Uncertainty Notes
                    s_ev_list = selected_cand.get("evidence_summary", [])
                    s_unc_list = selected_cand.get("uncertainty_notes", [])
                    if s_ev_list or s_unc_list:
                        with st.expander("🔎 Detailed Spatiotemporal Evidence & Uncertainty Audit", expanded=True):
                            if s_ev_list:
                                st.markdown("<strong style='font-size:11px; color:#38bdf8; font-family:var(--font-mono);'>CORROBORATING EVIDENCE POINTS:</strong>", unsafe_allow_html=True)
                                for ev in s_ev_list:
                                    st.markdown(f"<div class='evidence-bullet'><span>✓</span> <span>{ev}</span></div>", unsafe_allow_html=True)
                            if s_unc_list:
                                st.markdown("<strong style='font-size:11px; color:#fbbf24; font-family:var(--font-mono); margin-top:8px; display:block;'>UNCERTAINTY & SENSOR BOUNDS:</strong>", unsafe_allow_html=True)
                                for unc in s_unc_list:
                                    st.markdown(f"<div class='uncertainty-bullet'><span>⚠️</span> <span>{unc}</span></div>", unsafe_allow_html=True)

                    if st.button("✕ Deselect Target (View Full Fleet)", key="ais_deselect_target_btn", use_container_width=True, type="secondary"):
                        st.session_state["selected_vessel_mmsi"] = None
                        st.session_state["focus_target"] = None
                        st.rerun()

                    st.markdown("<hr style='border:0; border-top:1px solid rgba(255,255,255,0.08); margin:16px 0 12px 0;'>", unsafe_allow_html=True)
                    st.markdown("##### 👥 Other Candidate Vessels in Corridor")

                # List remaining candidates
                for i, cand in enumerate(filtered_candidates):
                    c_mmsi = str(cand.get("mmsi", ""))
                    if selected_cand and c_mmsi == str(selected_cand.get("mmsi")):
                        continue  # Already shown above

                    score = cand.get("score", 0.0)
                    score_css = "vessel-score-high" if score >= 70 else ("vessel-score-med" if score >= 40 else "vessel-score-low")
                    bar_color = "#f87171" if score >= 70 else ("#fbbf24" if score >= 40 else "#64748b")
                    bdown = cand.get("breakdown", {})
                    c_telemetry = get_vessel_telemetry_at_time(c_mmsi, current_scrub_time)
                    c_lat, c_lon = c_telemetry["lat"], c_telemetry["lon"]
                    c_speed = c_telemetry["speed_knots"]
                    c_course = c_telemetry["course"]
                    c_type = c_telemetry["vessel_type"]
                    c_dist_spill = haversine_km(c_lat, c_lon, spill_lat, spill_lon)
                    c_dist_source = cand.get("min_distance_km", haversine_km(c_lat, c_lon, source_lat, source_lon))

                    dim_class = "vessel-dimmed" if (st.session_state.get("selected_vessel_mmsi") and str(st.session_state.get("selected_vessel_mmsi")) != c_mmsi) else ""

                    st.markdown(f"""
                    <div class="vessel-card glass-panel {dim_class}">
                        <div class="vessel-header">
                            <div>
                                <span class="telemetry-micro-label">CANDIDATE #{i+1}</span>
                                <div class="vessel-name">🚢 {cand.get('name', 'UNKNOWN')}</div>
                                <div style="display:flex; gap:6px; align-items:center; margin-top:2px;">
                                    <span class="vessel-mmsi">MMSI: {c_mmsi}</span>
                                    <span class="vessel-type-tag">{c_type.upper()}</span>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <span class="telemetry-micro-label">ASSOCIATION</span>
                                <div class="vessel-score {score_css}">{score:.0f}<span style="font-size:12px; color:#64748b;">/100</span></div>
                            </div>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width:{score}%; background:{bar_color};"></div>
                        </div>
                        <div class="telemetry-grid-4">
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">POSITION</span>
                                <strong class="telemetry-value-sm" style="font-size:10.5px;">{c_lat:.3f}°N, {c_lon:.3f}°E</strong>
                            </div>
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">SPEED & COURSE</span>
                                <strong class="telemetry-value-sm">{c_speed:.1f} KN • {c_course:.0f}°</strong>
                            </div>
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">SOURCE PROXIMITY</span>
                                <strong class="telemetry-value-sm">{c_dist_source:.1f} KM</strong>
                            </div>
                            <div class="telemetry-metric-unit">
                                <span class="telemetry-label">TRAJECTORY</span>
                                <strong class="telemetry-value-sm">{bdown.get('trajectory', 0.0):.0f}%</strong>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_lbl = f"🎯 Inspect & Fly Map to {cand.get('name', c_mmsi)}"
                    if st.button(btn_lbl, key=f"foc_btn_ais_cand_{c_mmsi}", use_container_width=True, type="primary" if score >= 70 else "secondary"):
                        st.session_state["selected_vessel_mmsi"] = cand["mmsi"]
                        st.session_state["focus_target"] = "vessel"
                        st.rerun()
    else:
        st.info("Execute pipeline to generate candidate vessel association rankings.")
        if st.button("Run Pipeline Now", key="ais_tab_empty_run_btn", type="primary", use_container_width=True):
            st.session_state["trigger_pipeline_run"] = True
            st.rerun()


# =========================================================================
# SECTION 5: DRIFT INTELLIGENCE WORKSPACE
# =========================================================================
def render_drift_tab(final_state, is_demo):
    st.markdown("#### ⏱️ Drift Intelligence & Hydrodynamic Advection")
    st.caption("Euler advection hindcast, forward drift trajectory projection, and dynamic spatial dispersion envelopes.")

    if "drift_h" not in st.session_state:
        st.session_state["drift_h"] = 0.0
    if "drift_play" not in st.session_state:
        st.session_state["drift_play"] = False

    if final_state and final_state.get("hindcast_done"):
        hindcast = final_state.get("hindcast_result", {})
        c_speed = hindcast.get("current_speed_ms", 0.48)
        c_bearing = hindcast.get("current_bearing_deg", 118.0)
        w_speed = hindcast.get("wind_speed_ms", 6.2)
        w_bearing = hindcast.get("wind_bearing_deg", 135.0)
        w_factor = hindcast.get("wind_factor", 0.03)

        # Net drift vector formulation
        cx = c_speed * math.sin(math.radians(c_bearing))
        cy = c_speed * math.cos(math.radians(c_bearing))
        wx = w_factor * w_speed * math.sin(math.radians(w_bearing))
        wy = w_factor * w_speed * math.cos(math.radians(w_bearing))
        vx = cx + wx
        vy = cy + wy
        net_drift_speed = math.sqrt(vx**2 + vy**2)
        net_drift_bearing = math.degrees(math.atan2(vx, vy)) % 360
        reverse_bearing = (net_drift_bearing + 180) % 360
        net_speed_knots = net_drift_speed * 1.94384

        spill_lat = final_state.get("spill_lat", DEMO_SPILL_LAT)
        spill_lon = final_state.get("spill_lon", DEMO_SPILL_LON)

        # ── 1. Oceanographic Model & Scientific Restraint Disclosure ──
        st.markdown(f"""
        <div class="glass-panel" style="padding:14px 18px; margin-bottom:14px; border-left:4px solid #38bdf8;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="telemetry-label" style="color:#38bdf8 !important;">OCEANOGRAPHIC DRIFT MODEL & SENSOR DISCLOSURE</span>
                <div style="display:flex; gap:6px;">
                    <span class="data-tag tag-simulated">SIMULATED REGIONAL VECTOR</span>
                    <span class="data-tag tag-official">EULER NUMERICAL SCHEME</span>
                </div>
            </div>
            <div style="font-size:12px; color:#cbd5e1; margin-top:6px; line-height:1.5;">
                <strong>Deterministic Advection Model:</strong> Integrates constant regional surface current (<code style="color:#00e5ff;">{c_speed:.2f} m/s @ {c_bearing:.0f}° True</code>) with 3.0% empirical wind leeway (<code style="color:#38bdf8;">{w_speed:.1f} m/s @ {w_bearing:.0f}° True</code>) via discrete Euler numerical advection. Net advection vector: <code style="color:#34d399;">{net_drift_speed:.2f} m/s ({net_speed_knots:.2f} kn) @ {net_drift_bearing:.0f}° True</code>.
                <br><span style="color:#fbbf24;">⚠️ <strong>Scientific Disclosure:</strong></span> Offline operational demonstration evaluates deterministic transport across a simulated constant velocity vector field. Real-time 3D baroclinic oceanographic assimilation (Copernicus Marine Service / HYCOM) is inactive. Uncertainty envelopes expand linearly with advection duration (dispersion growth rate 0.35 km/km advected).
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 2. Playback & Timeline Scrubber Suite ──
        st.markdown("""
        <div class="glass-panel" style="padding:10px 14px; margin-bottom:10px;">
            <div style="font-size:10px; font-weight:700; color:#38bdf8; font-family:'JetBrains Mono'; letter-spacing:0.8px;">
                ⏱️ TIMELINE MILESTONE ACQUISITION & CONTINUOUS FORENSIC SCRUBBER
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row A: Play / Step controls
        ctl_c1, ctl_c2, ctl_c3, ctl_c4 = st.columns([3, 2, 2, 2])
        with ctl_c1:
            is_playing = st.session_state.get("drift_play", False)
            if is_playing:
                if st.button("⏸ PAUSE SIMULATION", key="btn_drift_pause", use_container_width=True, type="primary"):
                    st.session_state["drift_play"] = False
                    st.rerun()
            else:
                if st.button("▶ PLAY SIMULATION", key="btn_drift_play", use_container_width=True, type="secondary"):
                    st.session_state["drift_play"] = True
                    st.rerun()
        with ctl_c2:
            if st.button("◀ -1.0h", key="btn_drift_step_back", use_container_width=True):
                st.session_state["drift_h"] = max(-12.0, round(st.session_state["drift_h"] - 1.0, 1))
                st.session_state["drift_play"] = False
                st.rerun()
        with ctl_c3:
            if st.button("▶ +1.0h", key="btn_drift_step_fwd", use_container_width=True):
                st.session_state["drift_h"] = min(24.0, round(st.session_state["drift_h"] + 1.0, 1))
                st.session_state["drift_play"] = False
                st.rerun()
        with ctl_c4:
            if st.button("🎯 NOW (T=0)", key="btn_drift_reset_now", use_container_width=True):
                st.session_state["drift_h"] = 0.0
                st.session_state["drift_play"] = False
                st.rerun()

        # Row B: 6 Discrete Milestone Buttons
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        milestones = [
            (-12.0, "⏪ T-12h", m_col1),
            (-6.0,  "◀ T-6h",   m_col2),
            (0.0,   "🎯 NOW",   m_col3),
            (6.0,   "▶ T+6h",   m_col4),
            (12.0,  "⏩ T+12h", m_col5),
            (24.0,  "⏭ T+24h", m_col6),
        ]
        for m_val, m_label, col in milestones:
            with col:
                is_active_milestone = abs(st.session_state["drift_h"] - m_val) < 0.25
                if st.button(m_label, key=f"btn_ms_{m_val}", use_container_width=True, type="primary" if is_active_milestone else "secondary"):
                    st.session_state["drift_h"] = m_val
                    st.session_state["drift_play"] = False
                    st.rerun()

        # Row C: Continuous Timeline Slider
        scrub_val = st.slider(
            "Forensic Advection Horizon",
            min_value=-12.0,
            max_value=24.0,
            value=float(st.session_state["drift_h"]),
            step=0.5,
            format="%+.1f hrs",
            key="drift_slider_input",
            label_visibility="collapsed",
        )
        if scrub_val != st.session_state["drift_h"]:
            st.session_state["drift_h"] = scrub_val

        # Temporal and spatial calculations for active hour
        cur_h = float(st.session_state["drift_h"])
        ts_str = final_state.get("detection_timestamp", "2026-09-14T15:30:00+00:00")
        try:
            base_dt = datetime.fromisoformat(ts_str)
            sim_dt = base_dt + timedelta(hours=cur_h)
            sim_time_str = sim_dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            sim_time_str = "2026-09-14 15:30 UTC"

        if abs(cur_h) < 0.01:
            active_lat, active_lon = spill_lat, spill_lon
            active_dist = 0.0
            active_unc = 0.8
            phase_label = "OBSERVED SLICK // SAR PASS (NOW)"
            phase_tag = "OBSERVED"
        elif cur_h < 0:
            active_dist = net_drift_speed * (abs(cur_h) * 3600) / 1000.0
            active_lat, active_lon = destination_point(spill_lat, spill_lon, reverse_bearing, active_dist)
            active_unc = 0.8 + active_dist * 0.35
            phase_label = f"HISTORICAL HINDCAST // T{cur_h:+0.1f}h PRIOR TO DETECTION"
            phase_tag = "HINDCAST"
        else:
            active_dist = net_drift_speed * (cur_h * 3600) / 1000.0
            active_lat, active_lon = destination_point(spill_lat, spill_lon, net_drift_bearing, active_dist)
            active_unc = 0.8 + active_dist * 0.35
            phase_label = f"FORWARD DRIFT FORECAST // T{cur_h:+0.1f}h PROJECTION"
            phase_tag = "PREDICTED"

        # Nearest sensitive shoreline asset
        nearest_shore_dist = float("inf")
        nearest_shore_name = "Coastline"
        for sa in (CHENNAI_SCENARIO.sensitive_areas or []):
            d = haversine_km(active_lat, active_lon, sa["lat"], sa["lon"])
            if d < nearest_shore_dist:
                nearest_shore_dist = d
                nearest_shore_name = sa["name"]

        # ── 3. Map Tactical Status Header ──
        st.markdown(f"""
        <div class="map-tactical-header">
            <div>
                <span class="status-pulse-sm"></span>
                <span class="map-tactical-title">DRIFT INTELLIGENCE // PRIMARY CANVAS [{phase_label}]</span>
            </div>
            <div>
                SIMULATION EPOCH: {sim_time_str} • DELTA: {cur_h:+0.1f}h • POS: {active_lat:.3f}°N, {active_lon:.3f}°E • SPREAD: ±{active_unc:.1f} KM
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4. Primary Map Canvas (Rock-Solid Stable Camera at Corridor Center) ──
        fmap_drift = build_investigation_map(
            final_state,
            mode="DRIFT",
            drift_hours=cur_h,
        )
        st_folium(fmap_drift, height=580, use_container_width=True, key="drift_intelligence_folium_map", returned_objects=[])

        # ── 5. Live Dynamic Telemetry Grid (Smooth Real-Time Updates) ──
        tm1, tm2, tm3, tm4, tm5, tm6 = st.columns(6)
        with tm1:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">SIMULATION EPOCH</div>
                <div class="metric-value" style="font-size:16px;">{sim_time_str[-9:]}</div>
                <div class="metric-sub">{sim_time_str[:10]} // T{cur_h:+0.1f}H</div>
            </div>
            """, unsafe_allow_html=True)
        with tm2:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">ACTIVE SLICK COORDS</div>
                <div class="metric-value" style="font-size:16px;">{active_lat:.4f}°N</div>
                <div class="metric-sub">{active_lon:.4f}°E // EPSG:4326</div>
            </div>
            """, unsafe_allow_html=True)
        with tm3:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">NET ADVECTION DISTANCE</div>
                <div class="metric-value">{active_dist:.1f} KM</div>
                <div class="metric-sub">{active_dist * 0.539957:.1f} NM FROM DETECTION</div>
            </div>
            """, unsafe_allow_html=True)
        with tm4:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">DISPERSION ENVELOPE</div>
                <div class="metric-value">±{active_unc:.1f} KM</div>
                <div class="metric-sub">AREA: ~{math.pi * active_unc**2:.1f} KM²</div>
            </div>
            """, unsafe_allow_html=True)
        with tm5:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">NET ADVECTION VECTOR</div>
                <div class="metric-value">{net_drift_speed:.2f} M/S</div>
                <div class="metric-sub">{net_speed_knots:.2f} KN @ {net_drift_bearing:.0f}° TRUE</div>
            </div>
            """, unsafe_allow_html=True)
        with tm6:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">SHORELINE PROXIMITY</div>
                <div class="metric-value">{nearest_shore_dist:.1f} KM</div>
                <div class="metric-sub">NEAREST: {nearest_shore_name.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── 6. Deterministic Animation Playback Loop ──
        if st.session_state.get("drift_play", False):
            time.sleep(0.35)
            next_step = round(cur_h + 1.0, 1)
            if next_step > 24.0:
                st.session_state["drift_play"] = False
                st.session_state["drift_h"] = 24.0
            else:
                st.session_state["drift_h"] = next_step
            st.rerun()

    else:
        st.info("Execute pipeline to generate hydrodynamic drift vectors, hindcast origin, and forward forecast cones.")
        if st.button("⚡ Run Intelligence Pipeline Now", key="drift_tab_run_empty_btn", type="primary", use_container_width=True):
            st.session_state["trigger_pipeline_run"] = True
            st.rerun()


# =========================================================================
# SECTION 6: COASTAL RISK & SHORELINE VULNERABILITY SCREEN
# =========================================================================
def render_risk_tab(final_state, is_demo):
    st.markdown("#### 🏖️ Coastal Impact & Environmental Asset Vulnerability")
    st.caption("Evaluates shoreline approach vector, sensitive ecological zones, and tactical countermeasure rules.")

    if final_state and final_state.get("risk_done"):
        risk = final_state.get("risk_assessment", {})
        risk_level = risk.get("level", "UNKNOWN")
        coastal = final_state.get("coastal_impact", {})

        # Risk Banner
        r_css = "color:#ef4444;" if risk_level == "CRITICAL" else ("color:#f59e0b;" if risk_level in ("HIGH", "MEDIUM") else "color:#34d399;")
        st.markdown(f"""
        <div class="glass-panel" style="padding:18px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="telemetry-label">INCIDENT THREAT ASSESSMENT TIER</div>
                    <div style="font-size:24px; font-weight:800; font-family:'JetBrains Mono'; {r_css}">{risk_level} — SCORE: {risk.get('overall_score', 0):.0f}/100</div>
                </div>
                {render_tag('PREDICTED')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Coastal Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">SHORELINE PROXIMITY</div>
                <div class="metric-value">{coastal.get('shortest_distance_to_coast_km', 0.0):.1f} KM</div>
                <div class="metric-sub">NEAREST: {coastal.get('nearest_shoreline_point', {}).get('name', 'N/A').upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            eta = coastal.get("eta_to_coast_hours")
            eta_str = f"{eta:.1f} HRS" if eta else "NO LANDFALL"
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">LANDFALL ETA</div>
                <div class="metric-value">{eta_str}</div>
                <div class="metric-sub">TRAJECTORY PROJECTION</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">COASTAL VULNERABILITY</div>
                <div class="metric-value">{coastal.get('coastal_vulnerability_score', 0.0):.0f}/100</div>
                <div class="metric-sub">TIER: {coastal.get('risk_tier', 'LOW')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card glass-panel">
                <div class="telemetry-label">THREATENED ASSETS</div>
                <div class="metric-value">{coastal.get('threatened_assets_count', 0)}</div>
                <div class="metric-sub">ECOLOGICAL & INFRASTRUCTURE</div>
            </div>
            """, unsafe_allow_html=True)

        # Geospatial Risk & Threat Map
        st.markdown(f"""
        <div class="map-tactical-header">
            <div><span class="status-pulse-sm"></span><span class="map-tactical-title">TACTICAL COASTAL RISK & SENSITIVITY CORRIDOR // EPSG:4326</span></div>
            <div>NEAREST: {coastal.get('nearest_shoreline_point', {}).get('name', 'SHORELINE').upper()} • ETA: {eta_str}</div>
        </div>
        """, unsafe_allow_html=True)
        fmap_risk = build_investigation_map(
            final_state,
            slider_minutes=st.session_state.get("timeline_min", 0),
            selected_vessel_mmsi=st.session_state.get("selected_vessel_mmsi"),
            mode="RISK",
        )
        st_folium(fmap_risk, height=480, use_container_width=True, key="risk_tab_map", returned_objects=[])

        # Threatened Assets Table
        threatened = coastal.get("threatened_assets", [])
        if threatened:
            st.markdown("##### 🛡️ High-Priority Protected Assets in Threat Corridor")
            for t in threatened:
                t_level = t.get("threat_level", "MONITOR")
                t_color = "#f87171" if t_level == "IMMINENT" else ("#fbbf24" if t_level == "HIGH_RISK" else "#38bdf8")
                st.markdown(f"""
                <div class="glass-panel" style="border-left:3px solid {t_color} !important; padding:12px 16px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong style="color:#f8fafc; font-size:13px;">{t['name']} ({t.get('category', 'Asset').upper()})</strong>
                        <span style="color:{t_color}; font-weight:700; font-size:10px; font-family:'JetBrains Mono';">{t_level} • ESI {t.get('esi', 5)}/10</span>
                    </div>
                    <div style="font-size:11px; font-family:'JetBrains Mono'; color:#94a3b8; margin:4px 0;">
                        DISTANCE: {t.get('distance_from_spill_km', 0):.1f} KM | AUTHORITY: {t.get('contact_authority', 'Port Trust').upper()}
                    </div>
                    <div style="font-size:12px; color:#34d399;">
                        STRATEGY: {t.get('recommended_strategy', 'Deploy containment booms')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Environmental Countermeasure Rules
        crecs = coastal.get("containment_recommendations", [])
        dres = coastal.get("dispersant_restrictions", [])
        if crecs or dres:
            st.markdown("##### 🎯 Environmental Countermeasure Rules")
            for r in crecs:
                st.markdown(f"• 🛡️ {r}")
            for d in dres:
                st.markdown(f"• 🚫 **{d}**")
    else:
        st.info("Execute pipeline to generate coastal vulnerability assessment.")


# =========================================================================
# SECTION 7: REPORTS SCREEN
# =========================================================================
def render_reports_tab(final_state, is_demo):
    st.markdown("#### 📄 Incident Dossier & Regulatory Intelligence Reports")
    st.caption("Export tamper-evident PDF dossiers and structured JSON reports for Coast Guard & Port Authorities.")

    if final_state and final_state.get("report_done"):
        report = final_state.get("incident_report", {})

        # Dossier Summary Box
        st.markdown(f"""
        <div class="glass-panel" style="padding:18px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="telemetry-label">INCIDENT DOSSIER IDENTIFIER</div>
                    <div style="font-size:22px; font-weight:700; font-family:'JetBrains Mono'; color:#f8fafc;">{report.get('incident_id', 'JR-2026-001')}</div>
                </div>
                {render_tag('OFFICIAL')}
            </div>
            <div style="margin-top:10px; font-size:11px; font-family:'JetBrains Mono'; color:#94a3b8;">
                CLASSIFICATION: <strong style="color:#e2e8f0;">{report.get('classification', 'CONFIDENTIAL')}</strong> • GENERATED: {report.get('generated_at', '2026-09-14 15:35 UTC')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download Actions
        col_down1, col_down2 = st.columns(2)
        with col_down1:
            import tempfile
            pdf_path = os.path.join(tempfile.gettempdir(), f"{report.get('incident_id', 'JR-REPORT')}.pdf")
            try:
                generate_pdf_report(report, pdf_path)
                with open(pdf_path, "rb") as f_pdf:
                    pdf_bytes = f_pdf.read()
                st.download_button(
                    "📄 Download Official PDF Dossier",
                    data=pdf_bytes,
                    file_name=f"{report.get('incident_id', 'JR-REPORT')}_Dossier.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"PDF generator notice: {e}")

        with col_down2:
            report_json_str = json.dumps(report, indent=2, default=str)
            st.download_button(
                "📥 Export Machine-Readable JSON",
                data=report_json_str,
                file_name=f"incident_report_{report.get('incident_id', 'jal_rakshak')}.json",
                mime="application/json",
                type="secondary",
                use_container_width=True,
            )

        st.markdown("---")

        # Dossier Sections Explorer
        st.markdown("##### Dossier Content Sections")
        for s in report.get("sections", []):
            with st.expander(f"📑 {s.get('title', 'Section')} ({s.get('data_classification', 'OBSERVED')})", expanded=False):
                st.json(s.get("content", {}))

        limitations = report.get("limitations", [])
        if limitations:
            st.markdown("##### ⚠️ Operational Disclosures & Known Limitations")
            for lim in limitations:
                st.caption(f"• {lim}")
    else:
        st.info("Execute pipeline to generate full forensic incident dossier.")


# =========================================================================
# SENSOR DATA INGESTION PANEL
# =========================================================================
def render_ingestion_panel():
    st.markdown("#### 📁 Sensor Data Ingestion & Target Coordinates")
    st.caption("Mount external radar imagery (GeoTIFF / PNG / JPG) and historical AIS fleet archives (CSV / JSON / GeoJSON).")

    col_ing1, col_ing2 = st.columns(2)
    with col_ing1:
        st.markdown("##### 🛰️ SAR Imagery Feeds")
        uploaded_sar = st.file_uploader(
            "Upload SAR Imagery (GeoTIFF / PNG / JPG)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="panel_sar_uploader",
        )
        if uploaded_sar is not None:
            temp_p = "temp_upload.jpg"
            with open(temp_p, "wb") as f:
                f.write(uploaded_sar.getbuffer())
            st.session_state["active_image_path"] = temp_p
            st.session_state["current_scene_name"] = uploaded_sar.name
            st.success(f"Mounted SAR Imagery: {uploaded_sar.name}")

    with col_ing2:
        st.markdown("##### 🚢 AIS Historical Archive")
        uploaded_ais = st.file_uploader(
            "Upload Fleet Positions (CSV / JSON / GeoJSON)",
            type=["csv", "json", "geojson"],
            key="panel_ais_uploader",
        )
        if uploaded_ais is not None:
            temp_ais = "temp_historical_ais.csv"
            with open(temp_ais, "wb") as f:
                f.write(uploaded_ais.getbuffer())
            st.session_state["active_ais_path"] = temp_ais
            st.success(f"Mounted AIS Archive: {uploaded_ais.name}")

    st.markdown("---")
    st.markdown("##### 📍 Observation Coordinate Anchors")
    c_lat, c_lon = st.columns(2)
    with c_lat:
        in_lat = st.number_input(
            "Target Latitude (°N)",
            value=float(st.session_state.get("spill_lat", DEMO_SPILL_LAT)),
            format="%.4f",
            key="panel_lat_input",
        )
        st.session_state["spill_lat"] = in_lat
    with c_lon:
        in_lon = st.number_input(
            "Target Longitude (°E)",
            value=float(st.session_state.get("spill_lon", DEMO_SPILL_LON)),
            format="%.4f",
            key="panel_lon_input",
        )
        st.session_state["spill_lon"] = in_lon

    act_img = st.session_state.get("active_image_path")
    if act_img and os.path.exists(act_img):
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ EXECUTE MULTI-NODE INTELLIGENCE PIPELINE", type="primary", use_container_width=True, key="panel_exec_btn"):
            st.session_state["trigger_pipeline_run"] = True
            st.rerun()


# =========================================================================
# MINIMAL HIGH-IMPACT LANDING SCREEN
# =========================================================================
def render_landing_screen():
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { max-width: 1200px !important; padding-top: 2.5rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # Hero Container
    st.markdown("""
    <div class="landing-hero-container">
        <div class="landing-badge">
            <span class="status-pulse-sm"></span>
            SENTINEL-1 C-SAR OPERATIONAL // EPSG:4326 // AUTONOMOUS MARITIME C2
        </div>
        <h1 class="landing-title">JAL-RAKSHAK</h1>
        <p class="landing-subtitle">
            Autonomous Maritime Satellite Intelligence & Forensic Oil Spill Attribution Platform. 
            Coupling Sentinel-1 Synthetic Aperture Radar, 3-tier classical consensus verification, 
            hydrodynamic Euler hindcast drift, and spatiotemporal AIS fleet reconstruction.
        </p>
        
        <div class="landing-telemetry-strip">
            <div class="landing-telemetry-item">
                <span class="status-pulse-sm" style="background:#10b981;"></span>
                <span>RADAR: <strong>SENTINEL-1A C-SAR (VV+VH)</strong></span>
            </div>
            <div class="landing-telemetry-item">
                <span class="status-pulse-sm" style="background:#00e5ff;"></span>
                <span>OCEAN: <strong>INCOIS / GFS 0.1° CURRENTS</strong></span>
            </div>
            <div class="landing-telemetry-item">
                <span class="status-pulse-sm" style="background:#38bdf8;"></span>
                <span>AIS RECON: <strong>LIVE SPATIOTEMPORAL STREAM</strong></span>
            </div>
            <div class="landing-telemetry-item">
                <span class="status-pulse-sm" style="background:#8b5cf6;"></span>
                <span>ENGINE: <strong>LANGGRAPH 11-NODE GRAPH</strong></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2 Mode Selection Cards
    col_card1, col_card2 = st.columns(2)

    with col_card1:
        st.markdown("""
        <div class="mode-card mode-card-live">
            <div>
                <span class="mode-card-badge">LIVE SATELLITE COMMAND</span>
                <div class="mode-card-title">
                    <span>🛰️</span> Live Operations Center
                </div>
                <p class="mode-card-desc">
                    Direct tactical multi-sensor interface. Draw arbitrary polygon bounding boxes to query regional ship traffic, inspect live AIS vessel telemetry, calibrate dark-spot thresholds, and command on-demand radar passes.
                </p>
                <ul class="mode-card-features">
                    <li><span class="feat-bullet">⚡</span> Arbitrary Polygon Selection & Regional Spatial Queries</li>
                    <li><span class="feat-bullet">🚢</span> Real-Time Fleet Radar with Marker Clustering</li>
                    <li><span class="feat-bullet">🎯</span> Interactive Target Focus & Candidate Drawer</li>
                    <li><span class="feat-bullet">📁</span> Custom SAR GeoTIFF & AIS Archive Ingestion</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ENTER LIVE OPERATIONS →", key="btn_enter_live", type="primary", use_container_width=True):
            st.session_state["app_mode"] = "live"
            st.query_params["mode"] = "live"
            st.rerun()

    with col_card2:
        st.markdown("""
        <div class="mode-card mode-card-demo">
            <div>
                <span class="mode-card-badge">CURATED EVALUATION WALKTHROUGH</span>
                <div class="mode-card-title">
                    <span>🧪</span> Guided Demo & Evaluation
                </div>
                <p class="mode-card-desc">
                    5-step investigative narrative of the confirmed Chennai Port Outer Anchorage spill. Step-by-step evaluation designed for SIH judges and port authorities demonstrating end-to-end evidence synthesis.
                </p>
                <ul class="mode-card-features">
                    <li><span class="feat-bullet">🔬</span> Step 1: SAR Detection & Consensus Scorecard</li>
                    <li><span class="feat-bullet">⚖️</span> Step 2: AIS Vessel Correlation (MT Ocean Pioneer)</li>
                    <li><span class="feat-bullet">⏱️</span> Step 3: Origin Hindcast & Euler Drift Backtrack</li>
                    <li><span class="feat-bullet">🛡️</span> Step 4: Forward Trajectory & Coastal ESI Threat</li>
                    <li><span class="feat-bullet">📄</span> Step 5: Automated Tamper-Evident PDF Dossier</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("LAUNCH GUIDED DEMO →", key="btn_enter_demo", type="secondary", use_container_width=True):
            st.session_state["app_mode"] = "demo"
            st.query_params["mode"] = "demo"
            st.session_state["active_image_path"] = get_or_create_demo_sar_patch()
            st.session_state["spill_lat"] = CHENNAI_SCENARIO.spill_lat
            st.session_state["spill_lon"] = CHENNAI_SCENARIO.spill_lon
            st.session_state["current_scene_name"] = "Chennai Port Outer Anchorage (512x512)"
            st.session_state["demo_step"] = 1
            st.session_state["auto_run"] = True
            st.rerun()

    # Bottom Operational Presets
    st.markdown("<br><hr style='border-color:rgba(255,255,255,0.06); margin:32px 0 24px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-family:var(--font-mono); font-size:11px; color:#64748b; margin-bottom:12px;'>QUICK CALIBRATED SCENARIO ANCHORS</div>", unsafe_allow_html=True)

    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        if st.button("⚡ Mount Chennai Confirmed Incident (Bay of Bengal)", key="landing_sc_chennai", use_container_width=True):
            st.session_state["app_mode"] = "demo"
            st.query_params["mode"] = "demo"
            st.session_state["active_image_path"] = get_or_create_demo_sar_patch()
            st.session_state["spill_lat"] = CHENNAI_SCENARIO.spill_lat
            st.session_state["spill_lon"] = CHENNAI_SCENARIO.spill_lon
            st.session_state["current_scene_name"] = "Chennai Port Outer Anchorage (512x512)"
            st.session_state["demo_step"] = 1
            st.session_state["auto_run"] = True
            st.rerun()
    with col_sc2:
        if st.button("🛡️ Mount Istanbul Negative Control (Bosphorus Strait)", key="landing_sc_istanbul", use_container_width=True):
            st.session_state["app_mode"] = "live"
            st.query_params["mode"] = "live"
            st.session_state["active_image_path"] = "data/test_sar_scene.jpg"
            st.session_state["spill_lat"] = 41.1100
            st.session_state["spill_lon"] = 29.0500
            st.session_state["current_scene_name"] = "Istanbul Bosphorus Strait (1222x1600)"
            st.session_state["auto_run"] = True
            st.rerun()


# =========================================================================
# LIVE OPERATIONS SCREEN
# =========================================================================
def render_live_operations_screen(final_state, is_demo, spill_lat, spill_lon, active_image):
    live_tabs = st.tabs([
        "🛰️ Tactical Map & Radar",
        "🚢 Fleet Intelligence",
        "🔬 Sensor Analysis",
        "⏱️ Ocean Dynamics",
        "📋 Regulatory Reports",
        "📁 Ingestion & Feeds",
    ])

    with live_tabs[0]:
        render_overview_tab(final_state, is_demo, spill_lat, spill_lon)
    with live_tabs[1]:
        render_ais_tab(final_state, is_demo)
    with live_tabs[2]:
        sub_sar1, sub_sar2 = st.tabs(["🔬 SAR Detection & Consensus", "📡 Radar Filter Calibration"])
        with sub_sar1:
            render_sar_tab(final_state, is_demo, spill_lat, spill_lon, active_image)
        with sub_sar2:
            render_calibration_tab(final_state, is_demo, active_image)
    with live_tabs[3]:
        sub_drift1, sub_drift2 = st.tabs(["⏱️ Drift Trajectory & Hindcast", "🛡️ Shoreline Threat & ESI"])
        with sub_drift1:
            render_drift_tab(final_state, is_demo)
        with sub_drift2:
            render_risk_tab(final_state, is_demo)
    with live_tabs[4]:
        render_reports_tab(final_state, is_demo)
    with live_tabs[5]:
        render_ingestion_panel()


# =========================================================================
# GUIDED DEMO EVALUATION SCREEN (5-STEP SIH JURY STORYTELLING WORKFLOW)
# =========================================================================
def render_demo_screen(final_state, is_demo, spill_lat, spill_lon, active_image):
    # Ensure active demo scenario is loaded if not already present
    if not final_state:
        st.info("ℹ️ Initializing Chennai Incident intelligence pipeline...")
        st.session_state["active_image_path"] = get_or_create_demo_sar_patch()
        st.session_state["spill_lat"] = CHENNAI_SCENARIO.spill_lat
        st.session_state["spill_lon"] = CHENNAI_SCENARIO.spill_lon
        st.session_state["current_scene_name"] = "Chennai Port Outer Anchorage (512x512)"
        st.session_state["trigger_pipeline_run"] = True
        st.rerun()

    if "demo_step" not in st.session_state:
        st.session_state["demo_step"] = 1
    current_step = st.session_state["demo_step"]

    steps = [
        {"idx": 1, "title": "SAR Detection & Consensus", "tag": "STEP 01"},
        {"idx": 2, "title": "AIS Vessel Attribution", "tag": "STEP 02"},
        {"idx": 3, "title": "Origin Hindcast Backtrack", "tag": "STEP 03"},
        {"idx": 4, "title": "Forward Drift & Coastal Risk", "tag": "STEP 04"},
        {"idx": 5, "title": "Incident Dossier & PDF", "tag": "STEP 05"},
    ]

    st.markdown("### 🧪 Guided Forensic Investigation Walkthrough")
    st.caption("Evaluation sequence designed for SIH jury and maritime operators — 5 interconnected stages of automated forensic intelligence.")

    # Stepper selector pills
    step_cols = st.columns(len(steps))
    for i, s in enumerate(steps):
        with step_cols[i]:
            is_active = (s["idx"] == current_step)
            is_completed = (s["idx"] < current_step)
            btn_type = "primary" if is_active else "secondary"
            check_icon = "✓ " if is_completed else ("● " if is_active else f"{s['idx']} ")
            if st.button(f"{check_icon}{s['title']}", key=f"demo_step_btn_{s['idx']}", type=btn_type, use_container_width=True):
                st.session_state["demo_step"] = s["idx"]
                st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:16px 0 20px 0;'>", unsafe_allow_html=True)

    if current_step == 1:
        st.markdown("""
        <div class="demo-narrative-box">
            <strong style="color:#00e5ff;">PHASE 1 // SAR DETECTION & MULTI-TIER CONSENSUS VERIFICATION:</strong>
            Sentinel-1A C-band SAR pass acquired over Chennai Port Outer Anchorage. 
            Deep learning (YOLOv8) proposes dark candidate regions. To eliminate lookalikes (calm waters, biogenic slicks, land shadows), 
            the 3-tier Classical Consensus Engine validates radar backscatter damping, Otsu/K-means segmentation, and marine domain masking.
        </div>
        """, unsafe_allow_html=True)
        render_sar_tab(final_state, is_demo=True, spill_lat=spill_lat, spill_lon=spill_lon, active_image=active_image)

    elif current_step == 2:
        st.markdown("""
        <div class="demo-narrative-box">
            <strong style="color:#00e5ff;">PHASE 2 // AIS FLEET CORRELATION & CANDIDATE ATTRIBUTION:</strong>
            Fuses historical AIS broadcast archives with the observed spill footprint. Calculates spatiotemporal proximity, 
            heading consistency, speed anomalies, and drift alignment to produce transparent, mathematical culpability scores.
            Notice how <strong>MT Ocean Pioneer (MMSI: 413289000)</strong> scores 87.4% due to coincident presence and loitering.
        </div>
        """, unsafe_allow_html=True)
        render_ais_tab(final_state, is_demo=True)

    elif current_step == 3:
        st.markdown("""
        <div class="demo-narrative-box">
            <strong style="color:#00e5ff;">PHASE 3 // HYDRODYNAMIC ADVECTION HINDCAST (BACKTRACKING):</strong>
            Because oil drifts downwind and with ocean currents between release and satellite overpass, we reverse-integrate 
            the Euler advection equations using regional INCOIS / GFS currents (0.48 m/s @ 118°) and 3% windage. 
            This backtracks the slick 180 minutes to its origin point (13.1380°N, 80.3710°E), which directly intersects the suspect vessel's track.
        </div>
        """, unsafe_allow_html=True)
        render_drift_tab(final_state, is_demo=True)

    elif current_step == 4:
        st.markdown("""
        <div class="demo-narrative-box">
            <strong style="color:#00e5ff;">PHASE 4 // FORWARD TRAJECTORY FORECAST & SHORELINE IMPACT (ESI):</strong>
            Projects the 48-hour forward drift envelope to compute time-to-beach and evaluate Environmental Sensitivity Index (ESI) 
            risk. Pinpoints high-priority vulnerable zones (Chennai Port harbor basin, Marina Beach turtle nesting shores, Ennore Creek mangroves) 
            for targeted boom containment.
        </div>
        """, unsafe_allow_html=True)
        render_risk_tab(final_state, is_demo=True)

    elif current_step == 5:
        st.markdown("""
        <div class="demo-narrative-box">
            <strong style="color:#00e5ff;">PHASE 5 // TAMPER-EVIDENT REGULATORY DOSSIER & PDF DISPATCH:</strong>
            Synthesizes all multi-spectral sensor telemetry, mathematical consensus proof, AIS vessel telemetry, and forecast models 
            into an official, cryptographically hashed incident dossier for the Indian Coast Guard and Directorate General of Shipping.
        </div>
        """, unsafe_allow_html=True)
        render_reports_tab(final_state, is_demo=True)

    # Bottom Step Stepper Navigation
    st.markdown("<br><hr style='border-color:rgba(255,255,255,0.06); margin:24px 0 16px 0;'>", unsafe_allow_html=True)
    col_nav1, col_nav2, col_nav3 = st.columns([3, 4, 3])
    with col_nav1:
        if current_step > 1:
            if st.button("◀ PREVIOUS STEP", key="demo_btn_prev", use_container_width=True):
                st.session_state["demo_step"] = current_step - 1
                st.rerun()
    with col_nav2:
        st.markdown(f"<div style='text-align:center; font-family:var(--font-mono); font-size:12px; color:#94a3b8; padding-top:8px;'>STAGE {current_step} OF {len(steps)}: {steps[current_step-1]['title'].upper()}</div>", unsafe_allow_html=True)
    with col_nav3:
        if current_step < len(steps):
            if st.button("NEXT STEP ▶", key="demo_btn_next", type="primary", use_container_width=True):
                st.session_state["demo_step"] = current_step + 1
                st.rerun()
        else:
            if st.button("🛰️ OPEN IN LIVE OPERATIONS", key="demo_btn_finish", type="primary", use_container_width=True):
                st.session_state["app_mode"] = "live"
                st.query_params["mode"] = "live"
                st.rerun()


# ──────────────────────────────────────────────────────────────
# MAIN APPLICATION ROUTING CONTROLLER
# ──────────────────────────────────────────────────────────────
final_state = st.session_state.get("pipeline_result")
active_image = st.session_state.get("active_image_path")

if app_mode == "landing":
    render_landing_screen()
elif app_mode == "demo":
    render_demo_screen(final_state, is_demo=True, spill_lat=spill_lat, spill_lon=spill_lon, active_image=active_image)
else:
    render_live_operations_screen(final_state, is_demo=is_demo, spill_lat=spill_lat, spill_lon=spill_lon, active_image=active_image)