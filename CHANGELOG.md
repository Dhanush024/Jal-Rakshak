# Jal-Rakshak — Architecture & UI Redesign Changelog

## [2.1.0] - 2026-09-28 — Maritime Operations Center Rebuild

### 1. Map-Centric Command Center Architecture
- **Geospatial Hero Surface (`command_surface.py`):**
  - Integrated high-performance MapLibre GL WebGL interactive geospatial component into `app.py`.
  - Prominently positioned as the primary hero surface across **Analysis**, **AIS**, **Drift**, and **Risk** views (>= 60% viewport height), eliminating the previous tab-buried layout.
  - Interactive vector layers:
    - Detected Oil Slick Polygon with pulsing perimeter boundary.
    - Probable Origin Bayesian Credible Zones ($P=0.50$, $P=0.75$, $P=0.90$) and centroid marker.
    - Backward Euler Hindcast Trajectory vector.
    - Forward Hydrodynamic Drift Forecast trajectory ($T+6\text{h}$).
    - AIS Fleet historical track LineStrings.
    - Dynamic Vessel candidate markers.
    - Sensitive Shoreline Assets with Environmental Sensitivity Index (ESI) chips.
- **Client-Side Playback Timeline Scrubber:**
  - Interactive time dock ($T-180\text{min}$ to $T+60\text{min}$) with Play/Pause button, 1x/2x/5x speed selector, and epoch timestamp readout.
  - Smooth client-side interpolation of vessel positions along their AIS tracks at 60fps in browser memory with **zero Streamlit full-app reruns**.
- **Synchronized Context Inspector Panel:**
  - Right context panel automatically slides in and updates based on geospatial selection:
    - **Vessel Candidate:** Association score ring gauge, spatial/temporal/trajectory breakdown, MMSI, and mandatory legal notice.
    - **Spill Slick:** Area (km²), perimeter, estimated empirical volume, Fay spreading age, and YOLO/classical consensus agreement.
    - **Drift Origin:** Hindcast parameters, ocean surface current vector, and wind leeway transfer values.
    - **Coastal Asset:** Shoreline vulnerability index, ESI rank, and protection priority.
- **Keyboard Shortcuts & Quick Command Palette:**
  - Added Command Palette (`Cmd/Ctrl+K` or pressing `K`) for instant fly-to and layer toggles.
  - Shortcut keys: `1`-`7` for switching views, `Space` for timeline play/pause, `F` for fitting map bounds, and `Esc` for clearing selections.

---

### 2. Elimination of Carto API-Key Dependency
- **Root Cause:**
  - The legacy configuration and Folium layer included `cartocdn.com` URLs requiring a Carto API key, producing broken tiles and public API-key warnings.
- **Resolution:**
  - Replaced all basemaps with 100% token-free, legitimate **OpenStreetMap** raster tiles (`https://tile.openstreetmap.org/{z}/{x}/{y}.png`).
  - Styled tiles via hardware-accelerated dark tactical CSS/paint filter (`brightness(0.65) invert(1) contrast(3.2) hue-rotate(200deg) saturate(0.28)`).
  - Preserved standard OpenStreetMap attribution with zero "API KEY REQUIRED" text anywhere across local and deployed environments.

---

### 3. Telemetry Mode & State Bug Fixes
- **Live vs Demo Mode Fix:**
  - Resolved the bug in `app.py` where line 1201 previously had `is_demo = True` hardcoded, overwriting user selections and forcing simulated mode even when Live Telemetry was selected.
  - The application now persists `st.session_state["app_mode"]` accurately, passing `app_mode="live"` to `run_pipeline(...)` when live telemetry is engaged and dynamically updating header data classification badges (`OBSERVED` vs `SIMULATED`).
- **Elimination of Streamlit Map Rerun Thrashing:**
  - Previous `st_folium` calls triggered full-app reruns on every pan or zoom interaction.
  - The new client-side MapLibre GL command surface handles panning, zooming, layer toggles, and scrubbing entirely on the client, maintaining full application state and performance.
- **Graceful Negative Control Empty-State Handling:**
  - Verified that negative control scenes (e.g., Istanbul Bosphorus terrestrial false positive) render without crashing on missing keys (`candidate_scores: None`, `forecast_results: None`).
  - Clear rationale displayed: *"90.9% terrestrial overlap violates maritime domain constraint"*.

---

### 4. Design System & Typography Modernization
- **Centralized Design System (`assets/theme.css`):**
  - Moved legacy inline CSS blocks to a structured, modular design token file.
  - Established strict typography scale (11px, 12px, 13px, 15px, 20px, 28px) utilizing **Inter** for UI elements and **JetBrains Mono** for coordinates, MMSI, and telemetry metrics.
  - Added tabular numerals (`font-variant-numeric: tabular-nums`) across all numerical metrics.
  - Replaced emojis with clean SVG icons and standardized classification badges (`OBSERVED`, `INFERRED`, `PREDICTED`, `SIMULATED`, `OFFICIAL`).
  - Ensured WCAG AA contrast compliance and implemented `@media (prefers-reduced-motion: reduce)`.

---

### 5. Backend Protection & Verification
- **Read-Only Backend Integrity:**
  - Verified that `pipeline/`, `sar/`, `geospatial/`, `demo/`, `reporting/`, and `config/` remain completely untouched.
  - `git diff --stat` confirms only presentation layer files (`app.py`, `assets/theme.css`, `command_surface.py`) were modified or added.
- **Automated Test Suite:**
  - All **102/102 tests** passing successfully (`pytest tests/ -v`).
