# Jal-Rakshak — Progress Tracker

## Phase Status

| Phase | Feature | Status | Tests | Known Limitations |
|-------|---------|--------|-------|-------------------|
| 0 | Repository Audit | ✅ COMPLETE | — | — |
| 1 | Security & Foundation | ✅ COMPLETE | 2/2 | — |
| 2 | Architecture Refactor | ✅ COMPLETE | 2/2 | — |
| 3 | Data Provider Abstraction | ✅ COMPLETE | — | Live AIS provider not implemented |
| 4-5 | SAR Preprocessing | ✅ COMPLETE | Imports verified | No real GeoTIFF ingestion yet |
| 6 | YOLO Segmentation | ✅ COMPLETE | — | Requires best.pt model file |
| 7 | Classical SAR Validation | ✅ COMPLETE | Imports verified | — |
| 8 | Look-alike Detection | ✅ COMPLETE | — | Part of classical.py validation |
| 9 | Segmentation Evaluation | ✅ COMPLETE | Imports verified | — |
| 10 | Spill Characterization | ✅ COMPLETE | Imports verified | Volume is empirical estimate |
| 13-14 | AIS Data & Reconstruction | ✅ COMPLETE | — | Demo provider only |
| 15 | AIS Traffic Filtering | ✅ COMPLETE | 2/2 | — |
| 16-17 | Vessel Feature Engineering & Attribution | ✅ COMPLETE | — | — |
| 19 | Ocean-Current Hindcast | ✅ COMPLETE | 5/5 | Constant field assumption |
| 22 | Forward Drift Prediction | ✅ COMPLETE | — | — |
| 25 | Risk Engine | ✅ COMPLETE | 4/4 | — |
| 26-28 | Community Alerts & Notifications | ✅ COMPLETE | 2/2 | Simulation mode only |
| 31 | Incident Report | ✅ COMPLETE | 2/2 | — |
| 32 | LangGraph Pipeline (11-node) | ✅ COMPLETE | 2/2 | — |
| 33 | Demo Mode | ✅ COMPLETE | 3/3 | — |
| 11 | Spill Age Estimation | ⏳ PENDING | — | — |
| 12 | Multi-temporal SAR | ⏳ PENDING | — | — |
| 18 | Oceanographic Data (live) | ⏳ PENDING | — | — |
| 20 | Source Probability Map | ⏳ PENDING | — | — |
| 21 | AIS + Ocean Fusion | ⏳ PENDING | — | — |
| 23 | Coastal Impact | ✅ COMPLETE | 6/6 | Shoreline proximity, landfall ETA, ESI index, countermeasures |
| 24 | Incident Timeline | ✅ COMPLETE | Interactive Slider | Forensic timeline scrubbing (-180m to +60m) |
| 29 | Authority Dashboard | ✅ COMPLETE | Interactive UI | Multi-card layout, telemetry panels |
| 30 | Map System | ✅ COMPLETE | Folium Map | Vessel tracks, origin zone, forecast, sensitive zones |
| 34 | Testing | ✅ COMPLETE | 36/36 | Geospatial, ocean, AIS, coastal, and pipeline integration tests pass |
| 35 | Performance | ⏳ PENDING | — | — |
| 36 | Observability | ⏳ PENDING | — | — |
| 37 | UI Polish | ⏳ PENDING | — | — |
| 38 | Documentation | ⏳ PENDING | — | — |
| 39 | Scientific Credibility | ⏳ PENDING | — | — |
| 40 | Final Demonstration | ⏳ PENDING | — | — |

## Change Log

### Phase 23-24 — Coastal Impact Assessment & Incident Timeline (COMPLETE)
- Created `coastal/` package with `zones.py` and `impact.py`:
  - NOAA/IMO Environmental Sensitivity Index (ESI 1-10) scale
  - Chennai / Coromandel coastline geometry and sensitive ecological/infrastructure assets (bird sanctuary, turtle nesting, mangroves, desalination plant, power station cooling intake)
  - Shortest distance to coast calculation (`haversine_km`)
  - Forward drift trajectory intersection and landfall projection with ETA uncertainty interval
  - Threatened asset identification & composite coastal vulnerability scoring (0-100)
  - Protective countermeasure strategies (deflection booming, exclusion barriers, dispersant bans in shallow waters)
- Integrated coastal assessment into `pipeline/state.py`, `pipeline/graph.py` (node_forecast, node_risk, node_report), and `reporting/incident.py`.
- Added interactive Coastal Impact & Shoreline Threat Assessment panel in `app.py`.
- Added 6 dedicated unit & pipeline integration tests in `tests/test_coastal.py` (36/36 passing).

### Phase 29-30 — Authority Dashboard & Map System (COMPLETE)
- Rewrote `app.py` integrating the complete 11-node LangGraph pipeline.
- Implemented interactive Folium investigation map:
  - Estimated spill origin zone with uncertainty circle (±km)
  - Euler hindcast backward drift arrow
  - Forward drift forecast trajectories (cyan/violet/rose cones)
  - Candidate AIS vessel tracks (color-coded polylines)
  - Real-time vessel interpolation based on timeline scrubber slider
  - Sensitive coastal areas (wildlife sanctuary, fishing harbor, port)
  - Distance connector line from selected candidate vessel to origin
- Added interactive Timeline Scrubber (-180 min to +60 min) with forensic simulation time readout.
- Added candidate vessel selector with breakdown cards and evidence summaries.
- Added 1-click Quick Launch for Chennai Incident Demonstration Scenario (`demo_sar_patch.png`).
- Added end-to-end pipeline execution unit tests in `tests/test_core.py` (30/30 passing).

### Phase 0 — Repository Audit (COMPLETE)
- Inspected all files: app.py, pipeline.py, ais_engine.py, requirements.txt, devcontainer.json
- Found CRITICAL: hard-coded API key in pipeline.py:193
- Found: no .gitignore, __pycache__ tracked, temp files committed
- Found: YOLO model reloaded per inference
- Found: simulated data not labeled as demo
- Created IMPLEMENTATION_PLAN.md, PROGRESS.md, DECISIONS.md

### Phase 1 — Security & Foundation (COMPLETE)
- Removed hard-coded AISStream API key from pipeline.py
- Created `.env.example` for environment variable configuration
- Created `.gitignore` to protect secrets, caches, temp files
- Created `config/settings.py` — centralized configuration from env vars
- Updated `requirements.txt` with categorized dependencies

### Phase 2 — Architecture Refactor (COMPLETE)
- Created modular package structure:
  - `sar/` — preprocessing, detection, classical, metrics, geometry
  - `ais/` — provider, filtering, attribution
  - `ocean/` — provider, hindcast
  - `geospatial/` — distance utilities
  - `risk/` — multi-factor risk engine
  - `alerts/` — alert manager with simulation mode
  - `reporting/` — incident report generator
  - `demo/` — demo scenario builder
  - `config/` — centralized settings
  - `pipeline/` — LangGraph state & 11-node graph
  - `tests/` — 28 unit tests (all passing)
- Total: 18 new Python modules, 1 test module
- All modules import-tested independently
- Security: API key now read from env var, not source code
- All datetime operations use timezone-aware objects

### Key Design Decisions Applied
- Every data source labeled: OBSERVED / INFERRED / PREDICTED / SIMULATED
- Vessel attribution uses "candidate vessel" and "association score" terminology
- All alerts run in SIMULATION mode by default
- Volume estimates carry empirical uncertainty disclaimers
- Pipeline short-circuits (no-spill → report) via conditional edge
