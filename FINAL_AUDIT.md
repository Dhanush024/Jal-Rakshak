# Jal-Rakshak — Final Pre-Hardening System Audit (SIH 2026)

**Problem Statement SIH26143:** *Leveraging satellite imagery to determine Oil spills at sea along with AIS data correlations to identify vessel responsible for the spill.*

---

## 1. Executive Summary & Audit Methodology

This audit inspects the Jal-Rakshak codebase across all architectural layers, comparing claimed functionality in `PROGRESS.md` against true runtime behaviors, data flows, and external interfaces.

Every component is classified into one of five rigorous reality categories:
- **REAL:** Production-grade implementation executing on real physical inputs or real sensor data with no synthetic placeholders.
- **REAL ALGORITHM / SIMULATED DATA:** Mathematically rigorous, verified algorithm operating on synthetic, simulated, or default environmental parameters.
- **PARTIAL:** Working functional logic that is incomplete (e.g., handles local images but lacks live satellite API connectors).
- **SIMULATED:** Functionality that operates explicitly on deterministic mock scenarios, synthetic AIS tracks, or mock alerts by design.
- **PLACEHOLDER:** Abstract interface, stubbed function, or unconfigured network gateway.

---

## 2. Phase-by-Phase Reality Classification

| Phase | Component | Reality Classification | Findings & Code Inspection Details |
|---|---|---|---|
| **0** | Repository Audit | **REAL** | Initial audit complete; mapped architecture, debt, and risks. |
| **1** | Security & Foundation | **REAL** | Hardcoded keys removed; `.env.example`, `.gitignore`, `config/settings.py` centralized settings. Zero tracked secrets. |
| **2** | Architecture Refactor | **REAL** | Clean separation of concerns across `sar/`, `ais/`, `ocean/`, `coastal/`, `risk/`, `alerts/`, `reporting/`, `pipeline/`. |
| **3** | Data Provider Abstraction | **PLACEHOLDER** | `AISProvider`, `OceanProvider`, `WeatherProvider` in `ais/provider.py` and `ocean/provider.py` are abstract base classes, but **no concrete classes (`DemoAISProvider`, `HistoricalAISProvider`, etc.) inherit from them**. The pipeline calls module-level functions directly. |
| **4** | Satellite / SAR Ingestion | **PARTIAL** | Supports uploaded JPG/PNG files and synthetic demo patch generation. **No GeoTIFF ingestion, no CRS coordinate transform, no live STAC/Copernicus API client.** |
| **5** | SAR Preprocessing | **REAL** | Lee speckle filtering, CLAHE, median filtering, Wiener approximation, land/sea masking, 512×512 tiling implemented in OpenCV/NumPy (`sar/preprocessing.py`). |
| **6** | YOLO Segmentation | **REAL** | Real YOLOv8 instance segmentation (`best.pt`, 6.7 MB) with multi-detection polygon extraction and bounding boxes (`sar/detection.py`). |
| **7** | Classical SAR Validation | **REAL** | Adaptive Gaussian thresholding, Otsu, K-means 3-cluster radiometric segmentation, dark-spot extraction, and mask agreement analysis (`sar/classical.py`). |
| **8** | Look-alike Detection | **PARTIAL** | Rule-based heuristics in `sar/classical.py` evaluate low-wind thresholds, area ratios, and mask disagreement. No live oceanographic wave radar or biogenic film sensors. |
| **9** | Segmentation Evaluation | **REAL (Algorithm)** | Mathematical implementations of IoU/Jaccard, Sørensen-Dice, Precision, Recall, F1, and boundary distance F-score (`sar/metrics.py`). No benchmark dataset stored in repo; metrics compare prediction vs classical reference. |
| **10** | Spill Characterization | **REAL** | Real OpenCV geometry calculations: Centroid, perimeter, area, fitted ellipse axes, aspect ratio, orientation, convexity, solidity (`sar/geometry.py`). Volume is explicitly an empirical estimate. |
| **11** | Spill Age Estimation | **PARTIAL / INFERRED** | `SpillAgeEstimator` in `sar/weathering.py` applies Fay spreading regimes and morphology heuristics. Falls back to `AGE: UNKNOWN` when area $< 50$ px, but output is an empirical inference. |
| **12** | Multi-Temporal SAR | **REAL ALGORITHM / SIMULATED DATA** | `MultiTemporalSARTracker` calculates $dA/dt$, net drift vector, velocity in knots, and fragmentation progression. Verified via unit tests, but no live multi-pass satellite feed exists. |
| **13** | AIS Live Data | **PLACEHOLDER** | In `pipeline/graph.py` line 301, live mode literally has `# TODO: implement live AIS provider`, returning `raw_tracks = {}` and setting data mode to `UNAVAILABLE`. |
| **14** | Historical AIS Reconstruction | **SIMULATED** | Trajectories in demo mode come from `generate_demo_ais_tracks()` in `demo/scenario.py` (4 synthetic vessels: MT ARCTIC STAR, MV PACIFIC TRADER, OCEAN VOYAGER, SEA EXPLORER). No real historical AIS database/API is connected. |
| **15** | AIS Traffic Filtering | **REAL** | Real 4-stage progressive filtering in `ais/filtering.py` (spatial bounding box, temporal window, source envelope intersection, and closest point of approach). |
| **16** | Vessel Feature Engineering | **REAL** | Real feature extraction in `ais/attribution.py`: Closest approach distance ($d_{\text{CPA}}$), time delta ($\Delta t_{\text{CPA}}$), speed profiles, heading consistency, and AIS gap detection. |
| **17** | Candidate Attribution Scoring | **REAL** | Real explainable 0–100 scoring in `ais/attribution.py` with weighted factor breakdown cards, evidence summaries, and non-adjudicative disclaimers. |
| **18** | Oceanographic Data (Live) | **PLACEHOLDER / SIMULATED** | Abstract base classes in `ocean/provider.py` have no live client (INCOIS, NOAA, Copernicus). System defaults to constant parameters (`DEFAULT_CURRENT_SPEED_MS = 0.48`, `DEFAULT_WIND_SPEED_MS = 6.2`). |
| **19** | Ocean-Current Hindcast | **REAL ALGORITHM / SIMULATED DATA** | Real backward Euler numerical integration and Monte Carlo particle ensemble in `ocean/hindcast.py` operating on default constant current/wind vectors. |
| **20** | Source Probability Map | **REAL ALGORITHM / SIMULATED DATA** | Real 2D Gaussian likelihood surface and P50/P75/P95 Rayleigh quantile polygon generators in `ocean/probability.py`. |
| **21** | AIS + Ocean Fusion | **REAL** | Real multi-criteria evidence fusion in `ais/attribution.py` combining spatial proximity, temporal alignment, and drift consistency. |
| **22** | Forward Drift Prediction | **REAL ALGORITHM / SIMULATED DATA** | Real forward Euler drift integration for +6h, +12h, +24h with leeway wind transport (3%) in `ocean/hindcast.py`. |
| **23** | Coastal Impact Assessment | **REAL** | Real geographical coordinates for Chennai shoreline and 5 sensitive assets (Pulicat Lake ESI 10, Marina Beach ESI 8, Nemmeli Desalination ESI 2) in `coastal/zones.py`. Real ray-casting, landfall ETA, ESI vulnerability score, and countermeasure rules in `coastal/impact.py`. |
| **24** | Incident Timeline / Replay | **REAL (UI) on SIMULATED (Tracks)** | Interactive Streamlit slider (-180m to +60m) dynamically interpolating vessel coordinates and updating Folium markers and distance connectors in real time. |
| **25** | Risk Engine | **REAL** | Real multi-factor risk categorization (LOW, MEDIUM, HIGH, CRITICAL) based on spill area, coastal landfall ETA, ESI vulnerability, and detection confidence in `risk/engine.py`. |
| **26-27** | Community Early Warning & Alerts | **REAL (Logic & Content)** | Structured, plain-language action bulletins for fishermen and residents with avoidance zones in `alerts/manager.py`. |
| **28** | Multi-Channel Notifications | **SIMULATED / PLACEHOLDER** | Operates in simulation mode (`is_simulated = True`). Dashboard delivery works. Real Email and SMS delivery print logger warnings (`"Email delivery not configured"`, `"SMS delivery not configured"`). |
| **29** | Authority Dashboard | **REAL** | Multi-panel Streamlit command center in `app.py` with telemetry metrics, intelligence cards, risk banners, candidate rankings, and report viewers. |
| **30** | Map System | **REAL** | Interactive Folium map with dark tiles, spill marker, P50/P75/P95 Bayesian source likelihood polygons, forecast drift lines, sensitive coastal assets, vessel tracks, and dynamic scrubber markers. |
| **31** | Incident Report & PDF Export | **REAL** | JSON report generation in `reporting/incident.py` and publication-quality two-pass multi-page PDF generation via ReportLab in `reporting/pdf.py`. |
| **32** | LangGraph Pipeline | **REAL** | 11-node stateful workflow graph in `pipeline/graph.py` with conditional routing on spill detection. |
| **33** | Demo Mode | **REAL (System) running SIMULATED (Scenario)** | Self-contained, deterministic Chennai Coast scenario with synthetic SAR patch, 4 synthetic AIS tracks, and synthetic ocean parameters. |
| **34** | Automated Testing | **REAL** | 44 automated tests in `tests/test_core.py`, `tests/test_coastal.py`, and `tests/test_advanced.py` covering all mathematical and pipeline modules. |
| **35** | Performance | **REAL** | Lazy model loading (`self._model is None` in `YOLODetector`), Streamlit session state caching. |
| **36** | Observability | **PARTIAL** | Standard Python logging (`logging.getLogger("jal_rakshak.*")`). Structured JSON logging/telemetry tracing not implemented. |
| **37** | UI Polish | **REAL** | Modern dark-theme CSS, responsive grid layouts, color-coded badges, SVG icons. |
| **38** | Documentation | **REAL** | Complete 608-line `README.md` covering all 24 required sections. |
| **39** | Scientific Credibility | **REAL** | Audited terminology ("candidate vessel", "association score", "probable source region", "estimated volume (empirical)", data classification badges). |
| **40** | Final Demonstration Workflow | **REAL (Workflow) on SIMULATED (Scenario)** | 1-click execution through all 11 nodes. |

---

## 3. High-Priority Engineering Gaps Status (RESOLVED)

1. **Concrete Provider Architecture:** ✅ **RESOLVED**
   - Implemented `DemoAISProvider` (offline synthetic Chennai fleet) and `FileHistoricalAISProvider` (ingests standard NMEA CSV/JSON/GeoJSON with deduplication and temporal sorting).
   - Implemented `DemoOceanProvider`, `ConstantOceanProvider` (with configurable vectors), and `DemoSatelliteProvider`, `FileSatelliteProvider` in `sar/ingestion.py`.
   - Implemented `SimulationNotificationProvider`, `ConsoleNotificationProvider`, `DashboardNotificationProvider` in `alerts/manager.py`.
   - Fully decoupled `pipeline/graph.py` to use provider factories with `DATA_MODE = DEMO | REAL | AUTO`.

2. **Real SAR Input & Georeferencing:** ✅ **RESOLVED**
   - Implemented `SARSceneMetadata` and `SARSceneLoader` in `sar/ingestion.py` supporting GeoTIFF/PNG/JPG rasters, CRS coordinate mapping, spatial resolution, and sidecar metadata.

3. **Map / Basemap Provider Watermark:** ✅ **RESOLVED**
   - Switched default Folium basemap to `OpenStreetMap`, removing broken `carto.com/basemaps/apikey` watermark for all offline demos.
   - Retained optional `CARTO_API_KEY` configuration for Dark Matter tiles.

4. **SIH 2026 Presentation Assets:** ✅ **RESOLVED**
   - `SIH_DEMO_SCRIPT.md`: Precise 5-minute timed presentation script.
   - `JURY_QA.md`: Scientifically defensible answers to the 18 most critical jury questions.

---

## 4. YOLO Segmentation Accuracy & Rendering Resolution

- **Root Cause:**
  - `best.pt` (YOLOv8n-seg, 6.45 MB, 1 class: `oill`) was trained on marine oil slicks without negative coastal land samples. High-contrast coastal scenes (e.g., Istanbul Bosphorus scene) triggered false positives over the landmass (84% land overlap).
  - Contour extraction from polygon point arrays produced 1-pixel bridge artifacts across waterways.
- **Resolution:**
  1. Extracted masks directly from native `result.masks.data`, resized to original raster dimensions.
  2. Implemented OpenCV external contour extraction (`cv2.RETR_EXTERNAL`), eliminating bridge artifacts.
  3. Implemented adaptive Otsu landmasking in `sar/preprocessing.py` and marine boundary constraints in `sar/detection.py` (`max_land_overlap = 0.40`, `max_scene_coverage = 0.35`).
  4. Non-marine false alarms are cleanly rejected (`is_valid_marine = False`) with explicit forensic logging, short-circuiting the pipeline to report without drawing spurious polygons over land.
- **Ground-Truth Accuracy Disclaimer:**
  *Model accuracy on independent ground truth cannot be established from the current repository because no independent ground-truth validation dataset is configured.*
- **Regression Tests:** 10 new regression tests added in `tests/test_detection_regression.py` (total test suite: 70/70 passing).

