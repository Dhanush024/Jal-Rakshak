# Jal-Rakshak

**AI-Powered Maritime Oil Spill Intelligence and Early Warning System**

---

> [!IMPORTANT]
> **Scientific Credibility & Legal Disclaimer**  
> *Jal-Rakshak* is an operational decision-support platform designed to assist maritime authorities, environmental response agencies, and coastal communities.  
> All vessel attribution scores, drift trajectories, and landfall estimates represent **statistical associations and numerical model inferences**.  
> The system identifies **candidate vessels** based on spatial, temporal, and hydrodynamic consistency; it does **not** independently establish legal culpability or replace forensic admiralty investigations.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Motivation & Context](#2-motivation--context)
3. [The Jal-Rakshak Solution](#3-the-jal-rakshak-solution)
4. [Key Features](#4-key-features)
5. [System Architecture](#5-system-architecture)
6. [SAR Ingestion & Preprocessing Pipeline](#6-sar-ingestion--preprocessing-pipeline)
7. [YOLOv8 Detection & Classical Validation](#7-yolov8-detection--classical-validation)
8. [Spill Morphology & Age Estimation](#8-spill-morphology--age-estimation)
9. [Oceanographic Hindcast & Probable Source Region](#9-oceanographic-hindcast--probable-source-region)
10. [AIS Trajectory Reconstruction & Progressive Filtering](#10-ais-trajectory-reconstruction--progressive-filtering)
11. [Vessel Attribution & Multi-Factor Scoring](#11-vessel-attribution--multi-factor-scoring)
12. [Forward Drift Prediction & Hydrodynamics](#12-forward-drift-prediction--hydrodynamics)
13. [Coastal & Environmental Impact Assessment](#13-coastal--environmental-impact-assessment)
14. [Explainable Multi-Factor Risk Engine](#14-explainable-multi-factor-risk-engine)
15. [Community Early Warning & Geofenced Alerts](#15-community-early-warning--geofenced-alerts)
16. [LangGraph 11-Node Orchestration Engine](#16-langgraph-11-node-orchestration-engine)
17. [Investigation Reporting & PDF Dossier](#17-investigation-reporting--pdf-dossier)
18. [Reproducible Chennai Coast Demo Scenario](#18-reproducible-chennai-coast-demo-scenario)
19. [Installation & Prerequisites](#19-installation--prerequisites)
20. [Configuration & Environment Variables](#20-configuration--environment-variables)
21. [Running Locally](#21-running-locally)
22. [Automated Testing & Verification](#22-automated-testing--verification)
23. [Methodological Limitations & Uncertainty Accounting](#23-methodological-limitations--uncertainty-accounting)
24. [Data Sources & Future Work](#24-data-sources--future-work)

---

## 1. Problem Statement

Maritime oil spills—whether accidental (vessel collisions, groundings, pipeline ruptures) or deliberate (illicit bilge dumping, tank washing under cover of darkness)—inflict catastrophic damage on marine ecosystems, coastal fisheries, desalination intakes, and port infrastructure. 

Existing response workflows face critical bottlenecks:
- **Detection latency:** Satellite imagery is reviewed in silos, often hours or days after the discharge.
- **Attribution opacity:** Identifying the discharging vessel from hundreds of transit ships across busy maritime corridors is manual and error-prone.
- **Drift uncertainty:** Backward tracking to the origin point and forward forecasting toward the coast rarely account for windage uncertainty, hydrodynamic current shear, and spreading dynamics simultaneously.
- **Community disconnect:** Coastal fishing communities and aquaculture facilities receive warnings too late to deploy containment booms or divert fleets.

---

## 2. Motivation & Context

Under the **Smart India Hackathon (SIH)** framework, *Jal-Rakshak* was conceived to deliver an end-to-end, technically rigorous system that closes the loop between **spaceborne SAR observation**, **oceanographic numerical modeling**, **AIS transponder telemetry**, **coastal protection**, and **community alerting**.

Rather than stopping at a bounding box or pixel mask, Jal-Rakshak answers five essential operational questions:
1. *Where is the spill, what is its extent, and is it a true spill or a look-alike?*
2. *How old is the slick based on weathering and Fay spreading dynamics?*
3. *Where did the slick originate in the past, and which candidate vessels transited that source window?*
4. *Where will the slick drift over the next 6, 12, and 24 hours, and when will it impact the shoreline?*
5. *Which sensitive ecological habitats, ports, and fishing villages require immediate warning and booming countermeasures?*

---

## 3. The Jal-Rakshak Solution

```
DETECT ──► VALIDATE ──► CHARACTERIZE ──► AGE ──► HINDCAST ──► CORRELATE AIS ──► ATTRIBUTE ──► FORECAST ──► IMPACT ──► RISK ──► ALERT ──► REPORT
```

Jal-Rakshak bridges raw sensor telemetry with automated intelligence:
- **Synthetic Aperture Radar (SAR):** Night/all-weather observation via Sentinel-1 / RISAT imagery.
- **Deep Learning + Classical Validation:** YOLOv8 instance segmentation cross-validated against adaptive Otsu thresholding, Lee speckle filtering, and K-Means dark-spot extraction.
- **Reverse Hydrodynamics (Euler Hindcast):** Traces surface drift backwards using surface ocean current vectors and wind leeway transfer functions (3%).
- **Probable Source Likelihood Surface:** 2D Monte Carlo particle ensemble with Gaussian dispersion generating 50%, 75%, and 95% Bayesian credible contours.
- **AIS Traffic Reconstruction:** Progressive 4-stage filtering of vessel tracks, computing closest point of approach (CPA), speed changes, course anomalies, and AIS transmission gaps.
- **Shoreline Vulnerability & Coastal Impact:** Intersects forward drift cones with NOAA/IMO Environmental Sensitivity Index (ESI 1–10) coastal corridors.
- **Multi-Channel Alert Dispatch:** Automated, geofenced alerts for port authorities, disaster managers, and local fishing hamlets.
- **Forensic PDF Dossier:** One-click export of an official, classified investigation document.

---

## 4. Key Features

| Category | Capability | Implementation |
|---|---|---|
| **SAR Processing** | Lee speckle filtering, median filtering, contrast enhancement, land/sea masking | `sar/preprocessing.py`, `sar/classical.py` |
| **Spill Detection** | YOLOv8 instance segmentation trained on SAR oil slicks | `sar/detection.py` (`best.pt`) |
| **Validation** | IoU/Dice overlap analysis, look-alike heuristic scoring (low wind, biogenic films) | `sar/classical.py`, `sar/metrics.py` |
| **Characterization** | Area (km²), perimeter, major/minor axes, aspect ratio, solidity, empirical volume | `sar/geometry.py` |
| **Weathering & Age** | Fay spreading regime analysis, multi-temporal SAR growth back-projection | `sar/weathering.py` |
| **Ocean Hindcast** | Backward Euler numerical integration with expanding spatial uncertainty envelope | `ocean/hindcast.py` |
| **Source Likelihood** | 2D Bayesian credible contours (P50 red, P75 orange, P95 yellow) & Monte Carlo particles | `ocean/probability.py` |
| **AIS Analytics** | Spatial, temporal, heading, speed, and trajectory filtering; CPA distance/time calculation | `ais/filtering.py`, `ais/attribution.py` |
| **Attribution** | Explainable 0–100 candidate association score with anomaly breakdown cards | `ais/attribution.py` |
| **Drift Forecast** | +6h, +12h, +24h forward drift trajectory cones with leeway wind transport | `ocean/hindcast.py` |
| **Coastal Impact** | Shortest distance to coast, landfall projection, ESI rating, containment strategies | `coastal/impact.py`, `coastal/zones.py` |
| **Risk Engine** | Multi-factor risk scoring (LOW, MEDIUM, HIGH, CRITICAL) | `risk/scoring.py` |
| **Alerts & Warnings** | Geofenced community broadcasts, authority escalation, test alert simulations | `alerts/community.py` |
| **Reporting** | Machine-readable JSON and ReportLab multi-page official PDF export | `reporting/incident.py`, `reporting/pdf.py` |
| **Interactive UI** | Streamlit authority dashboard with Folium map and timeline scrubber (-180m to +60m) | `app.py` |

---

## 5. System Architecture

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    JAL-RAKSHAK                         │
                               │   AI Maritime Oil Spill Intelligence & Early Warning   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │  MONITORING GEOFENCE  │
                                               │ (Polygon / Sentinel-1)│
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │     SAR INGESTION     │
                                               │ (Swath / Patch Upload)│
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │   SAR PREPROCESSING   │
                                               │ (Lee / Median / Mask) │
                                               └───────────┬───────────┘
                                                           │
                                ┌──────────────────────────┴──────────────────────────┐
                                ▼                                                     ▼
                    ┌───────────────────────┐                             ┌───────────────────────┐
                    │   YOLO SEGMENTATION   │                             │ CLASSICAL VALIDATION  │
                    │   (Learned Network)   │                             │  (Lee/Otsu/K-Means)   │
                    └───────────┬───────────┘                             └───────────┬───────────┘
                                │                                                     │
                                └──────────────────────────┬──────────────────────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │  AGREEMENT & METRICS  │
                                               │ (IoU/Dice/Look-Alike) │
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │   CHARACTERIZATION    │
                                               │ (Area/Solidity/Volume)│
                                               └───────────┬───────────┘
                                                           │
                                ┌──────────────────────────┴──────────────────────────┐
                                ▼                                                     ▼
                    ┌───────────────────────┐                             ┌───────────────────────┐
                    │    WEATHERING & AGE   │                             │    OCEAN HINDCAST     │
                    │ (Fay Spreading Model) │                             │   (Euler Transport)   │
                    └───────────┬───────────┘                             └───────────┬───────────┘
                                │                                                     │
                                └──────────────────────────┬──────────────────────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │  SOURCE PROBABILITY   │
                                               │ (P50/P75/P95 Contours)│
                                               └───────────┬───────────┘
                                                           │
                                ┌──────────────────────────┴──────────────────────────┐
                                ▼                                                     ▼
                    ┌───────────────────────┐                             ┌───────────────────────┐
                    │  HISTORICAL AIS TRACKS│                             │   4-STAGE FILTERING   │
                    │   (Telemetric Feeds)  │                             │(Spatial/Time/Traj/CPA)│
                    └───────────┬───────────┘                             └───────────┬───────────┘
                                │                                                     │
                                └──────────────────────────┬──────────────────────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │ CANDIDATE ATTRIBUTION │
                                               │ (Explainable Scores)  │
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │FORWARD DRIFT FORECAST │
                                               │ (+6h / +12h / +24h)   │
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │    COASTAL IMPACT     │
                                               │ (Landfall ETA / ESI)  │
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │   MULTI-FACTOR RISK   │
                                               │(Low/Med/High/Critical)│
                                               └───────────┬───────────┘
                                                           │
                                ┌──────────────────────────┼──────────────────────────┐
                                ▼                          ▼                          ▼
                    ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
                    │   PORT AUTHORITIES    │  │  COASTAL RESIDENTS    │  │  LOCAL FISHERMEN      │
                    │ (Incident Operations) │  │  (Beach Notifications)│  │ (Marine Booming Zones)│
                    └───────────┬───────────┘  └───────────┬───────────┘  └───────────┬───────────┘
                                │                          │                          │
                                └──────────────────────────┴──────────────────────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────────┐
                                               │   INCIDENT DOSSIER    │
                                               │ (JSON & Official PDF) │
                                               └───────────────────────┘
```

---

## 6. SAR Ingestion & Preprocessing Pipeline

SAR imagery provides high-contrast radar backscatter signatures over the ocean. Smooth oil slicks dampen capillary and short gravity waves, appearing as characteristic low-backscatter "dark spots."

The `sar/preprocessing.py` module supports configurable filter stages:
1. **Speckle Reduction (Lee Filter):** Implements local statistics speckle filtering:
   $$\hat{I} = \bar{I} + W \cdot (I - \bar{I}), \quad W = \frac{\sigma^2 - \sigma_n^2}{\sigma^2}$$
2. **Median & Wiener Filtering:** Suppresses impulse noise and sensor dropouts without blurring sharp slick boundaries.
3. **Contrast Normalization (CLAHE):** Local adaptive histogram equalization to boost distinction between ocean clutter and thin slicks.
4. **Land/Sea Masking:** Eliminates coastal landmasses and false-positive terrestrial water bodies.

---

## 7. YOLOv8 Detection & Classical Validation

### Learned Detector
Jal-Rakshak utilizes a YOLOv8-seg neural network (`best.pt`) optimized for marine dark-spot segmentation, outputting instance polygon contours, confidence scores, and bounding boxes.

### Classical Cross-Validation (Independent Check)
To guard against deep learning false positives and hallucinated detections, `sar/classical.py` independently runs:
- **Adaptive Otsu Thresholding:** Dynamically isolates dark anomalies against local sea clutter.
- **K-Means Clustering:** Unsupervised 3-cluster radiometric segmentation (slick, clean water, land/vessel).
- **Mask Agreement:** Computes the Intersection-over-Union (IoU) and Sørensen-Dice coefficient between the YOLO mask and classical masks.
- **Look-Alike Risk Engine:** Evaluates environmental indicators (e.g., surface wind $< 3$ m/s, wave shadow zones, biogenic grease films) to estimate look-alike probability.

---

## 8. Spill Morphology & Age Estimation

The `sar/weathering.py` module evaluates slick geometry through physical spreading theory:
- **Fay Spreading Regimes (1971):**
  1. *Gravity-Inertia Regime* ($t < 1$ hr): Thickness and density drive initial rapid expansion.
  2. *Gravity-Viscous Regime* ($1 \le t \le 6$ hrs): Viscous shear balances gravity spreading.
  3. *Surface-Tension Viscous Regime* ($t > 6$ hrs): Slick elongates into windrows and breaks into tar patches.
- **Morphological Descriptors:** Aspect ratio ($\text{Length} / \text{Width}$), solidity ($\text{Area} / \text{Convex Hull Area}$), and fragmentation count ($N$).
- **Multi-Temporal SAR Growth Tracking:** When multiple satellite passes ($T_0, T_1, \dots$) are available, the system tracks true areal growth rate $dA/dt$ and drift velocity $dX/dt$, projecting backwards to estimate release timing.
- **Honest Fallback:** If the detection is smaller than 50 pixels or morphology is ambiguous, the system outputs `AGE: UNKNOWN` rather than fabricating numbers.

---

## 9. Oceanographic Hindcast & Probable Source Region

### Backward Trajectory Numerical Integration
Given observed spill centroid $(x_{\text{spill}}, y_{\text{spill}})$ at detection time $T_{\text{det}}$, the slick drift velocity is computed as:
$$\vec{V}_{\text{drift}} = \vec{V}_{\text{current}} + \alpha \cdot \vec{V}_{\text{wind}}$$
where $\alpha \approx 0.03$ (standard 3% wind leeway factor).

The backward trajectory is integrated using Euler time-stepping ($\Delta t = 5$ min):
$$\vec{x}(t - \Delta t) = \vec{x}(t) - \vec{V}_{\text{drift}} \cdot \Delta t$$
Spatial uncertainty grows quadratically with drift duration:
$$\sigma_{\text{spatial}}(t) = d_{\text{cumulative}}(t) \times \gamma, \quad \gamma = 0.35$$

### 2D Source Likelihood Surface
In `ocean/probability.py`, Monte Carlo particle ensemble endpoints seed a 2D Gaussian dispersion model generating concentric Bayesian credible zones:
- **P50 (Red, 50% Bayesian Envelope):** Core origin area ($r = 1.177 \sigma$).
- **P75 (Orange, 75% Envelope):** Intermediate dispersion zone ($r = 1.665 \sigma$).
- **P95 (Yellow, 95% Boundary):** Broad plausible envelope ($r = 2.447 \sigma$).

---

## 10. AIS Trajectory Reconstruction & Progressive Filtering

Raw maritime corridors can contain hundreds of transiting vessels. Jal-Rakshak filters irrelevant traffic across four successive stages:

```
Total Vessels in Sector (~100)
    │
    ▼ [Stage 1: Spatial Filter] (Distance to spill/origin ≤ 50 km)
Relevant Vessels (~40)
    │
    ▼ [Stage 2: Temporal Filter] (Transited within ±3h of release window)
Temporally Plausible (~15)
    │
    ▼ [Stage 3: Trajectory Filter] (Closest Approach Distance ≤ 10 km)
Trajectory Candidates (~5)
    │
    ▼ [Stage 4: Feature Scoring & Fusion]
Ranked Candidate Vessels (1 to 4)
```

For each candidate, the engine computes:
- Minimum distance to estimated origin ($d_{\text{CPA}}$).
- Time difference at closest approach ($\Delta t_{\text{CPA}}$).
- Heading alignment with ocean current drift.
- Anomalies: abrupt speed drops, sharp course deviations, or **AIS transmission gaps** (flagged neutrally as data gaps, not proof of wrongdoing).

---

## 11. Vessel Attribution & Multi-Factor Scoring

The candidate association score ($S \in [0, 100]$) is computed with explainable weights:

$$S = w_s S_{\text{spatial}} + w_t S_{\text{temporal}} + w_{\text{tr}} S_{\text{trajectory}} + w_h S_{\text{heading}} + w_v S_{\text{speed}} + w_d S_{\text{drift}}$$

```
MT ARCTIC STAR (MMSI: 419001234)
──────────────────────────────────────────────────────────
Association Score: 87.4 / 100  [PRIMARY CANDIDATE VESSEL]

Score Breakdown:
  • Spatial Consistency:   95.0%  (Passed 1.2 km from estimated origin)
  • Temporal Consistency:  90.0%  (Approach within 14 min of release)
  • Trajectory Alignment: 85.0%  (Directly intersected source envelope)
  • Heading Alignment:    88.0%  (Aligned with prevailing current axis)
  • Speed Profile:        82.0%  (Transit speed: 13.8 knots)
  • Behavioral Anomaly:   45.0%  (14-minute AIS reporting gap detected)

DISCLAIMER: Association scores represent statistical correlation and do NOT
constitute legal proof of culpability.
```

---

## 12. Forward Drift Prediction & Hydrodynamics

From the spill detection time forward, Jal-Rakshak projects future slick transport across multiple operational horizons:
- **+6 Hours:** Immediate response window for deployment of harbor boom barriers.
- **+12 Hours:** Medium-range dispersal trajectory.
- **+24 Hours:** Strategic horizon for shoreline defense and coastal evacuation.

Trajectories incorporate expanding error cones representing wind shear variability and tidal fluctuations.

---

## 13. Coastal & Environmental Impact Assessment

The `coastal/` package integrates NOAA / IMO Environmental Sensitivity Index (ESI 1–10) data along the Coromandel / Chennai coastal corridor:
- **Shoreline Proximity:** Haversine distance from slick centroid to nearest land point.
- **Landfall Projection & ETA:** Trajectory ray-casting calculating expected shoreline impact time and uncertainty bounds (e.g., $14.5 \pm 2.0$ hours).
- **Threatened Asset Scoring:** Identifies specific vulnerable infrastructure and habitats within the impact envelope:
  - *Pulicat Lake Bird Sanctuary* (ESI 10 — Mangroves & Salt Marshes)
  - *Marina Beach Olive Ridley Turtle Nesting Zone* (ESI 8 — Sheltered Sand Beaches)
  - *Nemmeli Desalination Plant Seawater Intake* (ESI 2 — Industrial Seawater Intake)
  - *Ennore Creek Mangroves* (ESI 10 — Critical Wetland)
  - *Royapuram Fishing Harbor* (ESI 4 — High-density Artisanal Fishery)
- **Actionable Countermeasures:** Automated deployment rules (e.g., *Deflection booming at river mouth; Dispersant ban in shallow waters $< 20\text{m}$ depth*).

---

## 14. Explainable Multi-Factor Risk Engine

The risk engine (`risk/scoring.py`) synthesizes physical and ecological factors into an overarching threat level:

$$\text{Risk Score} = \sum_{k} w_k \cdot f_k(\text{Area}, \text{Landfall ETA}, \text{ESI Vulnerability}, \text{Confidence})$$

- **LOW ($< 25$):** Small, contained offshore slick with no projected coastal landfall.
- **MEDIUM ($25 - 50$):** Moderate slick drifting parallel to coast; routine monitoring.
- **HIGH ($50 - 75$):** Large slick or landfall projected within 24 hours near recreational or fishing areas.
- **CRITICAL ($> 75$):** Imminent shoreline impact ($< 16$ hrs) threatening ESI 9–10 sensitive ecological wetlands or municipal water intakes.

---

## 15. Community Early Warning & Geofenced Alerts

Alerts are partitioned into two operational channels (`alerts/`):
1. **Authority Tactical Notifications:** Dispatch messages containing candidate vessel MMSI, landfall coordinates, and technical containment recommendations.
2. **Community Action Bulletins:** Tailored, plain-language alerts for artisanal fishermen, aquaculture farmers, and coastal panchayats:
   - Specific marine zones to avoid.
   - Recommended actions: *Halt net deployment, retrieve in-water lobster pots, activate intake boom curtains.*
   - All notifications default to **SIMULATION MODE** to prevent accidental live broadcasts during drills.

---

## 16. LangGraph 11-Node Orchestration Engine

The entire processing lifecycle is compiled into a stateful, conditional LangGraph workflow (`pipeline/graph.py`):

```
                     ┌──────────────────┐
                     │ START / INGEST   │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ 1. PREPROCESS    │ (Lee filter, CLAHE, masking)
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ 2. DETECT (YOLO) │
                     └────────┬─────────┘
                              │
             ┌────────────────┴────────────────┐
             │ [Spill Detected?]               │
             ▼ (Yes)                           ▼ (No)
     ┌──────────────────┐              ┌──────────────────┐
     │ 3. VALIDATE      │              │ 11. REPORT       │
     └────────┬─────────┘              └────────┬─────────┘
              │                                 │
              ▼                                 ▼
     ┌──────────────────┐                     [END]
     │ 4. CHARACTERIZE  │ (Area, Perimeter, Fay Age Estimation)
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │ 5. HINDCAST      │ (Backward Euler, 2D Probability Surface)
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │ 6. AIS CORRELATE │ (Progressive 4-stage filtering)
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │ 7. ATTRIBUTION   │ (Explainable Candidate Scoring)
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │ 8. FORECAST      │ (Forward Drift + Coastal Impact)
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │ 9. RISK ENGINE   │ (Multi-factor classification)
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │ 10. ALERTS       │ (Authority & Community simulation)
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │ 11. REPORT       │ (JSON & ReportLab PDF compilation)
     └────────┬─────────┘
              │
              ▼
            [END]
```

---

## 17. Investigation Reporting & PDF Dossier

Clicking **"📄 Official PDF"** in the dashboard generates a comprehensive, multi-page intelligence dossier (`reporting/pdf.py`):
- **Classification Banners:** Color-coded provenance tags (`OBSERVED`, `INFERRED`, `PREDICTED`, `SIMULATED`).
- **Telemetry Overview:** Complete geometric measurements and Fay spreading regime summary.
- **Candidate Vessels Dossier:** Tabular breakdown of transiting ships, closest point of approach, time delta, and association score.
- **Coastal Defense Guidelines:** Protective booming layout, barrier coordinates, and dispersant restriction alerts.
- **Sign-off Block:** Formal chain-of-custody signatures for reviewing admiralty officers.

---

## 18. Reproducible Chennai Coast Demo Scenario

Jal-Rakshak includes an offline, zero-dependency demonstration scenario (`demo/scenario.py`):
- **Location:** 25 km offshore of Chennai Port / Ennore Headland ($13.125^\circ\text{N}, 80.450^\circ\text{E}$).
- **Environment:** 0.48 m/s current bearing $118^\circ$, 6.2 m/s wind bearing $135^\circ$.
- **Synthetic SAR Imagery:** Procedurally generated SAR patch with calibrated radar backscatter characteristics detected reliably by YOLOv8 (`best.pt`).
- **Deterministic AIS Fleet:** 4 transiting vessels with complete historical trajectories:
  - `MT ARCTIC STAR` (Crude Tanker, closest approach 1.2 km, association score ~87%).
  - `MV PACIFIC TRADER` (Container Ship, transited at higher offset, score ~52%).
  - `OCEAN VOYAGER` (Bulk Carrier, transited outside release window, score ~31%).
  - `SEA EXPLORER` (Offshore Tug, low proximity, score ~18%).

---

## 19. Installation & Prerequisites

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.14).
- Windows, macOS, or Linux.
- 4 GB RAM minimum.

### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/Dhanush024/Jal-Rakshak.git
cd Jal-Rakshak

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 20. Configuration & Environment Variables

Create a local `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `JAL_RAKSHAK_APP_MODE` | `demo` | Toggle `demo` (offline simulation) or `live` |
| `AISSTREAM_API_KEY` | *(empty)* | Optional API key for live AISStream.io WebSockets |
| `DEFAULT_PIXEL_RESOLUTION_M` | `10.0` | Default pixel scale in meters for SAR swaths |
| `DEFAULT_CURRENT_SPEED_MS` | `0.48` | Surface ocean current speed in m/s |
| `DEFAULT_CURRENT_BEARING_DEG`| `118.0`| Surface ocean current direction (degrees) |
| `DEFAULT_WIND_SPEED_MS` | `6.2` | 10-meter wind speed in m/s |
| `DEFAULT_WIND_BEARING_DEG` | `135.0`| Wind direction (degrees) |
| `SCORE_WEIGHT_TEMPORAL` | `0.30` | Attribution weight for temporal proximity |
| `SCORE_WEIGHT_SPATIAL` | `0.25` | Attribution weight for CPA distance |
| `SCORE_WEIGHT_TRAJECTORY` | `0.20` | Attribution weight for source envelope intersection |

---

## 21. Running Locally

Launch the interactive Streamlit command center:

```bash
streamlit run app.py
```

1. Open `http://localhost:8501` in your browser.
2. Click **"🚀 Launch Chennai Coast Demonstration Scenario"**.
3. Watch all 11 pipeline nodes execute with real-time telemetry.
4. Interact with the **Forensic Timeline Scrubber** ($-180$ min to $+60$ min) to observe vessel positions at the exact release moment.
5. Inspect candidate vessel cards, coastal landfall projections, and download the official PDF report.

---

## 22. Automated Testing & Verification

The repository contains an exhaustive automated test suite with **44 tests** covering all modules:

```bash
python -m pytest tests/ -v
```

```
tests/test_advanced.py::TestSpillAgeEstimation::test_age_estimation_compact_fresh PASSED
tests/test_advanced.py::TestSpillAgeEstimation::test_age_estimation_elongated_weathered PASSED
tests/test_advanced.py::TestSpillAgeEstimation::test_age_estimation_tiny_fallback_unknown PASSED
tests/test_advanced.py::TestSpillAgeEstimation::test_multitemporal_sar_growth PASSED
tests/test_advanced.py::TestSpillAgeEstimation::test_multitemporal_tracker PASSED
tests/test_advanced.py::TestSourceProbabilityMap::test_source_probability_generation PASSED
tests/test_advanced.py::TestSourceProbabilityMap::test_source_probability_with_particles PASSED
tests/test_advanced.py::TestReportPDFExport::test_pdf_report_compilation PASSED
tests/test_coastal.py::TestCoastalZones::test_nearest_shoreline_point PASSED
tests/test_coastal.py::TestCoastalZones::test_sensitive_areas_defined PASSED
tests/test_coastal.py::TestCoastalImpactEngine::test_coastal_impact_calculation PASSED
tests/test_coastal.py::TestCoastalImpactEngine::test_threatened_assets_detected PASSED
tests/test_coastal.py::TestCoastalImpactEngine::test_countermeasures_and_restrictions PASSED
tests/test_coastal.py::TestCoastalImpactEngine::test_pipeline_integration PASSED
tests/test_core.py::TestGeospatial::test_haversine_known_distance PASSED
tests/test_core.py::TestOceanHindcast::test_hindcast_produces_trajectory PASSED
tests/test_core.py::TestOceanHindcast::test_forecast_produces_results PASSED
tests/test_core.py::TestRiskEngine::test_high_risk PASSED
tests/test_core.py::TestAISFiltering::test_spatial_filter PASSED
tests/test_core.py::TestPipelineGraph::test_pipeline_execution_full_demo PASSED
============================= 44 passed in 7.14s ==============================
```

---

## 23. Methodological Limitations & Uncertainty Accounting

To maintain scientific integrity, Jal-Rakshak openly communicates system uncertainties:
1. **Constant Hydrodynamic Assumption:** The default Euler hindcast model operates on locally constant current and wind vectors. In high-resolution operations, spatiotemporal current fields from regional models (e.g., INCOIS, HYCOM) should be ingested.
2. **Thickness & Volume Ambiguity:** SAR radar backscatter indicates surface roughness damping, not oil layer thickness. Volume calculations rely on empirical thickness assumptions (0.1–1.0 mm) and must be treated as order-of-magnitude estimates.
3. **AIS Vulnerabilities:** AIS data is subject to transponder power-downs, clock drifts, satellite reception blind spots, and spoofing. An AIS data gap is flagged as an anomaly, not proof of wrongdoing.
4. **Non-Adjudicative Purpose:** Vessel attribution scores reflect geometric and physical correlation. Final legal attribution requires physical oil-fingerprinting (GC-MS analysis) and onboard ballast inspections.

---

## 24. Data Sources & Future Work

### Open Data Foundations
- **SAR Imagery:** European Space Agency (ESA) Copernicus Sentinel-1; ISRO RISAT-1.
- **Ocean Currents & Wind:** INCOIS (Indian National Centre for Ocean Information Services); NOAA GFS / ECMWF surface wind fields.
- **Vessel Tracking:** AISStream.io open WebSocket telemetry; historical AIS archival databases.
- **Coastal Classification:** NOAA Environmental Sensitivity Index (ESI) mapping guidelines.

### Roadmap
- [ ] Integration of real-time Copernicus Open Access Hub / Planetary Computer STAC APIs.
- [ ] 3D particle Lagrangian spill model incorporating baroclinic ocean currents and wave Stokes drift.
- [ ] Direct WhatsApp / Telegram bot integration for low-bandwidth artisanal fishermen alerts.
- [ ] High-resolution bathymetric surf-zone modeling for nearshore oil stranding.

---

**Developed for the Smart India Hackathon (SIH) // Project Jal-Rakshak**  
*Protecting Coastal Ecosystems & Maritime Sovereignty through Artificial Intelligence.*
