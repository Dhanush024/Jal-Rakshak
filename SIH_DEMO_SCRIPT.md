# Jal-Rakshak: 5-Minute SIH Jury Demonstration Script
**Problem Statement:** SIH26143 — Leveraging satellite imagery to determine Oil spills at sea along with AIS data correlations to identify vessel responsible for the spill.

---

### **Overview & Pitch (0:00 – 0:30)**
- **Presenter:**
  > "Respected Jury, maritime oil spills cause catastrophic ecological destruction along India's 7,516 km coastline. In incidents like the 2017 Ennore collision, identifying the source vessel and tracking coastal impact took days due to disconnected manual workflows.
  >
  > Today we present **Jal-Rakshak** — an end-to-end, forensic intelligence platform that executes the complete investigative lifecycle in under 10 seconds:
  > **DETECT → VALIDATE → TRACE → ATTRIBUTE → PREDICT → WARN**."

---

### **Phase 1: SAR Detection & Classical Validation (0:30 – 1:30)**
- **Action on Screen:**
  - Click **"🚀 Load Chennai Scenario"** in the sidebar.
  - Point to the raw synthetic Sentinel-1 SAR patch and run the 11-node LangGraph pipeline.
- **Presenter:**
  > "We begin with Synthetic Aperture Radar (SAR) imagery, which sees through night and cloud cover.
  > 
  > 1. **AI Segmentation:** Our YOLOv8 neural network extracts pixel-precise oil slick boundaries (IoU 0.88, 0.42 km² area).
  > 2. **Classical Validation Defense:** Deep learning alone can trigger false alarms on natural look-alikes like low-wind slicks or biogenic films. Jal-Rakshak runs an independent multi-algorithm classical filter (Lee despeckling, Otsu thresholding, and morphological solidity) yielding an 82% multi-spectral agreement score."

---

### **Phase 2: Forensic Hindcast & Source Region (1:30 – 2:30)**
- **Action on Screen:**
  - Scroll to the interactive Leaflet/Folium forensic map.
  - Highlight the detected spill (Red) and the backward drift trajectory (Orange line) leading to the P50/P75/P95 probability ellipses.
- **Presenter:**
  > "A spill detected at $T_0$ is already drifting. To find the source, we do not guess:
  > 
  > 1. **Spill Aging:** Fay spreading theory and perimeter-to-area weathering models estimate the slick is 2.5 to 3.5 hours old.
  > 2. **Ocean Hindcasting:** Using regional Bay of Bengal surface currents (0.45 m/s) and wind drift (2.5% windage factor), our Euler-Lagrange particle ensemble integrates backwards in time.
  > 3. Rather than an unrealistic single point, we generate calibrated **P50, P75, and P95 source probability envelopes** with Monte Carlo spatial dispersion."

---

### **Phase 3: AIS Correlation & Multi-Stage Vessel Filtering (2:30 – 3:30)**
- **Action on Screen:**
  - Zoom into the vessel trajectories crossing the source envelope.
  - Show the progressive 4-stage filter metrics:
    - **Total Fleet:** 4 vessels
    - **Spatial Filter:** 3 vessels
    - **Temporal Filter:** 2 vessels
    - **Corridor / CPA Filter:** 1 primary candidate
- **Presenter:**
  > "Now comes the core innovation: **Historical AIS Telemetry Fusion**.
  > 
  > Jal-Rakshak reconstructs all vessel tracks in the spatial-temporal window. Rather than simple proximity, we run a rigorous 4-stage filter:
  > - **Spatial:** Eliminates vessels beyond the drift zone.
  > - **Temporal:** Eliminates vessels not present during the discharge window ($T_{-180\text{m}}$ to $T_{-120\text{m}}$).
  > - **Corridor Intersection:** Tests geometric intersection with the hindcast drift cone.
  > - **Closest Point of Approach (CPA):** Calculates exact minimum separation distance and time."

---

### **Phase 4: Candidate Scoring & Explainable Attribution (3:30 – 4:15)**
- **Action on Screen:**
  - Point to the Candidate Vessels table.
  - Highlight **MT ARCTIC STAR (MMSI: 419001234)** with an Association Score of **88/100**.
  - Show the breakdown radar/bars: Spatial (92%), Temporal (90%), Trajectory (95%), Behavioral (75%).
- **Presenter:**
  > "Critically, Jal-Rakshak is an objective forensic tool, not an automated judge.
  > 
  > - We term this **Candidate Vessel** with an **Association Score**, never 'the culprit'.
  > - Every point in the 0–100 score is decomposed into explainable factors: CPA distance (0.34 km), time coincidence (within 8 minutes), speed consistency (13.8 kts tanker speed profile), and trajectory overlap.
  > - Other vessels (e.g. *MV PACIFIC TRADER* at 42/100, *SEA EXPLORER* at 28/100) are systematically de-prioritized with clear mathematical evidence."

---

### **Phase 5: Forward Forecast, Coastal Impact & PDF Dossier (4:15 – 5:00)**
- **Action on Screen:**
  - Show the forward forecast trajectory (+6h, +12h, +24h) approaching Pulicat Lake / Ennore Creek.
  - Show the Coastal Threat card (Environmental Sensitivity Index 8/10, Landfall ETA: 14.2 hrs).
  - Click **"📥 Download Official PDF Incident Dossier"** and open the two-pass generated report.
- **Presenter:**
  > "Simultaneously, the platform looks forward:
  > 
  > 1. **Forward Drift Forecast:** Predicts landfall in 14.2 hours near the ecologically sensitive Ennore mangrove sanctuary.
  > 2. **Multi-Channel Alerts:** Automatically dispatches structured advisories to the Indian Coast Guard and port authorities, while formatting simplified advisories for local fishing communities.
  > 3. **Audit-Ready Evidence Dossier:** Generates a comprehensive, cryptographic-quality PDF report complete with metadata, satellite coordinates, candidate ranking, and non-repudiation disclaimers.
  > 
  > Jal-Rakshak bridges satellite remote sensing and maritime law enforcement into a unified, transparent decision-support system for India. Thank you, and we welcome your questions!"
