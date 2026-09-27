# Jal-Rakshak: Technical Jury Q&A Reference Guide
**Problem Statement:** SIH26143 — Leveraging satellite imagery to determine Oil spills at sea along with AIS data correlations to identify vessel responsible for the spill.

---

### **1. Why Synthetic Aperture Radar (SAR) instead of Optical (RGB / Multispectral) Imagery?**
**Answer:**
Optical satellites (Sentinel-2, Landsat) cannot penetrate cloud cover and cannot acquire imagery during nighttime. Over tropical waters like the Bay of Bengal and Arabian Sea, cloud coverage exceeds 70% during the monsoon season. SAR (Sentinel-1 C-band) is an active microwave radar that operates 24/7 in all weather conditions. Oil dampens ocean capillary gravity waves, creating low-backscatter dark regions that are starkly identifiable in SAR.

---

### **2. Why YOLOv8 Segmentation instead of classical thresholding or UNet?**
**Answer:**
We employ a **hybrid, defense-in-depth architecture**:
- **YOLOv8-seg** performs rapid instance segmentation with high spatial feature learning, isolating distinct slick boundaries and separating overlapping features in under 150ms.
- We do not rely on YOLO alone: it is coupled with an independent **Classical Validation Module** (Lee despeckling, Otsu/K-means segmentation, and morphological filtering). This dual-path verification ensures deep learning generalizes while classical algorithms guard against hallucinations.

---

### **3. How do you distinguish oil spills from SAR "look-alikes"?**
**Answer:**
Natural phenomena like low wind zones (<3 m/s), biogenic algal films, internal waves, grease ice, and ship wakes also produce dark SAR patches. Jal-Rakshak filters look-alikes through:
1. **Morphological Solidity & Elongation:** Bilge dumps follow linear vessel corridors ($>3:1$ aspect ratio), unlike diffuse low-wind puddles.
2. **Contextual Wind Verification:** Wind fields under 2.5 m/s trigger look-alike warnings.
3. **Multi-Algorithm Mask Agreement:** We cross-correlate YOLO polygon boundaries with Otsu and Lee-filtered dark spots; low overlap flags high look-alike risk.

---

### **4. How do you estimate the age of an oil spill from satellite imagery?**
**Answer:**
We combine empirical spreading physics with morphological weathering:
1. **Fay Spreading Model:** Predicts slick radius expansion across gravity-inertia, gravity-viscous, and surface-tension-viscous regimes based on initial volume and elapsed time.
2. **Perimeter-to-Area Weathering:** As wave turbulence and ocean shear weather a slick, it fragments, increasing perimeter relative to area.
3. **Scientific Transparency:** When single-scene resolution is insufficient to constrain spreading parameters, the system outputs `AGE: UNKNOWN` rather than presenting an unverified number.

---

### **5. How does the Ocean Hindcast work?**
**Answer:**
We use a discrete **Lagrangian particle tracking model** with Euler-backward integration:
$$\vec{x}_{t - \Delta t} = \vec{x}_t - \left( \vec{u}_{\text{current}} + \alpha \vec{u}_{\text{wind}} \right) \Delta t$$
Where:
- $\vec{u}_{\text{current}}$ is the surface ocean current vector (m/s).
- $\vec{u}_{\text{wind}}$ is the 10m wind vector.
- $\alpha \approx 0.025 - 0.030$ is the empirical wind drift factor (Lehr & Simecek-Beatty).
- To capture turbulent diffusion and ocean model uncertainty, we release a Monte Carlo ensemble of 80 particles with Gaussian random walks, yielding calibrated **P50 (core 50%), P75, and P95 confidence envelopes**.

---

### **6. What happens if ocean current or wind data is unavailable or inaccurate?**
**Answer:**
1. If live ocean models (e.g. INCOIS / Copernicus Marine) cannot be reached, the system gracefully shifts to configurable historical regional defaults with a prominent **`SIMULATED`** or **`ASSUMED`** provenance badge.
2. The UI explicitly visualizes the growing spatial uncertainty cone ($\pm 15\%$ per hour backward). If current vectors have high variance, the source probability envelope widens proportionally, preventing false narrowing onto the wrong vessel.

---

### **7. Where does the historical AIS data come from?**
**Answer:**
The system is built on a clean provider architecture (`AISProvider`):
- **Live / Archival Ingestion:** Ingests standard NMEA CSV, GeoJSON, and JSON telemetry archives (matching USCG MarineCadastre, Danish Maritime Authority, and Indian coastal AIS formats).
- **Offline Demonstration:** Ships with a deterministic, calibrated 4-vessel scenario modeled on the Chennai/Ennore corridor for offline presentation reliability.
- The UI contains an interactive drag-and-drop uploader allowing authorities to supply raw coastal AIS dumps during a live investigation.

---

### **8. What happens when a vessel disables its AIS transponder (Dark Ship)?**
**Answer:**
Disabling AIS is a documented tactic during illegal bilge dumping:
1. **AIS Gap Analysis:** Our feature engineering calculates the time delta between consecutive AIS pings. If an AIS transponder abruptly goes silent near the hindcast corridor, it is recorded as an `AIS Coverage Gap` anomaly.
2. **Scientific Honesty:** Jal-Rakshak never treats a missing AIS ping as legal proof of guilt. It is presented as investigative evidence of non-reporting for Coast Guard maritime patrol aircraft (Dornier) verification.

---

### **9. How are candidate vessels scored and ranked?**
**Answer:**
We calculate a 0–100 **Association Score** based on transparent multi-criteria feature engineering:
- **Spatial Proximity (30 pts):** Minimum distance from vessel track to hindcast source envelope ($d_{\text{min}}$).
- **Temporal Alignment (25 pts):** Coincidence between vessel passage time and estimated discharge window ($|t_{\text{vessel}} - t_{\text{origin}}| < 30\text{ min}$).
- **Corridor Intersection (25 pts):** Direct geometric intersection with the drift trajectory cone.
- **Behavioral Profile (20 pts):** Speed anomalies (e.g. slow steaming during illegal pumping) and heading consistency.

---

### **10. Does a high association score prove legal responsibility for the spill?**
**Answer:**
**Absolutely not, and our system explicitly forbids that claim.**
Jal-Rakshak is a **decision-support platform** providing investigative intelligence. The platform labels all detections as **"Candidate Vessels"** with an **"Association Score"**. Every report includes a mandatory non-repudiation disclaimer stating that satellite radar and AIS correlation establish circumstantial spatial-temporal consistency, not legal culpability. Definitive attribution requires oil-fingerprinting (gas chromatography-mass spectrometry) by the Indian Coast Guard.

---

### **11. What is the difference between Real and Simulated data in Jal-Rakshak?**
**Answer:**
We strictly enforce a five-tier provenance taxonomy visible across the UI, maps, and PDF export:
- **`REAL` / `OBSERVED`:** User-uploaded SAR rasters, authentic AIS telemetry CSVs, georeferenced coordinates.
- **`REAL ALGORITHM / SIMULATED DATA`:** Real YOLO inference, real Euler-Lagrange hindcasting, real 4-stage filtering executed on calibrated Chennai demonstration data.
- **`INFERRED`:** Spill age, empirical volume, center of mass, drift speed.
- **`PREDICTED`:** Forward trajectory, landfall ETA, threatened coastal assets.
- **`SIMULATED`:** Default demonstration fleet positions, simulated SMS/Email notifications when external gateways are unconfigured.

---

### **12. How does the forward forecast predict coastal landfall?**
**Answer:**
Forward integration projects the slick forward for +6h, +12h, and +24h using forward surface current and wind vectors. At each hourly step, the model computes:
- Euclidean and haversine distance to the digitized Tamil Nadu / Andhra Pradesh coastline.
- Intersection with high-priority coastal polygons (Environmental Sensitivity Index 1–10: mangroves, turtle nesting sites, ports, desalination intakes).
- Estimated Time of Arrival (ETA) with $\pm 20\%$ windage uncertainty bands.

---

### **13. How are multi-channel alerts handled without sending spam?**
**Answer:**
Our `AlertManager` uses a role-based taxonomy:
- **Authorities (Coast Guard / DG Shipping):** Technical telemetry, high-precision coordinates, candidate MMSI numbers, and risk scoring.
- **Coastal Communities & Fishermen:** Plain-language safety bulletins advising exclusion zones, avoiding technical jargon.
- **Delivery Guardrails:** In offline or unconfigured environments, alerts run in `SIMULATION` mode, ensuring no external network calls fail or crash the dashboard.

---

### **14. Can Jal-Rakshak process full-resolution Sentinel-1 GeoTIFF scenes?**
**Answer:**
Yes. The `SARSceneLoader` reads TIFF/GeoTIFF raster metadata, extracts native bounding boxes and pixel resolutions, converts pixel coordinates to WGS-84 geographic coordinates, and tiles large scenes into standard 512×512 tiles for batched neural network inference before seamlessly re-projecting polygon coordinates to global latitude/longitude.

---

### **15. What makes Jal-Rakshak fundamentally different from existing academic projects?**
**Answer:**
Most existing tools stop at **Detection** (drawing a bounding box on an image).
Jal-Rakshak delivers the complete **Forensic Attribution & Decision Support Loop**:
$$\text{Detect} \longrightarrow \text{Validate} \longrightarrow \text{Trace} \longrightarrow \text{Attribute} \longrightarrow \text{Predict} \longrightarrow \text{Warn} \longrightarrow \text{Dossier}$$
We connect spaceborne radar to maritime AIS registries, run backward hydrodynamic hindcasts, rank candidate vessels transparently, and generate court-ready PDF dossiers in under 10 seconds.
