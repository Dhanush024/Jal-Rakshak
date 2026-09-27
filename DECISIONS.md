# Jal-Rakshak — Architecture Decision Records

## ADR-001: Preserve YOLO as Primary Detection

**Context:** The existing `best.pt` YOLOv8 segmentation model works for oil spill detection.
Classical SAR methods from the reference repository are available.

**Decision:** Keep YOLO as the primary detector. Use classical methods for independent validation,
not as a replacement. This follows the specification requirement (Rule 13, 14).

**Rationale:** YOLO provides fast, learned segmentation. Classical methods add interpretability
and cross-validation. Replacing YOLO would lose the existing trained model.

---

## ADR-002: Provider Pattern for External Data

**Context:** The system needs AIS, ocean, weather, and satellite data. Real APIs may be
unavailable during development and demo.

**Decision:** Implement a provider interface pattern. Every external data source has a
`RealProvider` and `DemoProvider`. The system never silently falls back to demo data.

**Rationale:** This satisfies the specification requirement to clearly distinguish
REAL / DEMO / UNAVAILABLE data modes.

---

## ADR-003: Streamlit Multi-Page Architecture

**Context:** The UI needs 9+ pages (overview, detection, vessels, reconstruction, etc.).
The current app is a single-file 681-line Streamlit script.

**Decision:** Use Streamlit's native page tabs/expanders within a single app.py entry point,
with UI logic split into separate modules under `ui/`. Avoid `pages/` directory approach
as it makes state sharing harder.

**Rationale:** Keeps the application as a single Streamlit app with clean module separation.
State is shared via `st.session_state`. Consistent sidebar across all views.

---

## ADR-004: Configuration via Environment Variables

**Context:** API keys were hard-coded in source. The system needs configurable parameters
for pixel resolution, drift model, scoring weights, etc.

**Decision:** Use a central `config/settings.py` that reads from environment variables
and `.env` files. Provide `.env.example` as documentation.

**Rationale:** Follows security best practices. Allows different configurations for
development, demo, and production without code changes.

---

## ADR-005: Candidate Vessel Language

**Context:** The specification explicitly forbids language that implies legal proof.

**Decision:** Throughout the codebase, always use:
- "candidate vessel" not "culprit" or "responsible vessel"
- "association score" / "correlation confidence" not "proof"
- "probable source region" not "exact source"
- "estimated" / "predicted" not stated as fact when uncertainty exists

**Rationale:** The system is decision support, not legal adjudication.

---

## ADR-006: LangGraph for Pipeline Orchestration

**Context:** The existing pipeline uses LangGraph with 6 nodes in a linear chain.

**Decision:** Expand the LangGraph pipeline to include all processing stages with
conditional branching (e.g., retry preprocessing if detection confidence low,
skip hindcast if no ocean data available).

**Rationale:** LangGraph provides stateful, debuggable workflow orchestration with
built-in checkpointing.

---

## ADR-007: Geospatial Distance/Bearing Module

**Context:** `ais_engine.py` contains `haversine_km`, `bearing_deg`, `destination_point`.
These are general-purpose geospatial utilities used across multiple modules.

**Decision:** Extract these into `geospatial/distance.py` as the single source of truth
for all geospatial calculations. All modules import from there.

**Rationale:** Avoids duplication and ensures consistent implementations across AIS,
ocean, risk, and coastal impact modules.

---

## ADR-008: Demo Mode as Explicit Toggle

**Context:** Current code silently uses simulated data without user awareness.

**Decision:** Add a clear `DEMO MODE` / `LIVE MODE` toggle in the sidebar.
When in demo mode, display a persistent banner: "🔶 DEMO MODE — Simulated Data".
Demo data is self-contained and works offline with no external API calls.

**Rationale:** Satisfies the specification requirement to never silently present
simulated data as real.

---

## ADR-009: Fay Spreading Model for Spill Age Estimation

**Context:** The problem statement requires estimating spill age where scientifically feasible,
while strictly prohibiting fabrication of arbitrary timestamps.

**Decision:** Implement J.A. Fay's physical spreading theory combined with morphological
indicators (solidity, aspect ratio elongation, patch count). If only a single image with
insufficient resolution (< 50 px) or ambiguous morphology is present, return `AGE: UNKNOWN`
with explicit scientific rationale. If multi-temporal SAR passes are available, project
areal growth $dA/dt$ back to point release.

**Rationale:** Preserves scientific credibility and prevents deceptive certainty.

---

## ADR-010: 2D Bayesian Source Likelihood Surface

**Context:** Hindcast drift estimates produce a single origin coordinate, but oceanic
turbulence and current shear create a spatial dispersion cloud.

**Decision:** Use Monte Carlo particle ensembles combined with 2D Gaussian dispersion
to generate concentric Bayesian credible zones: P50 (Red, 50% core), P75 (Orange, 75%),
and P95 (Yellow, 95% outer boundary). Overlay these contours on Folium with transparency.

**Rationale:** Accurately communicates spatial uncertainty to operational decision-makers
without presenting a single deterministic point as absolute truth.

---

## ADR-011: Official Incident Dossier PDF Generation via ReportLab

**Context:** Maritime authorities require an official, formal investigation report for
evidence logging and inter-agency coordination.

**Decision:** Use ReportLab Platypus with a dynamic two-pass canvas (`NumberedCanvas`)
to generate formatted, multi-page PDFs with running headers/footers ("Page X of Y"),
color-coded data classification badges (`OBSERVED`, `INFERRED`, `PREDICTED`, `SIMULATED`),
candidate vessel rankings, coastal defense recommendations, and formal review sign-off blocks.

**Rationale:** Provides an official, printable paper trail while embedding transparent
disclaimers on legal non-adjudication.
