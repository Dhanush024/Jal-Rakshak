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
