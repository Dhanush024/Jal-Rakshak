# Jal-Rakshak — Implementation Plan

## 1. Current Architecture Audit

### Existing Files

| File | Size | Purpose |
|------|------|---------|
| `app.py` | 681 lines | Streamlit dashboard — SAR upload, YOLO overlay, investigation mode with AIS timeline |
| `pipeline.py` | 294 lines | LangGraph pipeline — ingest → inference → drift → AIS correlation → validation → alert |
| `ais_engine.py` | 604 lines | Drift model, simulated AIS tracks, correlation scoring, geo utilities |
| `best.pt` | 6.7 MB | YOLOv8 segmentation model weights |
| `requirements.txt` | 6 deps | streamlit, folium, streamlit-folium, ultralytics, websocket-client, langgraph |
| `app_template.py` | empty | Unused placeholder |
| `temp_upload.jpg` | 53 KB | Leftover uploaded image (should not be committed) |
| `.devcontainer/` | — | Codespaces config with Python 3.11 |

### Existing Features (Working)

1. SAR Upload — manual JPG/PNG upload via sidebar
2. Geofencing — Folium map with Draw plugin for polygon/rectangle
3. YOLO Inference — best.pt segmentation, extracts polygon coordinates
4. Spill Overlay — red polygon overlay on uploaded image
5. Area/Volume Estimation — pixel-based with configurable resolution
6. Drift Model — backward drift with current + wind vectors
7. Simulated AIS — 4 deterministic vessel tracks around estimated origin
8. Vessel Correlation Scoring — weighted multi-factor scoring
9. Investigation Mode — timeline slider, vessel selector, Folium map with layers
10. Live AIS Toggle — AISStream.io WebSocket (single position report)
11. LangGraph Pipeline — 6-node linear graph

### Incomplete / Missing Features

1. No SAR preprocessing
2. No classical SAR validation
3. No look-alike detection
4. No segmentation evaluation metrics
5. No spill characterization beyond area/volume
6. No spill age estimation
7. No multi-temporal SAR support
8. No historical AIS reconstruction
9. No AIS traffic filtering pipeline
10. No ocean data integration (hard-coded)
11. No forward drift prediction
12. No coastal/environmental impact assessment
13. No risk engine
14. No community early warning system
15. No multi-channel notifications
16. No incident report generation
17. No testing
18. No logging/observability
19. No demo mode toggle

### Technical Debt

1. HARD-CODED API KEY in pipeline.py:193
2. No .gitignore
3. OpenCV runtime pip hack in app.py:6-31
4. YOLO model loaded per inference
5. Simulated data not labeled as DEMO
6. No error handling for missing model
7. Fixed demo coordinates regardless of input
8. Single spill assumption (only masks.xy[0])
9. Dead file — app_template.py
10. No type checking or linting

### Security Issues

| Severity | Issue | Location |
|----------|-------|----------|
| CRITICAL | API key committed to source | pipeline.py:193 |
| HIGH | No .gitignore | repository root |
| MEDIUM | No credential rotation noted | git history |

## 2. Recommended Architecture

See DECISIONS.md for architectural rationale.

## 3. Implementation Phases

| Phase | Name | Priority |
|-------|------|----------|
| 0 | Repository Audit | DONE |
| 1 | Security and Foundation | CRITICAL |
| 2 | Architecture Refactor | HIGH |
| 3 | Data Provider Abstraction | HIGH |
| 4-9 | SAR Pipeline | HIGH/MEDIUM |
| 10-12 | Spill Analysis | HIGH/MEDIUM |
| 13-17 | AIS Pipeline | HIGH |
| 18-21 | Ocean + Fusion | HIGH |
| 22-28 | Prediction + Alerts | HIGH/MEDIUM |
| 29-33 | Dashboard + Demo | HIGH |
| 34-40 | Testing + Polish | HIGH |

## 4. Risks

1. YOLO model compatibility with ultralytics version changes
2. AISStream.io key already compromised
3. No SAR test images or ground-truth masks in repository
4. Pixel-to-geo mapping entirely assumed (10m resolution)
5. Demo-vs-real data confusion in current code
