"""
Jal-Rakshak — UI Components & Design System Tokens
===================================================
Phase A: Reusable, glassy, minimal maritime presentation helpers.

Provides:
- Core design tokens (colors, typography scales, provenance levels, statuses)
- Safe HTML rendering with markdown indentation sanitization
- Minimal Progress Rail (01 SAR -> 02 VALIDATION -> 03 AIS -> 04 SOURCE -> 05 DRIFT -> 06 REPORT)
- Compact Stage Banner (Stage XX, Title, Purpose, Key Values, Single Primary Action)
- Compact Measurement/Metric Cards (Tabular monospace values, minimal footprint)
- Evidence & Signal Quorum Selectors (Validated, Supported, Consistent, Rejected, Insufficient)
- Candidate Vessel Cards (Ranked evidence cards with association %, distance, trajectory)
- Segmented Layer Controls (RAW, YOLO, LAND, CLASSICAL, CONSENSUS, FINAL)
- Analytical Evidence Chain (Spill -> Backtrack -> Source Region -> Vessel Intersection)
- Provenance & Environment Badges (OBSERVED, DERIVED, ESTIMATED, UNAVAILABLE, DEMO, LIVE)
- Calm, honest Empty States (No fake telemetry when data is missing)
- Diagnostic Matrix Workspace (6-panel compact view)
"""

import html
from typing import Any, Dict, List, Optional, Tuple, Union
import streamlit as st


# =========================================================================
# 1. CORE DESIGN TOKENS
# =========================================================================

class DesignTokens:
    """Design tokens matching the atmospheric, minimal, glassy naval aesthetic."""

    # Surface foundations
    BG_VOID = "#040810"
    BG_BASE = "#070c18"
    BG_GLASS = "rgba(10, 16, 28, 0.72)"
    BG_GLASS_ELEVATED = "rgba(13, 20, 36, 0.85)"
    BG_GLASS_SUBTLE = "rgba(7, 12, 22, 0.55)"
    BG_HOVER = "rgba(18, 28, 50, 0.75)"

    # Borders & Dividers
    BORDER_CRISP = "rgba(255, 255, 255, 0.08)"
    BORDER_SUBTLE = "rgba(255, 255, 255, 0.05)"
    BORDER_ACCENT = "rgba(56, 189, 248, 0.28)"
    BORDER_ACCENT_GLOW = "rgba(56, 189, 248, 0.55)"

    # Palette
    COLOR_PRIMARY = "#f8fafc"      # Off-white
    COLOR_SECONDARY = "#94a3b8"    # Muted blue-gray
    COLOR_MUTED = "#64748b"        # Dim blue-gray
    COLOR_DIM = "#475569"          # Dark blue-gray

    ACCENT_CYAN = "#38bdf8"        # Sky / ice blue
    ACCENT_ICE = "#7dd3fc"         # Light ice blue
    ACCENT_DEEP = "#0284c7"        # Oceanic blue

    WARNING_AMBER = "#f59e0b"      # Amber
    DANGER_RED = "#ef4444"         # Muted red
    SUCCESS_EMERALD = "#10b981"    # Emerald / teal

    # Typography
    FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    FONT_MONO = "'JetBrains Mono', 'SF Mono', Consolas, monospace"

    # Provenance classifications
    PROVENANCE_OBSERVED = "OBSERVED"
    PROVENANCE_DERIVED = "DERIVED"
    PROVENANCE_ESTIMATED = "ESTIMATED"
    PROVENANCE_UNAVAILABLE = "UNAVAILABLE"

    # Validation Quorum States
    SIGNAL_VALIDATED = "Validated"
    SIGNAL_SUPPORTED = "Supported"
    SIGNAL_CONSISTENT = "Consistent"
    SIGNAL_REJECTED = "Rejected"
    SIGNAL_INSUFFICIENT = "Insufficient evidence"

    # Standard Demo Stages (Section 4)
    DEMO_STAGES = [
        {"idx": 1, "code": "01", "name": "SAR", "title": "SAR Detection", "desc": "Synthetic aperture radar imagery and multi-layer masks"},
        {"idx": 2, "code": "02", "name": "VALIDATION", "title": "Validation Consensus", "desc": "Multi-signal physics and deep-learning quorum"},
        {"idx": 3, "code": "03", "name": "AIS", "title": "AIS Correlation", "desc": "Fleet proximity and deterministic candidate association"},
        {"idx": 4, "code": "04", "name": "SOURCE", "title": "Source Analysis", "desc": "Hydrodynamic backtrack and credible origin region"},
        {"idx": 5, "code": "05", "name": "DRIFT", "title": "Drift Forensics", "desc": "Advection hindcast and forward trajectory projection"},
        {"idx": 6, "code": "06", "name": "REPORT", "title": "Regulatory Report", "desc": "Evidence-backed, regulatory-ready incident dossier"},
    ]


# =========================================================================
# 2. HTML SANITIZATION & SAFE INJECTION
# =========================================================================

def clean_html_for_render(html_str: str) -> str:
    """
    Renders HTML cleanly into Streamlit, preventing markdown code block escaping bugs.
    Strips leading and trailing whitespace from every line so that no line begins
    with 4 spaces or tabs that would trigger markdown indented code blocks.
    """
    if not html_str:
        return ""
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    return "\n".join(clean_lines)


def render_html(html_str: str) -> None:
    """Renders sanitized HTML directly into Streamlit without raw markdown traps."""
    cleaned = clean_html_for_render(html_str)
    if cleaned:
        st.markdown(cleaned, unsafe_allow_html=True)


# =========================================================================
# 3. PROVENANCE & ENVIRONMENT BADGES
# =========================================================================

def render_provenance_badge(status: str) -> str:
    """
    Generates HTML string for provenance classification badges.
    Levels: OBSERVED | DERIVED | ESTIMATED | UNAVAILABLE
    """
    s = (status or "").upper().strip()
    if s == DesignTokens.PROVENANCE_OBSERVED:
        css_cls = "badge-provenance-observed"
        label = "OBSERVED"
    elif s == DesignTokens.PROVENANCE_DERIVED:
        css_cls = "badge-provenance-derived"
        label = "DERIVED"
    elif s == DesignTokens.PROVENANCE_ESTIMATED:
        css_cls = "badge-provenance-estimated"
        label = "ESTIMATED"
    else:
        css_cls = "badge-provenance-unavailable"
        label = "UNAVAILABLE"

    return f'<span class="badge-provenance {css_cls}">{html.escape(label)}</span>'


def render_mode_badge(mode: str) -> str:
    """
    Renders environment badge ensuring DEMO/SIMULATION is clearly distinguished from LIVE.
    """
    m = (mode or "").lower().strip()
    if m == "demo":
        return '<span class="hdr-mode-pill mode-pill-demo">DEMO / SIMULATION</span>'
    elif m == "live":
        return '<span class="hdr-mode-pill mode-pill-live">LIVE OPERATIONS</span>'
    return '<span class="hdr-mode-pill">OPERATIONS</span>'


# =========================================================================
# 4. MINIMAL PROGRESS RAIL (Section 4)
# =========================================================================

def render_progress_rail(
    current_step: int,
    steps: Optional[List[Dict[str, Any]]] = None,
    key_prefix: str = "demo_rail",
) -> Optional[int]:
    """
    Persistent minimal progress rail:
    01 SAR  ->  02 VALIDATION  ->  03 AIS  ->  04 SOURCE  ->  05 DRIFT  ->  06 REPORT

    Active stage: visually active (cyan accent, high contrast)
    Completed stages: quiet (subtle checkmark, muted text)
    Future stages: muted
    Returns clicked step index if clicked, else None.
    """
    if steps is None:
        steps = DesignTokens.DEMO_STAGES

    clicked_step: Optional[int] = None
    total = len(steps)

    cols = st.columns(total)
    for i, s in enumerate(steps):
        step_idx = s.get("idx", i + 1)
        code = s.get("code", f"{step_idx:02d}")
        name = s.get("name", s.get("title", f"STAGE {step_idx}")).upper()

        is_active = (step_idx == current_step)
        is_completed = (step_idx < current_step)

        if is_active:
            btn_type = "primary"
            prefix_symbol = "●"
        elif is_completed:
            btn_type = "secondary"
            prefix_symbol = "✓"
        else:
            btn_type = "secondary"
            prefix_symbol = "○"

        label = f"{prefix_symbol} {code} {name}"
        btn_key = f"{key_prefix}_step_{step_idx}"

        with cols[i]:
            if st.button(label, key=btn_key, type=btn_type, use_container_width=True):
                clicked_step = step_idx

    return clicked_step


# =========================================================================
# 5. COMPACT STAGE BANNER (Section 13)
# =========================================================================

def render_stage_banner(
    stage_num: int,
    title: str,
    purpose: str,
    metrics: Optional[List[Dict[str, Any]]] = None,
    action_label: Optional[str] = None,
    action_key: Optional[str] = None,
    action_type: str = "primary",
    provenance: Optional[str] = None,
    eyebrow: Optional[str] = None,
) -> bool:
    """
    Unified compact stage header replacing text-heavy blocks:
    EYEBROW
    TITLE
    one-line purpose
    compact metrics row
    ONE primary action button

    Returns True if primary action button was clicked.
    """
    stage_code = eyebrow if eyebrow else f"{stage_num:02d}"
    action_clicked = False

    metrics_html = ""
    if metrics:
        metric_items = []
        for m in metrics:
            m_label = html.escape(str(m.get("label", "")))
            m_val = html.escape(str(m.get("value", "")))
            m_unit = html.escape(str(m.get("unit", "")))
            unit_span = f'<span class="compact-metric-unit">{m_unit}</span>' if m_unit else ""
            metric_items.append(f"""
            <div class="stage-metric-chip">
                <span class="stage-metric-label">{m_label}</span>
                <span class="stage-metric-val">{m_val}{unit_span}</span>
            </div>
            """)
        metrics_html = f'<div class="stage-metrics-row">{"".join(metric_items)}</div>'

    prov_html = render_provenance_badge(provenance) if provenance else ""

    banner_markup = f"""
    <div class="stage-header-box">
        <div class="stage-header-meta">
            <span class="stage-code">{stage_code}</span>
            {prov_html}
        </div>
        <div class="stage-title-row">
            <h2 class="stage-headline">{html.escape(title)}</h2>
        </div>
        <p class="stage-purpose-line">{html.escape(purpose)}</p>
        {metrics_html}
    </div>
    """
    render_html(banner_markup)

    if action_label and action_key:
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button(action_label, key=action_key, type=action_type, use_container_width=True):
                action_clicked = True

    return action_clicked


# =========================================================================
# 6. COMPACT METRICS READOUT (Section 5, 11, 28)
# =========================================================================

def render_compact_metrics(metrics: List[Dict[str, Any]]) -> None:
    """
    Renders compact quantitative measurements without giant dashboard cards.
    Each item: {label: str, value: str, unit: Optional[str], accent: Optional[str], provenance: Optional[str]}
    """
    if not metrics:
        return

    chips = []
    for m in metrics:
        lbl = html.escape(str(m.get("label", "")))
        val = html.escape(str(m.get("value", "")))
        unit = html.escape(str(m.get("unit", "")))
        accent = m.get("accent", "default")
        prov = m.get("provenance")
        prov_tag = f" {render_provenance_badge(prov)}" if prov else ""

        val_class = "metric-val-default"
        if accent == "cyan":
            val_class = "metric-val-cyan"
        elif accent == "amber":
            val_class = "metric-val-amber"
        elif accent == "green":
            val_class = "metric-val-green"
        elif accent == "red":
            val_class = "metric-val-red"

        unit_str = f'<span class="compact-unit">{unit}</span>' if unit else ""

        chips.append(f"""
        <div class="compact-metric-pill">
            <div class="compact-metric-label">{lbl}{prov_tag}</div>
            <div class="compact-metric-value {val_class}">{val}{unit_str}</div>
        </div>
        """)

    html_out = f'<div class="compact-metrics-grid">{"".join(chips)}</div>'
    render_html(html_out)


# =========================================================================
# 7. SIGNAL EVIDENCE TABLE (Section 7)
# =========================================================================

def render_signal_evidence_table(signals: List[Dict[str, Any]]) -> None:
    """
    Renders evidence selector with concise signal states:
    signal name | status badge | one-line reason

    Avoids paragraph walls and maintains scientific traceability.
    """
    if not signals:
        return

    rows = []
    for s in signals:
        name = html.escape(str(s.get("name", "")))
        status = s.get("status", "Validated")
        reason = html.escape(str(s.get("reason", "")))
        metric = html.escape(str(s.get("metric", "")))
        metric_span = f'<span class="signal-metric">{metric}</span>' if metric else ""

        # Status badge class
        st_lower = status.lower()
        if "validated" in st_lower or "confirmed" in st_lower or "passed" in st_lower:
            badge_class = "signal-badge-success"
            icon = "✓"
        elif "supported" in st_lower or "consistent" in st_lower:
            badge_class = "signal-badge-cyan"
            icon = "✓"
        elif "rejected" in st_lower or "failed" in st_lower:
            badge_class = "signal-badge-danger"
            icon = "✗"
        else:
            badge_class = "signal-badge-neutral"
            icon = "○"

        rows.append(f"""
        <div class="signal-evidence-row">
            <div class="signal-col-name">
                <span class="signal-name-text">{name}</span>
            </div>
            <div class="signal-col-status">
                <span class="signal-badge {badge_class}">{icon} {html.escape(status)}</span>
            </div>
            <div class="signal-col-reason">
                <span class="signal-reason-text">{reason}</span>
                {metric_span}
            </div>
        </div>
        """)

    markup = f"""
    <div class="signal-evidence-container">
        {"".join(rows)}
    </div>
    """
    render_html(markup)


# =========================================================================
# 8. CANDIDATE VESSEL CARDS (Section 8)
# =========================================================================

def render_candidate_vessel_card(
    rank: int,
    vessel: Dict[str, Any],
    is_selected: bool = False,
    key: Optional[str] = None,
) -> bool:
    """
    Renders compact candidate evidence card:
    #1 MV EXAMPLE
    ASSOCIATION: 82%
    DISTANCE: 4.2 km
    TIME ALIGNMENT: Strong
    TRAJECTORY: Consistent
    BEHAVIOR: Relevant
    [ VIEW TRACK ]

    Returns True if user clicked [ VIEW TRACK ].
    """
    name = html.escape(str(vessel.get("name", "UNKNOWN VESSEL")))
    mmsi = html.escape(str(vessel.get("mmsi", "—")))
    vtype = html.escape(str(vessel.get("vessel_type", "Vessel")))
    score = float(vessel.get("score", vessel.get("association_score", 0.0)))
    dist = vessel.get("distance_km", "—")
    dist_str = f"{dist:.1f} km" if isinstance(dist, (int, float)) else str(dist)
    time_align = html.escape(str(vessel.get("time_alignment", "Coincident")))
    traj = html.escape(str(vessel.get("trajectory", "Consistent")))
    behavior = html.escape(str(vessel.get("behavior", "Transit")))

    selected_class = "candidate-card-selected" if is_selected else ""
    rank_str = f"#{rank}"

    card_markup = f"""
    <div class="candidate-evidence-card {selected_class}">
        <div class="cand-card-top">
            <span class="cand-rank">{rank_str}</span>
            <div class="cand-identity">
                <div class="cand-name">{name}</div>
                <div class="cand-meta">MMSI: {mmsi} • {vtype}</div>
            </div>
            <div class="cand-score-box">
                <span class="cand-score-label">ASSOCIATION</span>
                <span class="cand-score-val">{score:.1f}%</span>
            </div>
        </div>
        <div class="cand-evidence-grid">
            <div class="cand-ev-item">
                <span class="cand-ev-lbl">DISTANCE</span>
                <span class="cand-ev-val">{dist_str}</span>
            </div>
            <div class="cand-ev-item">
                <span class="cand-ev-lbl">TIME ALIGNMENT</span>
                <span class="cand-ev-val">{time_align}</span>
            </div>
            <div class="cand-ev-item">
                <span class="cand-ev-lbl">TRAJECTORY</span>
                <span class="cand-ev-val">{traj}</span>
            </div>
            <div class="cand-ev-item">
                <span class="cand-ev-lbl">BEHAVIOR</span>
                <span class="cand-ev-val">{behavior}</span>
            </div>
        </div>
    </div>
    """
    render_html(card_markup)

    btn_key = key or f"cand_track_btn_{mmsi}_{rank}"
    btn_label = "● TRACK FOCUSED" if is_selected else "VIEW TRACK ▶"
    btn_type = "secondary" if is_selected else "primary"

    return st.button(btn_label, key=btn_key, type=btn_type, use_container_width=True)


def render_candidate_vessel_list(
    candidates: List[Dict[str, Any]],
    selected_mmsi: Optional[str] = None,
    key_prefix: str = "cand",
) -> Optional[str]:
    """
    Renders ranked candidate evidence cards and returns clicked vessel MMSI if any.
    Deterministic association ranking, no speculative labels.
    """
    if not candidates:
        render_empty_state("NO CANDIDATES IDENTIFIED", "No vessels correlated within the spatiotemporal search window.")
        return None

    # Deterministic ranking by association score
    sorted_candidates = sorted(
        candidates,
        key=lambda x: float(x.get("score", x.get("association_score", 0.0))),
        reverse=True,
    )

    clicked_mmsi: Optional[str] = None

    for i, c in enumerate(sorted_candidates):
        rank = i + 1
        mmsi = str(c.get("mmsi", ""))
        is_sel = (selected_mmsi is not None and mmsi == selected_mmsi)
        btn_key = f"{key_prefix}_select_{mmsi}_{rank}"

        if render_candidate_vessel_card(rank, c, is_selected=is_sel, key=btn_key):
            clicked_mmsi = mmsi

    return clicked_mmsi


# =========================================================================
# 9. SEGMENTED LAYER CONTROL (Section 5, 9, 16)
# =========================================================================

def render_segmented_layer_control(
    layers: List[Dict[str, str]],
    active_layer_id: str,
    key_prefix: str = "layer_ctrl",
    on_change_state_key: Optional[str] = None,
) -> str:
    """
    Renders compact segmented layer control:
    [ COMPOSITE ] [ RAW ] [ YOLO ] [ LAND ] [ OCEAN ] [ CLASSICAL ] [ CONSENSUS ] [ FINAL ]
    or [ RAW ] [ MASKS ] [ COMPOSITE ]

    Returns the chosen layer_id without double-click lag.
    """
    if not layers:
        return active_layer_id

    options = [layer.get("id", str(i)) for i, layer in enumerate(layers)]
    labels_map = {layer.get("id", str(i)): layer.get("label", layer.get("id", str(i))).upper() for i, layer in enumerate(layers)}

    # Native Streamlit segmented_control if available (instant 1-click update)
    if hasattr(st, "segmented_control"):
        ctrl_key = f"{key_prefix}_seg"
        default_val = active_layer_id if active_layer_id in options else options[0]

        # Sync widget key with external active_layer_id if changed externally
        if ctrl_key not in st.session_state:
            st.session_state[ctrl_key] = default_val
        elif active_layer_id in options and st.session_state.get(f"{key_prefix}_last_ext") != active_layer_id:
            st.session_state[ctrl_key] = active_layer_id
            st.session_state[f"{key_prefix}_last_ext"] = active_layer_id

        selected = st.segmented_control(
            "Select Layer",
            options=options,
            default=st.session_state.get(ctrl_key, default_val),
            format_func=lambda opt: labels_map.get(opt, opt),
            key=ctrl_key,
            label_visibility="collapsed",
            width="stretch",
        )
        chosen = selected if selected else st.session_state.get(ctrl_key, default_val)
        if on_change_state_key:
            st.session_state[on_change_state_key] = chosen
        return chosen

    # Fallback with pre-render on_click callback to prevent 2-click lag
    state_store_key = f"{key_prefix}_active_state"
    if state_store_key not in st.session_state:
        st.session_state[state_store_key] = active_layer_id

    current_active = st.session_state.get(state_store_key, active_layer_id)
    if current_active not in options:
        current_active = active_layer_id

    def _set_active_layer(target_lid):
        st.session_state[state_store_key] = target_lid
        if on_change_state_key:
            st.session_state[on_change_state_key] = target_lid

    cols = st.columns(len(layers))
    for i, layer in enumerate(layers):
        lid = layer.get("id", str(i))
        label = labels_map.get(lid, lid)
        is_active = (lid == current_active)
        btn_type = "primary" if is_active else "secondary"
        btn_key = f"{key_prefix}_{lid}"

        with cols[i]:
            st.button(
                label,
                key=btn_key,
                type=btn_type,
                use_container_width=True,
                on_click=_set_active_layer,
                args=(lid,),
            )

    return st.session_state.get(state_store_key, active_layer_id)


# =========================================================================
# 10. ANALYTICAL EVIDENCE CHAIN (Section 10)
# =========================================================================

def render_evidence_chain(chain_nodes: List[Dict[str, str]]) -> None:
    """
    Renders visual evidence flow:
    SPILL OBSERVED  ↓  BACKTRACK  ↓  SOURCE REGION  ↓  VESSEL INTERSECTION
    """
    if not chain_nodes:
        return

    node_htmls = []
    total = len(chain_nodes)

    for i, node in enumerate(chain_nodes):
        title = html.escape(str(node.get("title", "")))
        desc = html.escape(str(node.get("desc", "")))
        prov = node.get("provenance")
        prov_badge = f" {render_provenance_badge(prov)}" if prov else ""

        connector = ""
        if i < total - 1:
            connector = '<div class="chain-arrow">↓</div>'

        node_htmls.append(f"""
        <div class="chain-step-card">
            <div class="chain-step-header">
                <span class="chain-step-num">{i+1:02d}</span>
                <span class="chain-step-title">{title}</span>
                {prov_badge}
            </div>
            <div class="chain-step-desc">{desc}</div>
        </div>
        {connector}
        """)

    markup = f"""
    <div class="evidence-chain-flow">
        {"".join(node_htmls)}
    </div>
    """
    render_html(markup)


# =========================================================================
# 11. HONEST EMPTY STATES (Rule 0 & Section 14, 27)
# =========================================================================

def render_empty_state(
    title: str,
    description: str,
    action_label: Optional[str] = None,
    action_key: Optional[str] = None,
) -> bool:
    """
    Renders calm, honest empty state when real data is unavailable.
    Never fabricates telemetry or vessels to fill empty screens.
    """
    markup = f"""
    <div class="empty-state-panel">
        <div class="empty-state-icon">○</div>
        <div class="empty-state-title">{html.escape(title)}</div>
        <div class="empty-state-desc">{html.escape(description)}</div>
    </div>
    """
    render_html(markup)

    clicked = False
    if action_label and action_key:
        _, c2, _ = st.columns([1, 2, 1])
        with c2:
            if st.button(action_label, key=action_key, type="secondary", use_container_width=True):
                clicked = True
    return clicked


# =========================================================================
# 12. DIAGNOSTIC MATRIX PREVIEW (Section 5 & 6)
# =========================================================================

def render_diagnostic_matrix_preview(panels: List[Dict[str, Any]]) -> None:
    """
    Renders expandable 6-panel diagnostic workspace:
    RAW SAR | LAND/OCEAN | YOLO | CLASSICAL | CONSENSUS | FINAL
    """
    if not panels:
        return

    cols = st.columns(min(len(panels), 3))
    for i, p in enumerate(panels):
        col_idx = i % 3
        title = html.escape(str(p.get("title", f"PANEL {i+1}")))
        status = html.escape(str(p.get("status", "AVAILABLE")))
        img_obj = p.get("image")

        with cols[col_idx]:
            header_markup = f"""
            <div class="diag-matrix-header">
                <span class="diag-panel-title">{title}</span>
                <span class="diag-panel-status">{status}</span>
            </div>
            """
            render_html(header_markup)
            if img_obj is not None:
                st.image(img_obj, use_container_width=True)
            else:
                render_empty_state(title, "Layer data not generated.")
