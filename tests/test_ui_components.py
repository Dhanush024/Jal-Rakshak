"""
Jal-Rakshak — UI Components & Design Tokens Tests
==================================================
Unit and regression tests for Phase A presentation tokens and reusable UI helpers.
"""

import sys
import os
import re
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_components import (
    DesignTokens,
    clean_html_for_render,
    render_provenance_badge,
    render_mode_badge,
    render_stage_banner,
    render_compact_metrics,
    render_signal_evidence_table,
    render_candidate_vessel_card,
    render_candidate_vessel_list,
    render_evidence_chain,
    render_empty_state,
    render_segmented_layer_control,
)


class TestDesignTokens:
    """Verifies that all design tokens conform to the glassy maritime visual specification."""

    def test_surface_tokens_valid(self):
        """Surface foundations must be dark ocean/navy colors."""
        assert DesignTokens.BG_VOID.startswith("#")
        assert DesignTokens.BG_BASE.startswith("#")
        assert "rgba(" in DesignTokens.BG_GLASS
        assert "rgba(" in DesignTokens.BG_GLASS_ELEVATED
        assert "rgba(" in DesignTokens.BORDER_CRISP

    def test_accent_palette_restrained(self):
        """Palette must feature ice blue/cyan, amber, red, and emerald."""
        assert DesignTokens.ACCENT_CYAN.startswith("#")
        assert DesignTokens.WARNING_AMBER.startswith("#")
        assert DesignTokens.DANGER_RED.startswith("#")
        assert DesignTokens.SUCCESS_EMERALD.startswith("#")

    def test_provenance_classifications(self):
        """Four strict scientific provenance classifications must be defined."""
        assert DesignTokens.PROVENANCE_OBSERVED == "OBSERVED"
        assert DesignTokens.PROVENANCE_DERIVED == "DERIVED"
        assert DesignTokens.PROVENANCE_ESTIMATED == "ESTIMATED"
        assert DesignTokens.PROVENANCE_UNAVAILABLE == "UNAVAILABLE"

    def test_validation_signal_states(self):
        """Validation signals must use scientifically defensible terms."""
        assert DesignTokens.SIGNAL_VALIDATED == "Validated"
        assert DesignTokens.SIGNAL_SUPPORTED == "Supported"
        assert DesignTokens.SIGNAL_CONSISTENT == "Consistent"
        assert DesignTokens.SIGNAL_REJECTED == "Rejected"
        assert DesignTokens.SIGNAL_INSUFFICIENT == "Insufficient evidence"

    def test_demo_stages_sequence(self):
        """Demo must define exactly 6 sequential stages conforming to Section 4."""
        stages = DesignTokens.DEMO_STAGES
        assert len(stages) == 6
        codes = [s["code"] for s in stages]
        names = [s["name"] for s in stages]
        assert codes == ["01", "02", "03", "04", "05", "06"]
        assert names == ["SAR", "VALIDATION", "AIS", "SOURCE", "DRIFT", "REPORT"]


class TestHTMLSanitizationAndSafety:
    """Verifies that HTML is stripped of indentation traps to prevent visible raw code blocks."""

    def test_clean_html_strips_indents(self):
        indented = """
            <div class="glass-panel">
                <span>Telemetry Value</span>
            </div>
        """
        cleaned = clean_html_for_render(indented)
        for line in cleaned.splitlines():
            assert not line.startswith("    "), f"Line still has 4-space markdown indent: {line}"
            assert not line.startswith("\t"), f"Line still has tab indent: {line}"

    def test_clean_html_empty_input(self):
        assert clean_html_for_render("") == ""
        assert clean_html_for_render(None) == ""


class TestBadgesAndProvenance:
    """Verifies provenance and mode badges."""

    def test_provenance_badge_levels(self):
        obs_html = render_provenance_badge("OBSERVED")
        assert "badge-provenance-observed" in obs_html
        assert "OBSERVED" in obs_html

        der_html = render_provenance_badge("DERIVED")
        assert "badge-provenance-derived" in der_html
        assert "DERIVED" in der_html

        est_html = render_provenance_badge("ESTIMATED")
        assert "badge-provenance-estimated" in est_html
        assert "ESTIMATED" in est_html

        unavail_html = render_provenance_badge("UNAVAILABLE")
        assert "badge-provenance-unavailable" in unavail_html
        assert "UNAVAILABLE" in unavail_html

    def test_mode_badge_separation(self):
        demo_html = render_mode_badge("demo")
        assert "mode-pill-demo" in demo_html
        assert "DEMO / SIMULATION" in demo_html

        live_html = render_mode_badge("live")
        assert "mode-pill-live" in live_html
        assert "LIVE OPERATIONS" in live_html


class TestCandidateVesselCards:
    """Verifies candidate vessel cards use deterministic association ranking and neutral language."""

    def test_neutral_terminology_in_vessel_card(self, monkeypatch):
        """Card must not contain biased terms like 'guilty' or 'culprit'."""
        rendered_htmls = []

        def mock_render_html(html_str):
            rendered_htmls.append(html_str)

        monkeypatch.setattr("ui_components.render_html", mock_render_html)
        monkeypatch.setattr("streamlit.button", lambda *args, **kwargs: False)

        vessel = {
            "name": "MT OCEAN PIONEER",
            "mmsi": "413289000",
            "vessel_type": "Crude Oil Tanker",
            "score": 87.4,
            "distance_km": 4.2,
            "time_alignment": "Coincident (12m offset)",
            "trajectory": "Consistent with backtrack origin",
            "behavior": "Cargo discharge loiter",
        }

        render_candidate_vessel_card(rank=1, vessel=vessel)

        full_output = "".join(rendered_htmls)
        assert "MT OCEAN PIONEER" in full_output
        assert "413289000" in full_output
        assert "87.4%" in full_output
        assert "ASSOCIATION" in full_output

        # Prohibited terms
        assert "guilty" not in full_output.lower()
        assert "culprit" not in full_output.lower()
        assert "perpetrator" not in full_output.lower()
        assert "responsible party" not in full_output.lower()

    def test_candidate_list_deterministic_sorting(self, monkeypatch):
        """Vessels must be sorted by association score descending."""
        rendered_ranks = []

        def mock_card(rank, vessel, is_selected=False, key=None):
            rendered_ranks.append((rank, vessel["name"], vessel["score"]))
            return False

        monkeypatch.setattr("ui_components.render_candidate_vessel_card", mock_card)

        vessels = [
            {"name": "Vessel B", "mmsi": "222", "score": 45.0},
            {"name": "Vessel A", "mmsi": "111", "score": 92.5},
            {"name": "Vessel C", "mmsi": "333", "score": 71.0},
        ]

        render_candidate_vessel_list(vessels)

        assert len(rendered_ranks) == 3
        # Should be ordered A (92.5) -> C (71.0) -> B (45.0)
        assert rendered_ranks[0][1] == "Vessel A"
        assert rendered_ranks[1][1] == "Vessel C"
        assert rendered_ranks[2][1] == "Vessel B"


class TestSignalEvidenceTable:
    """Verifies evidence table renders quorum states without text walls."""

    def test_signal_evidence_rendering(self, monkeypatch):
        captured = []
        monkeypatch.setattr("ui_components.render_html", lambda s: captured.append(s))

        signals = [
            {"name": "YOLOv8 Segmentation", "status": "Validated", "reason": "Dark patch detected with high contrast", "metric": "IoU 0.82"},
            {"name": "Classical Damping", "status": "Supported", "reason": "Radar backscatter damping confirmed", "metric": "3.4 dB"},
            {"name": "Land Mask Filter", "status": "Passed", "reason": "Marine domain verified; 0% coastal overlap", "metric": "0.0%"},
            {"name": "Consensus Engine", "status": "Confirmed", "reason": "Multi-signal quorum achieved", "metric": "4/4 Quorum"},
        ]

        render_signal_evidence_table(signals)

        assert len(captured) == 1
        output = captured[0]
        assert "YOLOv8 Segmentation" in output
        assert "Classical Damping" in output
        assert "Land Mask Filter" in output
        assert "signal-badge-success" in output
        assert "signal-badge-cyan" in output


class TestEmptyStateIntegrity:
    """Verifies empty states are honest and never fabricate data."""

    def test_empty_state_rendering(self, monkeypatch):
        captured = []
        monkeypatch.setattr("ui_components.render_html", lambda s: captured.append(s))

        render_empty_state(
            title="NO AIS TELEMETRY AVAILABLE",
            description="Live AIS receiver stream is disconnected. Connect a coastal receiver or upload historical NMEA.",
        )

        assert len(captured) == 1
        out = captured[0]
        assert "NO AIS TELEMETRY AVAILABLE" in out
        assert "Live AIS receiver stream is disconnected" in out
        assert "empty-state-panel" in out
