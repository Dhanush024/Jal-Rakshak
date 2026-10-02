"""
Jal-Rakshak — UI/UX Architecture & State Isolation Tests
========================================================
Validates Section 41 non-negotiable acceptance criteria:
1. Live starts without Demo spill.
2. Demo starts with Demo spill.
3. Switching Live -> Demo works.
4. Switching Demo -> Live clears Demo layers.
5. Demo vessel does not appear in Live.
6. Demo drift does not appear in Live.
7. Demo timeline does not appear in Live.
8. Live provider unavailable does not load Demo data automatically.
9. Clicking logo navigates to Home.
10. Home -> Live works.
11. Home -> Demo works.
12. Internal Live routes work.
13. Internal Demo routes work.
14. No duplicated pipeline controls.
15. No raw HTML is visible.
"""

import sys
import os
import re
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.scenario import CHENNAI_SCENARIO


class TestStateIsolationArchitecture:
    """Verifies complete state isolation between Live and Demo modes."""

    def test_live_and_demo_states_are_independent(self):
        """Live and Demo states must be strictly separate dictionaries."""
        import app

        # Test helper function that initializes separate states
        live_st = app.create_initial_live_state()
        demo_st = app.create_initial_demo_state()

        assert isinstance(live_st, dict)
        assert isinstance(demo_st, dict)
        assert live_st is not demo_st

        # Mutating demo state must NOT affect live state
        demo_st["pipeline_result"] = {"spill_detected": True, "area": 2.41}
        assert live_st.get("pipeline_result") is None
        assert live_st.get("vessels") == []

    def test_live_starts_without_demo_spill(self):
        """When entering Live Operations, the map must be clean and not show Demo spill."""
        import app
        live_st = app.create_initial_live_state()

        # Build live map with initial live state
        fmap = app.build_investigation_map(state=live_st, is_demo=False)

        # Inspect Folium map children
        child_names = [getattr(c, "layer_name", getattr(c, "name", "")) for c in fmap._children.values()]
        html_content = fmap.get_root().render()

        # Must not contain demo spill or Chennai coordinates / scenario artifacts
        assert "Detected Oil Spill Polygon" not in html_content
        assert "Detected Oil Spill Boundary" not in html_content
        assert "▲ ORIGIN ESTIMATE" not in html_content
        assert "Progressive Reverse Drift Backtrack Trail" not in html_content
        assert "Predicted Forward Drift Trajectory" not in html_content
        assert "Marina Beach" not in html_content

    def test_demo_starts_with_demo_spill(self):
        """When in Demo mode, the map must include the calibrated demo scenario layers."""
        import app
        demo_st = app.create_initial_demo_state()
        # Ensure demo pipeline result is loaded
        demo_st["pipeline_result"] = {
            "spill_detected": True,
            "spill_lat": CHENNAI_SCENARIO.spill_lat,
            "spill_lon": CHENNAI_SCENARIO.spill_lon,
            "source_lat": CHENNAI_SCENARIO.spill_lat,
            "source_lon": CHENNAI_SCENARIO.spill_lon,
            "source_uncertainty_km": 5.0,
            "candidate_scores": [{"name": "MT Sea Falcon", "mmsi": "419000123", "score": 88.0, "is_suspect": True}],
            "ais_tracks": {"419000123": [{"lat": 12.45, "lon": 80.23, "speed_knots": 10.0, "name": "MT Sea Falcon"}]},
            "hindcast_done": True,
            "hindcast_result": {"current_speed_ms": 0.48, "current_bearing_deg": 118.0},
            "forecast_results": [
                {"hours": 6, "destination_lat": 12.48, "destination_lon": 80.29},
                {"hours": 12, "destination_lat": 12.52, "destination_lon": 80.35},
            ],
        }

        fmap = app.build_investigation_map(state=demo_st, is_demo=True)
        html_content = fmap.get_root().render()

        assert "[DEMO]" in html_content
        assert "ORIGIN" in html_content or "Origin" in html_content
        assert "Backtrack" in html_content or "Advection" in html_content

    def test_switching_modes_isolates_layers(self):
        """Switching Live -> Demo -> Live must cleanly reset layer presence."""
        import app
        live_st = app.create_initial_live_state()
        demo_st = app.create_initial_demo_state()
        demo_st["pipeline_result"] = {
            "spill_detected": True,
            "spill_lat": CHENNAI_SCENARIO.spill_lat,
            "spill_lon": CHENNAI_SCENARIO.spill_lon,
        }

        # 1. Start Live -> Clean
        fmap_live1 = app.build_investigation_map(state=live_st, is_demo=False)
        html_live1 = fmap_live1.get_root().render()
        assert "[DEMO]" not in html_live1
        assert "Detected Oil Spill" not in html_live1

        # 2. Switch to Demo -> Demo layers present
        fmap_demo = app.build_investigation_map(state=demo_st, is_demo=True)
        html_demo = fmap_demo.get_root().render()
        assert "[DEMO]" in html_demo

        # 3. Switch back to Live -> Clean again!
        fmap_live2 = app.build_investigation_map(state=live_st, is_demo=False)
        html_live2 = fmap_live2.get_root().render()
        assert "[DEMO]" not in html_live2
        assert "Detected Oil Spill" not in html_live2
        assert "Marina Beach" not in html_live2

    def test_demo_vessel_does_not_appear_in_live(self):
        """Suspect demo tanker MT Ocean Pioneer must not appear in Live state."""
        import app
        live_st = app.create_initial_live_state()
        live_vessels = live_st.get("vessels", [])
        mmsi_list = [v.get("mmsi") for v in live_vessels]
        assert "413289000" not in mmsi_list

    def test_demo_drift_does_not_appear_in_live(self):
        """Drift trajectory from demo scenario must not appear in Live map."""
        import app
        live_st = app.create_initial_live_state()
        fmap = app.build_investigation_map(state=live_st, is_demo=False)
        html_content = fmap.get_root().render()
        assert "Progressive Reverse Drift Backtrack Trail" not in html_content
        assert "Predicted Forward Drift Trajectory" not in html_content

    def test_live_provider_unavailable_does_not_load_demo_data(self):
        """When live AIS provider is not connected, it must stay empty instead of falling back to demo."""
        import app
        live_st = app.create_initial_live_state()
        assert live_st.get("provider_connected") is False
        assert live_st.get("vessels") == []
        assert live_st.get("pipeline_result") is None

    def test_live_connect_disconnect_provider(self):
        """Connecting live feed populates live vessels without touching demo state."""
        import app
        live_st = app.create_initial_live_state()
        demo_st = app.create_initial_demo_state()

        assert live_st.get("provider_connected") is False
        assert len(live_st.get("vessels", [])) == 0

        # Connect live feed
        app.connect_live_ais_feed(live_st)
        assert live_st.get("provider_connected") is True
        assert len(live_st.get("vessels", [])) > 0
        # Demo suspect vessel MT Ocean Pioneer must not be in live feed
        live_mmsis = [v.get("mmsi") for v in live_st["vessels"]]
        assert "413289000" not in live_mmsis
        # Demo state must remain unaffected
        assert demo_st.get("pipeline_result") is None

        # Disconnect live feed
        app.disconnect_live_ais_feed(live_st)
        assert live_st.get("provider_connected") is False
        assert len(live_st.get("vessels", [])) == 0


class TestNavigationAndRouting:
    """Verifies the global single-header navigation model and route resolution."""

    def test_route_resolution_hierarchy(self):
        """Tests route resolution helper across home, live, and demo."""
        import app

        # Home / landing
        r_home = app.resolve_route(mode=None, view=None)
        assert r_home["mode"] == "landing"

        r_landing = app.resolve_route(mode="landing", view=None)
        assert r_landing["mode"] == "landing"

        # Live operations default to map
        r_live_def = app.resolve_route(mode="live", view=None)
        assert r_live_def["mode"] == "live"
        assert r_live_def["view"] == "map"

        # Live operations subviews
        for v in ("map", "sar", "ais", "drift", "reports"):
            r_live = app.resolve_route(mode="live", view=v)
            assert r_live["mode"] == "live"
            assert r_live["view"] == v

        # Demo route resolution
        r_demo = app.resolve_route(mode="demo", view=None)
        assert r_demo["mode"] == "demo"
        assert r_demo["step"] in (1, 2, 3, 4, 5)

    def test_logo_navigation_targets_home(self):
        """Clicking JAL-RAKSHAK brand logo resolves to Home route."""
        import app
        home_route = app.get_logo_navigation_target()
        assert home_route == "landing"


class TestNoDuplicateControls:
    """Verifies that duplicate pipeline buttons and competing controls are eliminated."""

    def test_no_redundant_pipeline_buttons_in_codebase(self):
        """Ensures 'Run Pipeline Now', 'EXECUTE MULTI-NODE', etc. do not exist in app.py."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Prohibited duplicate strings
        prohibited = [
            "Run Pipeline Now",
            "EXECUTE MULTI-NODE INTELLIGENCE PIPELINE",
            "⚡ EXECUTE PIPELINE",
            "⚡ Run Intelligence Pipeline Now",
        ]
        for p in prohibited:
            assert p not in content, f"Found redundant pipeline control string: '{p}'"

    def test_canonical_action_used(self):
        """The canonical action 'RUN ANALYSIS' should be used for execution."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "RUN ANALYSIS" in content, "Canonical action 'RUN ANALYSIS' must be present in app.py"


class TestHTMLRenderingSafety:
    """Verifies that render_html safely strips line indentation to prevent raw HTML code blocks."""

    def test_render_html_strips_indented_lines(self):
        import app
        test_block = """
            <div class="test-class">
                <span>Safe Content</span>
            </div>
        """
        cleaned = app.clean_html_for_render(test_block)
        for line in cleaned.splitlines():
            assert not line.startswith("    "), f"Indented line would trigger markdown code block: {line}"
            assert not line.startswith("\t")


class TestInternalUIPolish:
    """Verifies emoji removal on internal navigation and Home page preservation."""

    def test_home_page_function_intact(self):
        """Home page renderer render_landing_screen must remain present and intact."""
        import app
        assert hasattr(app, "render_landing_screen")
        assert callable(app.render_landing_screen)

    def test_internal_navigation_avoids_decorative_emojis(self):
        """Header internal buttons must not use decorative emojis like 🏠, 🛰️, 🧪, 🎯, 📄."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Prohibited UI button emojis in internal header
        forbidden_button_patterns = [
            'st.button("🏠',
            'st.button("🛰️',
            'st.button("🧪',
            'st.button("🎯',
            'st.button("📄',
            'st.button("🚢',
            'st.button("⚙️',
        ]
        for p in forbidden_button_patterns:
            assert p not in content, f"Found forbidden emoji in button label: {p}"


class TestPhase1StateAndInteraction:
    """Validates Phase 1 bug fixes: AIS state consistency, SAR layer control, and neutral language."""

    def test_no_candidates_done_trap_in_app(self):
        """candidates_done was a non-existent state key that caused AIS state inconsistency."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "candidates_done" not in content, "candidates_done must not be used in app.py"

    def test_ais_banner_uses_neutral_terminology(self):
        """Must not use 'Suspect' in AIS header metrics; prefer 'Candidate'."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert '"1 Suspect"' not in content, "Hardcoded '1 Suspect' must not appear in app.py"

    def test_sar_layer_names_not_truncated(self):
        """SAR layer labels must include COMPOSITE, RAW, YOLO, LAND, OCEAN, CLASSICAL, CONSENSUS, FINAL."""
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        for layer_label in ["COMPOSITE", "RAW", "YOLO", "LAND", "OCEAN", "CLASSICAL", "CONSENSUS", "FINAL"]:
            assert f'"{layer_label}"' in content or f"'{layer_label}'" in content

    def test_segmented_layer_control_on_click_support(self):
        """render_segmented_layer_control must support on_change_state_key and callbacks."""
        from ui_components import render_segmented_layer_control
        import inspect
        sig = inspect.signature(render_segmented_layer_control)
        assert "on_change_state_key" in sig.parameters


