"""Tests for concrete provider implementations: AIS, Ocean, SAR, and Notification providers."""

import pytest
import os
import tempfile
import numpy as np
from datetime import datetime, timezone, timedelta

from ais.provider import DemoAISProvider, FileHistoricalAISProvider, get_ais_provider, DataMode
from ocean.provider import DemoOceanProvider, ConstantOceanProvider, get_ocean_provider
from sar.ingestion import (
    SARSceneLoader,
    SARSceneMetadata,
    DemoSatelliteProvider,
    FileSatelliteProvider,
    get_satellite_provider,
)
from alerts.manager import (
    Alert,
    AlertChannel,
    AlertPriority,
    AlertTarget,
    SimulationNotificationProvider,
    ConsoleNotificationProvider,
    get_notification_provider,
)


class TestAISProviders:
    """Test AIS provider implementations."""

    def test_demo_ais_provider_returns_tracks(self):
        provider = DemoAISProvider()
        assert provider.mode == DataMode.DEMO

        bbox = (12.8, 80.0, 13.5, 80.6)
        res = provider.get_historical_tracks(bbox=bbox)
        assert res.mode == DataMode.DEMO
        assert len(res.tracks) >= 3
        for mmsi, records in res.tracks.items():
            assert len(records) > 0
            assert records[0].mmsi == mmsi

    def test_file_historical_ais_provider_loads_csv(self):
        sample_csv = os.path.join(os.path.dirname(__file__), "..", "data", "sample_historical_ais.csv")
        assert os.path.exists(sample_csv)

        provider = FileHistoricalAISProvider(sample_csv)
        assert provider.mode == DataMode.REAL

        bbox = (12.5, 80.0, 13.5, 80.6)
        start = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
        end = datetime(2026, 9, 14, 16, 0, tzinfo=timezone.utc)

        res = provider.get_historical_tracks(bbox=bbox, time_start=start, time_end=end)
        assert res.mode == DataMode.REAL
        assert len(res.tracks) >= 1
        mmsis = list(res.tracks.keys())
        assert "419001234" in mmsis or "419005678" in mmsis

    def test_file_historical_ais_empty_on_missing_file(self):
        provider = FileHistoricalAISProvider("non_existent_file.csv")
        res = provider.get_historical_tracks()
        assert res.mode == DataMode.UNAVAILABLE
        assert len(res.tracks) == 0

    def test_ais_factory_mode_switching(self):
        p_demo = get_ais_provider("DEMO")
        assert isinstance(p_demo, DemoAISProvider)
        assert p_demo.mode == DataMode.DEMO

        sample_csv = os.path.join(os.path.dirname(__file__), "..", "data", "sample_historical_ais.csv")
        p_real = get_ais_provider("REAL", file_path=sample_csv)
        assert isinstance(p_real, FileHistoricalAISProvider)
        assert p_real.mode == DataMode.REAL


class TestOceanProviders:
    """Test Ocean provider implementations."""

    def test_demo_ocean_provider(self):
        provider = DemoOceanProvider()
        now = datetime.now(timezone.utc)
        currents = provider.get_currents(13.1, 80.3, now)
        winds = provider.get_wind(13.1, 80.3, now)

        assert len(currents) > 0
        assert currents[0].speed_ms > 0
        assert len(winds) > 0
        assert winds[0].speed_ms > 0

    def test_constant_ocean_provider(self):
        provider = ConstantOceanProvider(
            current_speed_ms=0.5,
            current_bearing_deg=45.0,
            wind_speed_ms=8.0,
            wind_bearing_deg=180.0,
            is_real_source=True,
        )
        now = datetime.now(timezone.utc)
        curr = provider.get_currents(10.0, 70.0, now)
        assert len(curr) == 1
        assert curr[0].speed_ms == 0.5
        assert curr[0].direction_deg == 45.0
        assert provider.mode.value == "REAL"

    def test_ocean_factory(self):
        p1 = get_ocean_provider("DEMO")
        assert isinstance(p1, DemoOceanProvider)
        p2 = get_ocean_provider("REAL")
        assert isinstance(p2, ConstantOceanProvider)
        assert p2.mode.value == "REAL"


class TestSARIngestionAndProviders:
    """Test SAR Ingestion and providers."""

    def test_sar_scene_loader_synthetic(self):
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(temp_path, test_img)

            img, meta = SARSceneLoader.load(
                temp_path,
                override_lat=13.1,
                override_lon=80.35,
                pixel_res_m=10.0,
            )

            assert img.shape == (100, 100, 3)
            assert meta.height == 100
            assert meta.width == 100
            assert meta.center_lat == 13.1
            assert meta.center_lon == 80.35

            # Test coordinate conversion
            lat, lon = SARSceneLoader.pixel_to_geo(50, 50, meta)
            assert abs(lat - 13.1) < 0.01
            assert abs(lon - 80.35) < 0.01

            py, px = SARSceneLoader.geo_to_pixel(lat, lon, meta)
            assert abs(py - 50) <= 1
            assert abs(px - 50) <= 1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_satellite_providers(self):
        p_demo = get_satellite_provider("DEMO")
        assert isinstance(p_demo, DemoSatelliteProvider)

        file_path, meta = p_demo.acquire_scene(13.1, 80.35)
        assert os.path.exists(file_path)
        assert "Sentinel-1" in meta.sensor


class TestNotificationProviders:
    """Test Notification providers."""

    def test_simulation_notification_provider(self):
        provider = SimulationNotificationProvider()
        alert = Alert(
            alert_id="ALERT-001",
            timestamp=datetime.now(timezone.utc),
            priority=AlertPriority.URGENT,
            target=AlertTarget.AUTHORITIES,
            channels=[AlertChannel.DASHBOARD],
            title="High risk oil spill near Chennai",
            message="Detection confirmed by SAR model",
            location_description="Ennore Port approach",
        )
        assert provider.send(alert) is True

    def test_console_notification_provider(self):
        provider = ConsoleNotificationProvider()
        alert = Alert(
            alert_id="ALERT-002",
            timestamp=datetime.now(timezone.utc),
            priority=AlertPriority.INFO,
            target=AlertTarget.FISHERMEN,
            channels=[AlertChannel.CONSOLE],
            title="Spill Advisory",
            message="Avoid fishing near sector 4",
            location_description="Chennai coastal waters",
        )
        assert provider.send(alert) is True

    def test_notification_factory(self):
        p_sim = get_notification_provider(AlertChannel.DASHBOARD)
        assert p_sim is not None
        p_cons = get_notification_provider(AlertChannel.CONSOLE)
        assert isinstance(p_cons, ConsoleNotificationProvider)


class TestConfigurationAndMaps:
    """Test map basemap configuration and API key handling."""

    def test_default_basemap_is_openstreetmap_without_key(self):
        from config.settings import MAP_BASEMAP, CARTO_API_KEY
        # Default must not require any API key
        assert MAP_BASEMAP == "OpenStreetMap"
        assert CARTO_API_KEY == "" or isinstance(CARTO_API_KEY, str)

    def test_validate_config_demo_mode_clean(self):
        from config.settings import validate_config
        # Demo mode should not require live API keys
        warnings = validate_config()
        # Should not raise exception
        assert isinstance(warnings, list)


class TestGracefulDegradation:
    """Test graceful handling of corrupt, missing, or malformed inputs."""

    def test_yolo_detector_missing_image(self):
        from sar.detection import YOLODetector
        detector = YOLODetector()
        res = detector.detect("non_existent_sar_image_path.jpg")
        assert res.spill_detected is False
        assert len(res.detections) == 0

    def test_sar_scene_loader_missing_image(self):
        from sar.ingestion import SARSceneLoader
        with pytest.raises(FileNotFoundError):
            SARSceneLoader.load("non_existent_sar_image_path.tif")

