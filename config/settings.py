"""
Jal-Rakshak — Centralized Configuration
=========================================
All configurable parameters read from environment variables.
Never hard-code API keys, tokens, or secrets in source code.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env file if python-dotenv is available
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on system env vars


# ---------------------------------------------------------------------------
# Application Mode
# ---------------------------------------------------------------------------
APP_MODE = os.getenv("APP_MODE", "demo").lower()  # "demo" or "live"

# ---------------------------------------------------------------------------
# AIS Provider
# ---------------------------------------------------------------------------
AISSTREAM_API_KEY = os.getenv("AISSTREAM_API_KEY", "")

# ---------------------------------------------------------------------------
# SAR Configuration
# ---------------------------------------------------------------------------
DEFAULT_PIXEL_RESOLUTION_M = float(os.getenv("DEFAULT_PIXEL_RESOLUTION_M", "10.0"))

# Model path — relative to project root
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")

# ---------------------------------------------------------------------------
# Ocean / Weather (reserved for future providers)
# ---------------------------------------------------------------------------
OCEAN_DATA_API_KEY = os.getenv("OCEAN_DATA_API_KEY", "")

# ---------------------------------------------------------------------------
# Notification (reserved for future providers)
# ---------------------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMS_API_KEY = os.getenv("SMS_API_KEY", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Demo Configuration
# ---------------------------------------------------------------------------
DEMO_SPILL_LAT = float(os.getenv("DEMO_SPILL_LAT", "12.4500"))
DEMO_SPILL_LON = float(os.getenv("DEMO_SPILL_LON", "80.2300"))

# ---------------------------------------------------------------------------
# Drift Model Defaults
# ---------------------------------------------------------------------------
DEFAULT_CURRENT_SPEED_MS = float(os.getenv("DEFAULT_CURRENT_SPEED_MS", "0.48"))
DEFAULT_CURRENT_BEARING_DEG = float(os.getenv("DEFAULT_CURRENT_BEARING_DEG", "118.0"))
DEFAULT_WIND_FACTOR = float(os.getenv("DEFAULT_WIND_FACTOR", "0.03"))
DEFAULT_WIND_SPEED_MS = float(os.getenv("DEFAULT_WIND_SPEED_MS", "6.2"))
DEFAULT_WIND_BEARING_DEG = float(os.getenv("DEFAULT_WIND_BEARING_DEG", "135.0"))

# ---------------------------------------------------------------------------
# Scoring Weights (vessel attribution)
# ---------------------------------------------------------------------------
SCORE_WEIGHT_TEMPORAL = float(os.getenv("SCORE_WEIGHT_TEMPORAL", "0.30"))
SCORE_WEIGHT_SPATIAL = float(os.getenv("SCORE_WEIGHT_SPATIAL", "0.25"))
SCORE_WEIGHT_TRAJECTORY = float(os.getenv("SCORE_WEIGHT_TRAJECTORY", "0.20"))
SCORE_WEIGHT_HEADING = float(os.getenv("SCORE_WEIGHT_HEADING", "0.10"))
SCORE_WEIGHT_SPEED = float(os.getenv("SCORE_WEIGHT_SPEED", "0.05"))
SCORE_WEIGHT_DRIFT = float(os.getenv("SCORE_WEIGHT_DRIFT", "0.10"))


def is_demo_mode() -> bool:
    """Check if the application is running in demo mode."""
    return APP_MODE == "demo"


def is_live_mode() -> bool:
    """Check if the application is running in live mode."""
    return APP_MODE == "live"


def validate_config():
    """Validate critical configuration and return warnings."""
    warnings = []
    if is_live_mode() and not AISSTREAM_API_KEY:
        warnings.append("AISSTREAM_API_KEY not set — AIS live data unavailable")
    if not Path(MODEL_PATH).exists():
        warnings.append(f"YOLO model file not found: {MODEL_PATH}")
    return warnings
