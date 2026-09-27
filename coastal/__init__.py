"""
Jal-Rakshak — Coastal & Environmental Impact Assessment Package
=================================================================
Calculates shoreline proximity, trajectory interception, ETA to landfall,
Environmental Sensitivity Index (ESI), and protective countermeasures.
"""

from coastal.zones import (
    EnvironmentalSensitivityIndex,
    SensitiveArea,
    CoastalZone,
    CHENNAI_CORRIDOR_SHORELINE,
    CHENNAI_SENSITIVE_AREAS,
    get_nearest_shoreline_point,
)
from coastal.impact import (
    CoastalImpactResult,
    assess_coastal_impact,
)

__all__ = [
    "EnvironmentalSensitivityIndex",
    "SensitiveArea",
    "CoastalZone",
    "CHENNAI_CORRIDOR_SHORELINE",
    "CHENNAI_SENSITIVE_AREAS",
    "get_nearest_shoreline_point",
    "CoastalImpactResult",
    "assess_coastal_impact",
]
