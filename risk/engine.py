"""
Jal-Rakshak — Risk Engine
===========================
Explainable risk assessment for oil spill incidents.
Produces LOW / MEDIUM / HIGH / CRITICAL with factor breakdown.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskFactor:
    """Individual factor contributing to overall risk."""
    name: str
    value: float           # raw value
    score: float           # normalized 0-100
    weight: float          # weight in overall calculation
    description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "score": round(self.score, 1),
            "weight": self.weight,
            "description": self.description,
        }


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    level: RiskLevel
    overall_score: float       # 0-100
    factors: List[RiskFactor]
    recommendations: List[str]
    alert_required: bool
    alert_priority: str        # "none", "standard", "urgent", "immediate"

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "overall_score": round(self.overall_score, 1),
            "factors": [f.to_dict() for f in self.factors],
            "recommendations": self.recommendations,
            "alert_required": self.alert_required,
            "alert_priority": self.alert_priority,
        }


def assess_risk(
    spill_area_sq_km: float = 0.0,
    detection_confidence: float = 0.0,
    validation_confidence: float = 0.0,
    distance_to_coast_km: float = 999.0,
    eta_to_coast_hours: Optional[float] = None,
    sensitive_areas_nearby: int = 0,
    growth_rate_sq_km_per_hour: float = 0.0,
    source_confidence: float = 0.0,
    wind_speed_ms: float = 0.0,
    current_speed_ms: float = 0.0,
) -> RiskAssessment:
    """
    Calculate risk level from multiple factors.

    All factors are weighted and combined into an overall score.
    The score determines the risk level and alert priority.
    """
    factors = []

    # 1. Spill area (weight: 0.20)
    if spill_area_sq_km >= 10.0:
        area_score = 100
    elif spill_area_sq_km >= 5.0:
        area_score = 80
    elif spill_area_sq_km >= 1.0:
        area_score = 60
    elif spill_area_sq_km >= 0.1:
        area_score = 30
    else:
        area_score = 10
    factors.append(RiskFactor(
        "Spill Area", spill_area_sq_km, area_score, 0.20,
        f"{spill_area_sq_km:.3f} km² — {'large' if spill_area_sq_km >= 5 else 'moderate' if spill_area_sq_km >= 1 else 'small'} spill"
    ))

    # 2. Detection confidence (weight: 0.10)
    det_score = detection_confidence * 100
    factors.append(RiskFactor(
        "Detection Confidence", detection_confidence, det_score, 0.10,
        f"{detection_confidence:.2f} — {'high' if detection_confidence >= 0.8 else 'moderate' if detection_confidence >= 0.5 else 'low'}"
    ))

    # 3. Distance to coast (weight: 0.20)
    if distance_to_coast_km <= 5:
        coast_score = 100
    elif distance_to_coast_km <= 20:
        coast_score = 80
    elif distance_to_coast_km <= 50:
        coast_score = 50
    elif distance_to_coast_km <= 100:
        coast_score = 25
    else:
        coast_score = 5
    factors.append(RiskFactor(
        "Proximity to Coast", distance_to_coast_km, coast_score, 0.20,
        f"{distance_to_coast_km:.1f} km from nearest coastline"
    ))

    # 4. ETA to coast (weight: 0.15)
    if eta_to_coast_hours is not None:
        if eta_to_coast_hours <= 6:
            eta_score = 100
        elif eta_to_coast_hours <= 12:
            eta_score = 80
        elif eta_to_coast_hours <= 24:
            eta_score = 50
        elif eta_to_coast_hours <= 48:
            eta_score = 25
        else:
            eta_score = 5
        factors.append(RiskFactor(
            "ETA to Coast", eta_to_coast_hours, eta_score, 0.15,
            f"Estimated {eta_to_coast_hours:.1f} hours until coastal impact"
        ))
    else:
        factors.append(RiskFactor(
            "ETA to Coast", 0, 20, 0.15,
            "ETA unavailable — insufficient data for prediction"
        ))

    # 5. Sensitive areas (weight: 0.15)
    if sensitive_areas_nearby >= 3:
        sensitive_score = 100
    elif sensitive_areas_nearby >= 1:
        sensitive_score = 60
    else:
        sensitive_score = 10
    factors.append(RiskFactor(
        "Sensitive Areas", sensitive_areas_nearby, sensitive_score, 0.15,
        f"{sensitive_areas_nearby} sensitive area(s) within predicted impact zone"
    ))

    # 6. Growth rate (weight: 0.10)
    if growth_rate_sq_km_per_hour >= 1.0:
        growth_score = 100
    elif growth_rate_sq_km_per_hour >= 0.1:
        growth_score = 60
    else:
        growth_score = 10
    factors.append(RiskFactor(
        "Growth Rate", growth_rate_sq_km_per_hour, growth_score, 0.10,
        f"{growth_rate_sq_km_per_hour:.3f} km²/hr"
    ))

    # 7. Environmental conditions (weight: 0.10)
    env_score = min(100, (wind_speed_ms * 5) + (current_speed_ms * 50))
    factors.append(RiskFactor(
        "Environmental Conditions", wind_speed_ms, env_score, 0.10,
        f"Wind: {wind_speed_ms:.1f} m/s, Current: {current_speed_ms:.2f} m/s"
    ))

    # Overall score
    overall = sum(f.score * f.weight for f in factors)

    # Risk level
    if overall >= 75:
        level = RiskLevel.CRITICAL
        alert_priority = "immediate"
    elif overall >= 50:
        level = RiskLevel.HIGH
        alert_priority = "urgent"
    elif overall >= 30:
        level = RiskLevel.MEDIUM
        alert_priority = "standard"
    else:
        level = RiskLevel.LOW
        alert_priority = "none"

    alert_required = overall >= 30

    # Recommendations
    recommendations = []
    if level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
        recommendations.append("Alert coastal authorities immediately")
        recommendations.append("Deploy oil spill response teams")
        recommendations.append("Issue community advisory for affected coastal zones")
    if level == RiskLevel.CRITICAL:
        recommendations.append("Consider fishing ban in predicted impact zone")
        recommendations.append("Activate emergency response protocol")
    if distance_to_coast_km <= 20:
        recommendations.append("Monitor coastline for oil landfall")
    if sensitive_areas_nearby > 0:
        recommendations.append("Protect sensitive areas with boom deployment")

    return RiskAssessment(
        level=level,
        overall_score=overall,
        factors=factors,
        recommendations=recommendations,
        alert_required=alert_required,
        alert_priority=alert_priority,
    )
